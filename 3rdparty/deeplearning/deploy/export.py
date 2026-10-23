"""Split-module ONNX export + trtexec engine build for BEVFusion det / seg.

【部署前置步骤】Python 与 C++ 推理共用：导出分模块 ONNX -> trtexec 构建
TRT engine。部署只走 C++（deploy/cpp/）；本脚本与 export_bf.py 负责导出。

Follows the CUDA-BEVFusion (NVIDIA-AI-IOT/Lidar_AI_Solution) module split so
every exported graph is dense and TRT-parseable without custom plugins.

ONNX export is immediately followed by the trtexec engine build for every
graph (skip with --onnx-only). trtexec lookup: $TRTEXEC env, then PATH.

Usage (run from the model repo root 3rdparty/deeplearning):
  python deploy/export.py CONFIG CHECKPOINT --task det --verify --fp16
"""

import argparse
import os
import os.path as osp
import sys

# deeplearning 根（上 2 级）：models/datasets/scripts 与 deploy.python 包都在其下
_DL_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, _DL_ROOT)

import numpy as np
import torch
from torch import nn

from models import build_model
from scripts.config import load_config
from scripts.trainer import load_model_checkpoint

from deploy.python.graphs import DET_HEAD_KEYS, GRAPH_IO


# ------------------------------------------------------------------ wrappers
class CamBackbone(nn.Module):
    """Swin-T + GeneralizedLSSFPN -> (B, 6, 256, 32, 88)."""

    def __init__(self, model):
        super().__init__()
        self.backbone = model.encoders["camera"]["backbone"]
        self.neck = model.encoders["camera"]["neck"]

    def forward(self, img):
        B, N, C, H, W = img.size()
        x = img.view(B * N, C, H, W)
        x = self.backbone(x)
        x = self.neck(x)
        if not torch.is_tensor(x):
            x = x[0]
        BN, C, H, W = x.size()
        return x.view(B, N, C, H, W)


class CamVtransformDet(nn.Module):
    """DepthLSSTransform depthnet: (feats, depth) -> (B*N, D+C, fH, fW)."""

    def __init__(self, model):
        super().__init__()
        self.vt = model.encoders["camera"]["vtransform"]

    def forward(self, feats, depth):
        B, N, C, fH, fW = feats.size()
        x = feats.view(B * N, C, fH, fW)
        d = depth.view(B * N, *depth.shape[2:])
        x = self.vt.dtransform(d)
        x = torch.cat([x, feats.view(B * N, C, fH, fW)], dim=1)
        return self.vt.depthnet(x)


class CamVtransformSeg(nn.Module):
    """LSSTransform depthnet: feats -> (B*N, D+C, fH, fW)."""

    def __init__(self, model):
        super().__init__()
        self.vt = model.encoders["camera"]["vtransform"]

    def forward(self, feats):
        B, N, C, fH, fW = feats.size()
        x = feats.view(B * N, C, fH, fW)
        return self.vt.depthnet(x)


class FuserDecoder(nn.Module):
    """ConvFuser + SECOND + SECONDFPN -> (B, 512, H, W)."""

    def __init__(self, model):
        super().__init__()
        self.fuser = model.fuser
        self.decoder = model.decoder

    def forward(self, cam_bev, lidar_bev):
        x = self.fuser([cam_bev, lidar_bev])
        x = self.decoder["backbone"](x)
        x = self.decoder["neck"](x)
        return x[0]


class HeadBbox(nn.Module):
    """TransFusionHead raw pred tensors + query_labels (decode stays in
    postprocess). query_labels comes from the in-graph topk and is emitted
    as float32 so the TRT runtime (float-only buffers) can carry it."""

    def __init__(self, model):
        super().__init__()
        self.head = model.heads["object"]
        self.head.onnx_export = True  # unfold/amax instead of max_pool2d

    def forward(self, x):
        pd = self.head([x], [{}])
        d = pd[0][0]
        return (*(d[k] for k in DET_HEAD_KEYS[:-1]),
                self.head.query_labels.float())


class HeadMap(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.head = model.heads["map"]

    def forward(self, x):
        return self.head([x])


WRAPPERS = {
    "det": {
        "camera.backbone": (CamBackbone, ["img"], ["feats"]),
        "camera.vtransform": (CamVtransformDet, ["feats", "depth"], ["depthnet"]),
        "fuser": (FuserDecoder, ["cam_bev", "lidar_bev"], ["decoder_feat"]),
        "head.bbox": (HeadBbox, ["decoder_feat"], DET_HEAD_KEYS),
    },
    "seg": {
        "camera.backbone": (CamBackbone, ["img"], ["feats"]),
        "camera.vtransform": (CamVtransformSeg, ["feats"], ["depthnet"]),
        "fuser": (FuserDecoder, ["cam_bev", "lidar_bev"], ["decoder_feat"]),
        "head.map": (HeadMap, ["decoder_feat"], ["masks"]),
    },
}


def graph_args(name, inp):
    """Map a sampled-input dict to the positional args of a graph wrapper."""
    if name == "camera.backbone":
        return (inp["img"],)
    if name == "camera.vtransform":
        if inp["depth"] is not None:
            return (inp["feats"], inp["depth"])
        return (inp["feats"],)
    if name == "fuser":
        return (inp["cam_bev"], inp["lidar_bev"])
    return (inp["decoder_feat"],)


# ------------------------------------------------------------------ helpers
def _bn_snapshot(model):
    state = {}
    for name, m in model.named_modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d,
                          torch.nn.BatchNorm3d)):
            state[name] = (m.running_mean.clone(), m.running_var.clone(),
                           m.num_batches_tracked.clone(), m.training)
    return state


def _bn_restore(model, state):
    for name, m in model.named_modules():
        if name in state:
            rm, rv, nbt, tr = state[name]
            m.running_mean.copy_(rm)
            m.running_var.copy_(rv)
            m.num_batches_tracked.copy_(nbt)
            m.training = tr
    model.eval()


def _sample_inputs(cfg, model, task):
    """合成输入（Python 部署已移除；verify 仅做导出数值对拍）。

    全部使用固定形状的随机/单位张量：verify 只对比 onnx 与 torch 在
    同一输入下的输出相对差，不要求真实样本（不再依赖 deploy.python 推理）。
    """
    dev, dt = "cuda", torch.float16
    img = torch.randn(1, 6, 3, 256, 704, device=dev, dtype=dt)
    with torch.no_grad():
        feats = CamBackbone(model)(img)
        if task == "det":
            depth = torch.rand(1, 6, 1, 256, 704, device=dev, dtype=dt)
            depthnet = CamVtransformDet(model)(feats, depth)
        else:
            depth = None
            depthnet = CamVtransformSeg(model)(feats)
        eye4 = torch.eye(4, device=dev, dtype=torch.float32)
        batch = {
            "camera2lidar": eye4.unsqueeze(0).repeat(6, 1, 1),
            "camera_intrinsics": eye4.unsqueeze(0).repeat(6, 1, 1),
            "img_aug_matrix": eye4.unsqueeze(0).repeat(6, 1, 1),
            "lidar_aug_matrix": eye4.unsqueeze(0).repeat(6, 1, 1),
        }
        cam_bev = torch.randn(1, 80, 360, 360, device=dev, dtype=dt)
        lidar_bev = torch.randn(1, 256, 180, 180, device=dev, dtype=dt)
        decoder_feat = FuserDecoder(model)(cam_bev, lidar_bev)

    return dict(batch=batch, sample=None, img=img, feats=feats, depth=depth,
                depthnet=depthnet, cam_bev=cam_bev, lidar_bev=lidar_bev,
                decoder_feat=decoder_feat)


class OnnxExporter:
    """Export the module-split ONNX graphs and verify them against torch."""

    def __init__(self, model, task, outdir, opset=17):
        self.model = model
        self.task = task
        self.outdir = outdir
        self.opset = opset
        self.wrappers = {
            name: WRAPPERS[task][name][0](model)
            for name in GRAPH_IO[task]
        }
        self.in_names = {name: io[0] for name, io in GRAPH_IO[task].items()}
        self.out_names = {name: io[1] for name, io in GRAPH_IO[task].items()}

    def export(self, inp):
        """Write one ONNX per graph; returns {graph: path}."""
        # torch.onnx.export tracing flips BatchNorm into train mode and
        # mutates running stats; snapshot/restore keeps torch refs valid.
        snap = _bn_snapshot(self.model)
        paths = {}
        for name in self.wrappers:
            path = osp.join(self.outdir, f"{name}.onnx")
            with torch.no_grad():
                torch.onnx.export(
                    self.wrappers[name], graph_args(name, inp), path,
                    opset_version=self.opset,
                    input_names=self.in_names[name],
                    output_names=self.out_names[name],
                    do_constant_folding=True,
                )
            paths[name] = path
        _bn_restore(self.model, snap)
        return paths

    def verify(self, inp, paths):
        """ONNX Runtime parity + onnx.checker for every exported graph."""
        import onnx
        import onnxruntime as ort

        ok = True
        for name in self.wrappers:
            sess = ort.InferenceSession(paths[name],
                                        providers=["CPUExecutionProvider"])
            feeds = {n: a.detach().cpu().numpy()
                     for n, a in zip(self.in_names[name],
                                     graph_args(name, inp))}
            outs = sess.run(None, feeds)
            with torch.no_grad():
                refs = self.wrappers[name](*graph_args(name, inp))
            refs = refs if isinstance(refs, (tuple, list)) else (refs,)
            worst = 0.0
            for o, r in zip(outs, refs):
                rn = r.cpu().numpy()
                d = float(np.abs(o - rn).max())
                worst = max(worst, d / max(1.0, float(np.abs(rn).max())))
            tol = 5e-3
            ok &= worst < tol
            print(f"[verify] {name}: worst rel diff = {worst:.3e} "
                  f"(tol {tol:.0e})")
        for p in paths.values():
            onnx.checker.check_model(onnx.load(p))
        print("[verify] onnx.checker OK")
        assert ok, "ONNX parity failed"

    def run(self, cfg, verify=True):
        """Sample inputs -> export all graphs -> optional verify."""
        inp = _sample_inputs(cfg, self.model, self.task)
        paths = self.export(inp)
        print("[export] onnx:", paths)
        if verify:
            self.verify(inp, paths)
        return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config")
    ap.add_argument("checkpoint")
    ap.add_argument("--task", default="det", choices=["det", "seg"])
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--outdir", default=osp.join("outputs", "onnx"))
    ap.add_argument("--engine-dir", default=osp.join("outputs", "trt"),
                    help="TRT engine output dir; built right after export")
    ap.add_argument("--fp16", action="store_true",
                    help="build engines with FP16 (weak typing) / mixed-FP16 "
                         "via --autocast-fp16 (strong typing)")
    ap.add_argument("--autocast-fp16", action="store_true",
                    help="strong typing: ModelOpt-AutoCast the ONNX to mixed "
                         "FP16 before building (requires nvidia-modelopt)")
    ap.add_argument("--onnx-only", action="store_true",
                    help="export ONNX only; skip the trtexec engine build")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--options", nargs="*", default=[])
    args = ap.parse_args()

    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    cfg = load_config(args.config, options=args.options)
    model = build_model(cfg.model).cuda().eval()
    load_model_checkpoint(model, args.checkpoint, strict=True)

    os.makedirs(args.outdir, exist_ok=True)
    paths = OnnxExporter(model, args.task, args.outdir,
                         opset=args.opset).run(cfg, verify=args.verify)

    if not args.onnx_only:
        # engine 构建统一走 CUDA-BEVFusion 原版脚本（--fp16 + IO fp16:chw）：
        print("[export] engine 构建（deploy/build_trt_engine.sh，Python/C++ 共用）：")
        print("  cd 3rdparty/deeplearning && DEBUG_MODEL=<name> DEBUG_PRECISION=fp16 "
              "bash deploy/build_trt_engine.sh")


if __name__ == "__main__":
    main()
