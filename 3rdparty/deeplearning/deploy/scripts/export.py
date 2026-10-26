"""onnx 阶段：BEVFusion det 分模块 ONNX 导出（5 张图）。

用法（3rdparty/deeplearning 下）：
  python deploy/scripts/export.py CONFIG CHECKPOINT [--outdir outputs/onnx] [--verify]
  构建 engine：DEBUG_MODEL=resnet50 DEBUG_PRECISION=fp16 bash deploy/export_trt.sh

图：camera.backbone / camera.vtransform / fuser / head.bbox /
    lidar.backbone.xyz（SCN，libspconv 解析；BN+ReLU 必须融合进卷积）。
"""

import argparse
import os
import os.path as osp
import sys

_HERE = osp.dirname(osp.abspath(__file__))
_DEPLOY_ROOT = osp.dirname(_HERE)
_DL_ROOT = osp.dirname(_DEPLOY_ROOT)
sys.path.insert(0, _DL_ROOT)
sys.path.insert(0, _DEPLOY_ROOT)

import numpy as np
import torch
from torch import nn

BF_IO = {
    "camera.backbone": (["img", "depth"], ["camera_feature", "camera_depth_weights"]),
    "camera.vtransform": (["feat_in"], ["feat_out"]),
    "fuser": (["camera", "lidar"], ["middle"]),
    "head.bbox": (["middle"], ["score", "rot", "dim", "reg", "height", "vel"]),
}
SCN_ONNX_NAME = "lidar.backbone.xyz.onnx"


# ------------------------------------------------------------------ wrappers
class CamBackbone(nn.Module):
    """backbone + neck -> (B, N, C, fH, fW) feats。"""

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


class CamBackboneBf(nn.Module):
    """(img, depth) -> camera_feature (B,N,fH,fW,C) + camera_depth_weights (B,N,D,fH,fW)。"""

    def __init__(self, model):
        super().__init__()
        self.backbone = model.encoders["camera"]["backbone"]
        self.neck = model.encoders["camera"]["neck"]
        self.vt = model.encoders["camera"]["vtransform"]
        self.D = self.vt.D
        self.C = self.vt.C

    def forward(self, img, depth):
        B, N, C, H, W = img.size()
        x = img.view(B * N, C, H, W)
        x = self.backbone(x)
        x = self.neck(x)
        if not torch.is_tensor(x):
            x = x[0]
        BN, C, fH, fW = x.size()
        feat = x.view(B, N, C, fH, fW)

        d = depth.view(B * N, *depth.shape[2:])
        d = self.vt.dtransform(d)
        x = torch.cat([d, feat.view(B * N, C, fH, fW)], dim=1)
        x = self.vt.depthnet(x)  # (B*N, D+C, fH, fW)
        x = x.view(B, N, -1, fH, fW)
        depth_weights = x[:, :, : self.D].softmax(dim=2)      # (B,N,D,fH,fW)
        feature = x[:, :, self.D : self.D + self.C].permute(0, 1, 3, 4, 2)
        return feature, depth_weights                        # (B,N,fH,fW,C)


class CamVtransformDet(nn.Module):
    """DepthLSSTransform depthnet: (feats, depth) -> (B*N, D+C, fH, fW)。"""

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


class DownsampleBf(nn.Module):
    """bev_pool 输出 (1,80,360,360) -> (1,80,180,180)（= vtransform.downsample）。"""

    def __init__(self, model):
        super().__init__()
        self.downsample = model.encoders["camera"]["vtransform"].downsample

    def forward(self, feat_in):
        return self.downsample(feat_in)


class FuserDecoder(nn.Module):
    """ConvFuser + SECOND + SECONDFPN -> (B, 512, H, W)。"""

    def __init__(self, model):
        super().__init__()
        self.fuser = model.fuser
        self.decoder = model.decoder

    def forward(self, cam_bev, lidar_bev):
        x = self.fuser([cam_bev, lidar_bev])
        x = self.decoder["backbone"](x)
        x = self.decoder["neck"](x)
        return x[0]


class HeadBboxBf(nn.Module):
    """middle -> (score, rot, dim, reg, height, vel)，decode 留在 C++。"""

    def __init__(self, model):
        super().__init__()
        self.head = model.heads["object"]
        self.head.onnx_export = True  # unfold/amax 代替 max_pool2d
        self.num_classes = self.head.num_classes

    def forward(self, x):
        pd = self.head.forward_single(x, None, None)[0]
        one_hot = (
            torch.nn.functional.one_hot(
                self.head.query_labels.long(), self.num_classes
            )
            .permute(0, 2, 1)
            .float()
        )
        score = pd["heatmap"].sigmoid() * pd["query_heatmap_score"] * one_hot
        return score, pd["rot"], pd["dim"], pd["center"], pd["height"], pd["vel"]


class HeadMap(nn.Module):
    """旧 seg head.map（保留；部署链不再导出此图）。"""

    def __init__(self, model):
        super().__init__()
        self.head = model.heads["map"]

    def forward(self, x):
        return self.head([x])


WRAPPERS = {
    "camera.backbone": (CamBackboneBf, BF_IO["camera.backbone"][0], BF_IO["camera.backbone"][1]),
    "camera.vtransform": (DownsampleBf, BF_IO["camera.vtransform"][0], BF_IO["camera.vtransform"][1]),
    "fuser": (FuserDecoder, BF_IO["fuser"][0], BF_IO["fuser"][1]),
    "head.bbox": (HeadBboxBf, BF_IO["head.bbox"][0], BF_IO["head.bbox"][1]),
}


def graph_args(name, inp):
    """Map a sampled-input dict to the positional args of a graph wrapper（BF 结构）。"""
    if name == "camera.backbone":
        return (inp["img"], inp["depth"])
    if name == "camera.vtransform":
        return (inp["bev360"],)
    if name == "fuser":
        return (inp["cam_bev"], inp["lidar_bev"])
    return (inp["middle"],)  # head.bbox


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


def _sample_inputs(cfg, model):
    """固定形状的合成输入，供 --verify 对比 onnx 与 torch 输出。"""
    dev, dt = "cuda", torch.float32  # onnx 为 fp32；TRT 由 trtexec --fp16 转 fp16
    img = torch.randn(1, 6, 3, 256, 704, device=dev, dtype=dt)
    with torch.no_grad():
        feats = CamBackbone(model)(img)
        depth = torch.rand(1, 6, 1, 256, 704, device=dev, dtype=dt)
        depthnet = CamVtransformDet(model)(feats, depth)
        eye4 = torch.eye(4, device=dev, dtype=torch.float32).view(1, 1, 4, 4)
        batch = {
            "camera2lidar": eye4.repeat(1, 6, 1, 1),
            "camera_intrinsics": eye4.repeat(1, 6, 1, 1),
            "img_aug_matrix": eye4.repeat(1, 6, 1, 1),
            "lidar_aug_matrix": eye4[0],
        }
        lidar_bev = torch.randn(1, 256, 180, 180, device=dev, dtype=dt)
        vt = model.encoders["camera"]["vtransform"]
        B, N = img.shape[0], img.shape[1]
        D, C = vt.D, vt.C
        fH, fW = vt.feature_size
        d = depthnet[:, :D].softmax(dim=1).view(B, N, D, fH, fW)
        ft = depthnet[:, D:].view(B, N, C, fH, fW)
        x = (d.unsqueeze(2) * ft.unsqueeze(3))
        x = x.permute(0, 1, 3, 4, 5, 2).contiguous()
        geom = vt.get_geometry(
            batch["camera2lidar"][..., :3, :3],
            batch["camera2lidar"][..., :3, 3],
            batch["camera_intrinsics"][..., :3, :3],
            batch["img_aug_matrix"][..., :3, :3],
            batch["img_aug_matrix"][..., :3, 3],
            extra_rots=batch["lidar_aug_matrix"][..., :3, :3],
            extra_trans=batch["lidar_aug_matrix"][..., :3, 3],
        )
        bev360 = vt.bev_pool(geom, x)
        cam_bev = vt.downsample(bev360)
        middle = FuserDecoder(model)(cam_bev, lidar_bev)

    return dict(batch=batch, sample=None, img=img, feats=feats, depth=depth,
                depthnet=depthnet, cam_bev=cam_bev, lidar_bev=lidar_bev,
                bev360=bev360, middle=middle)


# ------------------------------------------------------------------ SCN onnx
def export_scn_onnx(model, save):
    """稀疏编码器 -> lidar.backbone.xyz.onnx（3DSparseConvolution 算子图）。

    BN+ReLU 必须融合进 SparseConvolution（activation 属性）；ReLU 留成独立
    节点时 libspconv 会算错稀疏邻域。xyz 布局：权重按 spconv 2.x OKI 原样导出。
    """
    from lean import exptool, funcs

    backbone = model.encoders["lidar"]["backbone"]
    backbone.eval().half()
    funcs.layer_fusion_bn_relu(backbone)
    voxels = torch.zeros(1, 5, dtype=torch.float16, device="cuda")
    coors = torch.zeros(1, 4, dtype=torch.int32, device="cuda")
    exptool.export_onnx(backbone, voxels, coors, 1, False, save)


# ------------------------------------------------------------------ 导出
class OnnxExporter:
    """Export the module-split ONNX graphs and verify them against torch."""

    def __init__(self, model, outdir, opset=17):
        self.model = model
        self.outdir = outdir
        self.opset = opset
        self.io = BF_IO
        self.wrappers = {
            name: WRAPPERS[name][0](model)
            for name in self.io
        }
        self.in_names = {name: io[0] for name, io in self.io.items()}
        self.out_names = {name: io[1] for name, io in self.io.items()}

    def export(self, inp):
        """逐图导出 ONNX，返回 {graph: path}。"""
        # trace 会把 BN 切到 train 并改 running stats，先快照、后恢复
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
        """onnxruntime 对齐 + onnx.checker。"""
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
        """导出全部图，可选 verify。"""
        inp = _sample_inputs(cfg, self.model)
        paths = self.export(inp)
        print("[export] onnx ->", self.outdir)
        if verify:
            self.verify(inp, paths)
        return paths


def main():
    from models import build_model
    from scripts.config import load_config
    from scripts.trainer import load_model_checkpoint

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("checkpoint")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--outdir", default=osp.join("outputs", "onnx"))
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--options", nargs="*", default=[])
    args = ap.parse_args()

    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    cfg = load_config(args.config, options=args.options)
    model = build_model(cfg.model).cuda().eval()
    load_model_checkpoint(model, args.checkpoint, strict=True)

    os.makedirs(args.outdir, exist_ok=True)
    paths = OnnxExporter(model, args.outdir,
                         opset=args.opset).run(cfg, verify=args.verify)

    # SCN 稀疏编码器（lidar.backbone.xyz.onnx）
    scn_path = osp.join(args.outdir, SCN_ONNX_NAME)
    export_scn_onnx(model, scn_path)

    print("[export] engine: DEBUG_MODEL=<name> DEBUG_PRECISION=fp16 "
          "bash deploy/export_trt.sh")


if __name__ == "__main__":
    main()
