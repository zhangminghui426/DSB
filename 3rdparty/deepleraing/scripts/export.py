"""Export BEVFusion to ONNX + TensorRT engines.

Follows the CUDA-BEVFusion (NVIDIA-AI-IOT/Lidar_AI_Solution) module split so
every exported graph is dense and TRT-parseable without custom plugins:

  det (4 graphs):
    camera.backbone.onnx  : img (1,6,3,256,704)              -> feats (1,6,256,32,88)
    camera.vtransform.onnx: feats + depth (1,6,1,256,704)    -> depthnet (1,6,D+C,32,88)
    fuser.onnx            : cam_bev + lidar_bev (1,80/256,180,180) -> (1,512,180,180)
    head.bbox.onnx        : (1,512,180,180)                  -> 8 pred tensors (no decode)
  seg (4 graphs): same, minus depth; head.map outputs masks (1,6,200,200)

Runtime split (deploy/infer_trt.py), mirroring CUDA-BEVFusion:
  TRT engines : camera backbone / depthnet / fuser+decoder / head
  torch+CUDA  : normalization + depth projection (deploy.preprocess), camera
                geometry + bev_pool (ops CUDA ext) + vtransform downsample,
                lidar voxelization + sparse encoder (spconv), box decode

ONNX only here; TensorRT deployment lives in the separate cuda-bevfusion
project (see ../cuda-bevfusion, referencing NVIDIA-AI-IOT/Lidar_AI_Solution).

Usage (repo root):
  python scripts/export.py CONFIG CHECKPOINT --task det --verify
"""

import argparse
import os
import os.path as osp
import sys

sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), ".."))

import numpy as np
import torch
from torch import nn

from datasets import build_dataset, collate_fn
from scripts.config import load_config
from scripts.trainer import load_model_checkpoint, to_device
from models import build_model

DET_HEAD_KEYS = [
    "dense_heatmap",
    "heatmap",
    "query_heatmap_score",
    "center",
    "height",
    "dim",
    "rot",
    "vel",
]


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
    """TransFusionHead raw pred tensors (decode stays in postprocess)."""

    def __init__(self, model):
        super().__init__()
        self.head = model.heads["object"]
        self.head.onnx_export = True  # unfold/amax instead of max_pool2d

    def forward(self, x):
        pd = self.head([x], [{}])
        d = pd[0][0]
        return tuple(d[k] for k in DET_HEAD_KEYS)


class HeadMap(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.head = model.heads["map"]

    def forward(self, x):
        return self.head([x])


# ------------------------------------------------------------------ helpers
def bn_snapshot(model):
    state = {}
    for name, m in model.named_modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            state[name] = (
                m.running_mean.clone(),
                m.running_var.clone(),
                m.num_batches_tracked.clone(),
                m.training,
            )
    return state


def bn_restore(model, state):
    for name, m in model.named_modules():
        if name in state:
            rm, rv, nbt, tr = state[name]
            m.running_mean.copy_(rm)
            m.running_var.copy_(rv)
            m.num_batches_tracked.copy_(nbt)
            m.training = tr
    model.eval()


def build_graphs(model, task):
    """task -> ordered list of (name, wrapper, input_names, output_names)."""
    if task == "det":
        return [
            ("camera.backbone", CamBackbone(model), ["img"], ["feats"]),
            (
                "camera.vtransform",
                CamVtransformDet(model),
                ["feats", "depth"],
                ["depthnet"],
            ),
            ("fuser", FuserDecoder(model), ["cam_bev", "lidar_bev"], ["decoder_feat"]),
            ("head.bbox", HeadBbox(model), ["decoder_feat"], DET_HEAD_KEYS),
        ]
    return [
        ("camera.backbone", CamBackbone(model), ["img"], ["feats"]),
        ("camera.vtransform", CamVtransformSeg(model), ["feats"], ["depthnet"]),
        ("fuser", FuserDecoder(model), ["cam_bev", "lidar_bev"], ["decoder_feat"]),
        ("head.map", HeadMap(model), ["decoder_feat"], ["masks"]),
    ]


def graph_args(name, inp):
    if name == "camera.backbone":
        return (inp["img"],)
    if name == "camera.vtransform":
        if inp["depth"] is not None:
            return (inp["feats"], inp["depth"])
        return (inp["feats"],)
    if name == "fuser":
        return (inp["cam_bev"], inp["lidar_bev"])
    return (inp["decoder_feat"],)


def sample_inputs(cfg, model, task):
    """One real val sample -> example tensors for every graph."""
    from deploy.bev_pool_torch import camera_bev_from_depthnet
    from deploy.lidar_branch import LidarBranchEncoder
    from deploy.preprocess import project_depth

    ds = build_dataset(cfg.data.val)
    sample = ds[0]
    batch = to_device(collate_fn([sample]), "cuda")
    img = batch["img"]

    lidar_enc = LidarBranchEncoder(model)
    # keep points on the model device so fuser can cat cam/lidar BEVs
    lidar_bev = lidar_enc([batch["points"][0]], use_fp16=False)

    with torch.no_grad():
        feats = CamBackbone(model)(img)
        if task == "det":
            depth = to_device(
                project_depth(
                    batch["points"][0].cpu(),
                    batch["lidar2image"][0].cpu(),
                    batch["img_aug_matrix"][0].cpu().float(),
                    batch["lidar_aug_matrix"][0].cpu().float(),
                ),
                "cuda",
            ).unsqueeze(0)
            depthnet = CamVtransformDet(model)(feats, depth)
        else:
            depth = None
            depthnet = CamVtransformSeg(model)(feats)
        cam_bev = camera_bev_from_depthnet(model, batch, depthnet, task)
        decoder_feat = FuserDecoder(model)(cam_bev, lidar_bev)

    return dict(
        batch=batch,
        sample=sample,
        img=img,
        feats=feats,
        depth=depth,
        depthnet=depthnet,
        cam_bev=cam_bev,
        lidar_bev=lidar_bev,
        decoder_feat=decoder_feat,
    )


def export_onnx(task, model, inp, outdir, opset):
    # torch.onnx.export tracing flips BatchNorm into train mode and mutates
    # running stats; snapshot/restore so later torch references stay valid
    snap = bn_snapshot(model)
    paths = {}
    for name, wrapper, in_names, out_names in build_graphs(model, task):
        args = graph_args(name, inp)
        path = osp.join(outdir, f"{name}.onnx")
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                args,
                path,
                opset_version=opset,
                input_names=in_names,
                output_names=out_names,
                do_constant_folding=True,
            )
        paths[name] = path
    bn_restore(model, snap)
    return paths


def verify_onnx(task, model, inp, paths):
    import onnx
    import onnxruntime as ort

    ok = True
    for name, wrapper, in_names, out_names in build_graphs(model, task):
        sess = ort.InferenceSession(paths[name], providers=["CPUExecutionProvider"])
        args = graph_args(name, inp)
        feeds = {n: a.detach().cpu().numpy() for n, a in zip(in_names, args)}
        outs = sess.run(None, feeds)
        with torch.no_grad():
            refs = wrapper(*args)
        refs = refs if isinstance(refs, (tuple, list)) else (refs,)
        worst = 0.0
        for o, r in zip(outs, refs):
            rn = r.cpu().numpy()
            d = float(np.abs(o - rn).max())
            worst = max(worst, d / max(1.0, float(np.abs(rn).max())))
        tol = 5e-3
        ok &= worst < tol
        print(f"[verify] {name}: worst rel diff = {worst:.3e} (tol {tol:.0e})")
    for p in paths.values():
        onnx.checker.check_model(onnx.load(p))
    print("[verify] onnx.checker OK")
    assert ok, "ONNX parity failed"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config")
    ap.add_argument("checkpoint")
    ap.add_argument("--task", default="det", choices=["det", "seg"])
    ap.add_argument("--format", default="onnx", choices=["onnx"])
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--outdir", default=osp.join("outputs", "onnx"))
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    cfg = load_config(args.config)
    model = build_model(cfg.model).cuda().eval()
    load_model_checkpoint(model, args.checkpoint, strict=True)

    inp = sample_inputs(cfg, model, args.task)
    paths = export_onnx(args.task, model, inp, args.outdir, args.opset)
    print("[export] onnx:", paths)
    if args.verify or args.format == "all":
        verify_onnx(args.task, model, inp, paths)


if __name__ == "__main__":
    main()
