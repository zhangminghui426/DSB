"""CUDA-BEVFusion 结构 ONNX 导出（det）。

与 NVIDIA-AI-IOT/CUDA-BEVFusion 的模型结构对齐（5 个 onnx）：
  camera.backbone    img + depth -> camera_feature(B,N,fH,fW,C) + camera_depth_weights(B,N,D,fH,fW)
                     （backbone+neck+dtransform+depthnet 合并导出，softmax 在图中）
  camera.vtransform  feat_in(1,80,360,360) -> feat_out(1,80,180,180)  （= vtransform.downsample）
  fuser              camera(1,80,180,180) + lidar(1,256,180,180) -> middle(1,512,180,180)
  head.bbox          middle -> score/rot/dim/reg/height/vel（decode 由 C++ head-transbbox kernel 完成）
  lidar.backbone.xyz 稀疏编码器（voxels + coors -> dense BEV，libspconv 解析）

用法：
  python deploy/export_bf.py configs/det_convfuser.py ckpts/bevfusion-det-bf.pth \
      --outdir outputs/onnx_bf --verify

SCN onnx 依赖 GPU（模型 .half() 推理 trace）；
engine 构建统一走 CUDA-BEVFusion 原版脚本（--fp16 + IO fp16:chw）：
  cd 3rdparty/deeplearning
  cp outputs/onnx_bf/camera.backbone.onnx outputs/onnx_bf/camera.vtransform.onnx outputs/onnx_bf/fuser.onnx outputs/onnx_bf/head.bbox.onnx model/resnet50/
  DEBUG_MODEL=resnet50 DEBUG_PRECISION=fp16 bash deploy/build_trt_engine.sh
"""

import argparse
import os
import os.path as osp
import sys

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))  # deploy/
sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))  # deeplearning/

import numpy as np
import torch
from torch import nn

# 复用现有基础设施（模型构建 / 采样 / BN 快照）
import export  # deploy/export.py
from export import (_bn_restore, _bn_snapshot, FuserDecoder,
                    _sample_inputs)

# ------------------------------------------------------------------ 图定义
BF_IO = {
    "camera.backbone": (["img", "depth"], ["camera_feature", "camera_depth_weights"]),
    "camera.vtransform": (["feat_in"], ["feat_out"]),
    "fuser": (["camera", "lidar"], ["middle"]),
    "head.bbox": (["middle"], ["score", "rot", "dim", "reg", "height", "vel"]),
}
SCN_ONNX_NAME = "lidar.backbone.xyz.onnx"


# ------------------------------------------------------------------ wrappers
class CamBackboneBf(nn.Module):
    """backbone + neck + dtransform + depthnet 合并导出（CUDA-BEVFusion 结构）。

    forward(img, depth):
        img   (B,N,3,H,W)   深度网络输入 depth (B,N,1,H,W)
        -> camera_feature (B,N,fH,fW,C)  （HWC！bevpool kernel 按此布局取数）
        -> camera_depth_weights (B,N,D,fH,fW)（已 softmax，bevpool 直接乘）
    """

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


class DownsampleBf(nn.Module):
    """bev_pool 输出 (1,80,360,360) -> (1,80,180,180)（= vtransform.downsample）。"""

    def __init__(self, model):
        super().__init__()
        self.downsample = model.encoders["camera"]["vtransform"].downsample

    def forward(self, feat_in):
        return self.downsample(feat_in)


class HeadBboxBf(nn.Module):
    """transfusion head 原始预测（decode 留在 C++）。

    forward(middle): (1,512,180,180)
        -> (score, rot, dim, reg, height, vel)
        score = heatmap.sigmoid() * query_heatmap_score * one_hot   (B,num_classes,num)
        reg = center（decode kernel 按 (2,num) 布局读取）
    """

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


WRAPPERS_BF = {
    "camera.backbone": (CamBackboneBf, BF_IO["camera.backbone"][0], BF_IO["camera.backbone"][1]),
    "camera.vtransform": (DownsampleBf, BF_IO["camera.vtransform"][0], BF_IO["camera.vtransform"][1]),
    "fuser": (FuserDecoder, BF_IO["fuser"][0], BF_IO["fuser"][1]),
    "head.bbox": (HeadBboxBf, BF_IO["head.bbox"][0], BF_IO["head.bbox"][1]),
}


# ------------------------------------------------------------------ 采样
def _sample_inputs_bf(cfg, model):
    """在现有采样基础上补算 bev_pool 输出（vtransform 输入，360x360）。"""
    inp = _sample_inputs(cfg, model, "det")
    with torch.no_grad():
        vt = model.encoders["camera"]["vtransform"]
        depthnet = inp["depthnet"]
        B, N = inp["img"].shape[0], inp["img"].shape[1]
        D, C = vt.D, vt.C
        fH, fW = vt.feature_size
        depth = depthnet[:, :D].softmax(dim=1).view(B, N, D, fH, fW)
        feat = depthnet[:, D:].view(B, N, C, fH, fW)
        x = (depth.unsqueeze(2) * feat.unsqueeze(3))
        x = x.permute(0, 1, 3, 4, 5, 2).contiguous()
        geom = vt.get_geometry(
            inp["batch"]["camera2lidar"][..., :3, :3],
            inp["batch"]["camera2lidar"][..., :3, 3],
            inp["batch"]["camera_intrinsics"][..., :3, :3],
            inp["batch"]["img_aug_matrix"][..., :3, :3],
            inp["batch"]["img_aug_matrix"][..., :3, 3],
            extra_rots=inp["batch"]["lidar_aug_matrix"][..., :3, :3],
            extra_trans=inp["batch"]["lidar_aug_matrix"][..., :3, 3],
        )
        inp["bev360"] = vt.bev_pool(geom, x)  # (B, C, 360, 360)
    return inp


def _bf_args(name, inp):
    if name == "camera.backbone":
        return (inp["img"], inp["depth"])
    if name == "camera.vtransform":
        return (inp["bev360"],)
    if name == "fuser":
        return (inp["cam_bev"], inp["lidar_bev"])
    return (inp["decoder_feat"],)


# ------------------------------------------------------------------ SCN onnx
def export_scn_onnx(model, save):
    """稀疏编码器 -> lidar.backbone.xyz.onnx（3DSparseConvolution 自定义算子图）。

    参考 qat/export-scn.py：voxels(1,5) + coors(1,4) 单点占位 trace，
    权重来自 checkpoint（非量化模型，跳过 PTQ 属性设置）。

    注意：参考是 PTQ 模型（BN 已在训练期折叠），我们的 FP 模型需要先
    layer_fusion_bn 把 conv+bn 融合进卷积权重，否则 trace 时 BN 输出
    tensor 无法注册（残差加法 get_tensor_id 断言失败）。
    """
    from lean import exptool, funcs

    backbone = model.encoders["lidar"]["backbone"]
    backbone.eval().half()
    funcs.layer_fusion_bn(backbone)  # 融合 conv+bn（ReLU 保留为独立节点）
    voxels = torch.zeros(1, 5, dtype=torch.float16, device="cuda")
    coors = torch.zeros(1, 4, dtype=torch.int32, device="cuda")
    # xyz/OKI 导出（inverse=False）：用户机器 libspconv(.so) 实测接受
    # (Cout,Kz,Ky,Kx,Cin) 布局（0 断言，Lidar Backbone 跑通）。
    # 若需 zyx（官方默认，性能略优）可改 True，C++ 侧不变。
    exptool.export_onnx(backbone, voxels, coors, 1, False, save)


# ------------------------------------------------------------------ 主流程
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config")
    ap.add_argument("checkpoint")
    ap.add_argument("--outdir", default=osp.join("outputs", "onnx_bf"))
    ap.add_argument("--opset", type=int, default=13)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--options", nargs="*", default=[])
    args = ap.parse_args()

    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    from scripts.config import load_config
    from models import build_model
    from scripts.trainer import load_model_checkpoint

    cfg = load_config(args.config, options=args.options)
    model = build_model(cfg.model).cuda().eval()
    load_model_checkpoint(model, args.checkpoint, strict=True)

    os.makedirs(args.outdir, exist_ok=True)

    # ---- camera 链 + head（onnx 导出 + 可选 verify）----
    inp = _sample_inputs_bf(cfg, model)
    snap = _bn_snapshot(model)
    paths = {}
    for name, (wrapper, in_names, out_names) in WRAPPERS_BF.items():
        path = osp.join(args.outdir, f"{name}.onnx")
        with torch.no_grad():
            torch.onnx.export(
                wrapper(model), _bf_args(name, inp), path,
                opset_version=args.opset,
                input_names=in_names,
                output_names=out_names,
                do_constant_folding=True,
            )
        paths[name] = path
    _bn_restore(model, snap)
    print("[export_bf] onnx:", paths)

    if args.verify:
        import onnx
        import onnxruntime as ort
        ok = True
        for name, (wrapper, in_names, out_names) in WRAPPERS_BF.items():
            sess = ort.InferenceSession(paths[name],
                                        providers=["CPUExecutionProvider"])
            feeds = {n: a.detach().cpu().numpy()
                     for n, a in zip(in_names, _bf_args(name, inp))}
            outs = sess.run(None, feeds)
            with torch.no_grad():
                refs = wrapper(model)(*_bf_args(name, inp))
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

    # ---- SCN onnx（GPU 推理 trace，非量化模型）----
    scn_path = osp.join(args.outdir, SCN_ONNX_NAME)
    export_scn_onnx(model, scn_path)
    print("[export_bf] scn:", scn_path)

    print()
    print("[export_bf] engine 构建（CUDA-BEVFusion 原版脚本，--fp16 + IO fp16:chw）：")
    print("  cd 3rdparty/deeplearning")
    print("  cp outputs/onnx_bf/camera.backbone.onnx outputs/onnx_bf/camera.vtransform.onnx outputs/onnx_bf/fuser.onnx outputs/onnx_bf/head.bbox.onnx model/resnet50/")
    print("  DEBUG_MODEL=resnet50 DEBUG_PRECISION=fp16 bash deploy/build_trt_engine.sh")


if __name__ == "__main__":
    main()
