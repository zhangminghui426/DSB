"""模型推理层（Deploy-time Inference）。

执行模块拆分的 TRT engine 图 + torch 侧自定义算子（bev_pool / 降采样 /
稀疏编码），产出网络预测张量。与 C++ 侧 ``src/cpp_infer/inference.*``
逐层对应（Inferencer 类 + camera_backbone / camera_vtransform / camera_bev /
lidar_branch / fuser / head）。
"""

import torch
from torch.nn import functional as F

from deploy.bev_pool_torch import camera_bev_from_depthnet
from deploy.graphs import DET_HEAD_KEYS

__all__ = ["Inferencer", "sparse_encoder_forward"]


def sparse_encoder_forward(model, feats, coords, sizes, batch_id=0):
    """mean-reduce + SparseEncoder -> (B, C*D, H, W) lidar BEV。

    coords 为体素化器产出的 (M, 3)（x,y,z），此处补 batch 维并做
    mean reduce 后送入稀疏骨干网络。
    """
    backbone = model.encoders["lidar"]["backbone"]
    coords = F.pad(coords, (1, 0), mode="constant", value=batch_id)
    if sizes is not None and getattr(model, "voxelize_reduce", True):
        feats = (feats.sum(dim=1, keepdim=False)
                 / sizes.type_as(feats).view(-1, 1)).contiguous()
    batch_size = coords[-1, 0] + 1
    return backbone(feats, coords, batch_size, sizes=sizes)


class Inferencer:
    """模型推理层：TRT engine 编排 + 自定义算子，输入 -> 预测张量。

    职责边界：只做模型前向（各子图 + torch 侧算子），
    不含输入准备（见 Preprocessor）与检测解码（见 Postprocessor）。
    """

    def __init__(self, model, engines, task="det"):
        self.model = model
        self.engines = engines
        self.task = task

    # ---- 子图 / 算子执行 ------------------------------------------------
    def camera_backbone(self, img):
        """图像 -> 多尺度特征 (1,6,256,32,88)。"""
        return self.engines["camera.backbone"].run({"img": img})["feats"]

    def camera_vtransform(self, feats, depth):
        """feats + lidar 深度 -> depthnet logits (1,6,D+C,32,88)。

        det 图带 depth 输入；seg 图无 depth 输入。
        """
        if self.task == "det":
            feeds = {"feats": feats, "depth": depth}
        else:
            feeds = {"feats": feats}
        return self.engines["camera.vtransform"].run(feeds)["depthnet"]

    def camera_bev(self, depthnet, batch):
        """depthnet -> camera BEV (1,80,180,180)（几何 + bev_pool + 降采样）。"""
        return camera_bev_from_depthnet(self.model, batch, depthnet, self.task)

    def lidar_branch(self, pre):
        """体素特征 -> lidar BEV (1,256,180,180)（mean reduce + 稀疏编码）。"""
        return sparse_encoder_forward(
            self.model, pre["vfeats"], pre["vcoords"], pre["vsizes"]
        )

    def fuser(self, cam_bev, lidar_bev):
        """camera + lidar BEV -> decoder_feat (1,512,180,180)。"""
        return self.engines["fuser"].run(
            {"cam_bev": cam_bev, "lidar_bev": lidar_bev}
        )["decoder_feat"]

    def head(self, decoder_feat):
        """decoder_feat -> 预测张量（det: 9-key dict；seg: masks）。"""
        if self.task == "det":
            preds = self.engines["head.bbox"].run(
                {"decoder_feat": decoder_feat}
            )
            return {k: preds[k] for k in DET_HEAD_KEYS}
        return self.engines["head.map"].run(
            {"decoder_feat": decoder_feat}
        )["masks"]

    # ---- 完整前向 -------------------------------------------------------
    @torch.no_grad()
    def infer(self, batch, pre):
        """完整模型前向：backbone -> vtransform -> bev_pool -> fuser -> head。

        Args:
            batch: collated 样本 dict（几何 / 矩阵用）。
            pre: Preprocessor 产出的模型输入 dict。

        Returns:
            (preds, decoder_feat)：det 为 9-key 预测 dict；seg 为 masks。
        """
        feats = self.camera_backbone(pre["img"])
        depthnet = self.camera_vtransform(feats, pre["depth"])
        cam_bev = self.camera_bev(depthnet, batch)
        lidar_bev = self.lidar_branch(pre)
        decoder_feat = self.fuser(cam_bev, lidar_bev)
        preds = self.head(decoder_feat)
        return preds, decoder_feat

    # ---- 一致性验证 -----------------------------------------------------
    @torch.no_grad()
    def verify(self, batch, decoder_feat):
        """Parity：torch bev_branch decoder_feat vs TRT fuser engine 输出。

        TRT engine 默认 FP32 + Ampere 上 TF32 kernel，torch 为严格 FP32
        （main 关闭 TF32）；卷积网络单点最大相对误差 ~1e-2 属预期。
        """
        b = batch
        torch_feat = self.model.bev_branch(
            b["img"], b["points"], None, b["camera2ego"], b["lidar2ego"],
            b["lidar2camera"], b["lidar2image"], b["camera_intrinsics"],
            b["camera2lidar"], b["img_aug_matrix"], b["lidar_aug_matrix"],
            b["metas"], depths=None, reverse=True,
        )[0]
        rel = float((decoder_feat - torch_feat).abs().max()
                    / max(1e-6, float(torch_feat.abs().max())))
        tol = 2e-2
        print(f"[verify] decoder_feat rel diff vs torch: {rel:.3e} "
              f"(tol {tol:.0e})")
        if self.task == "det" and rel < tol:
            out = self.model(**b)[0]
            boxes_t = out["boxes_3d"].tensor[:5]
            print(f"[verify] torch boxes[:5]:\n{boxes_t.cpu().numpy()}")
        return rel < 5e-3
