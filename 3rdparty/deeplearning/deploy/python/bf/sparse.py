"""lidar 稀疏编码器（spconv.pytorch，Python 侧等价 C++ 的 libspconv 运行时）。

与 deploy/cpp/src/bevfusion/lidar-scn.cpp 对应：voxels + coors ->
SparseEncoder -> dense BEV (1,256,180,180)。权重直接来自训练 checkpoint
（同一份权重，与导出的 lidar.backbone.xyz.onnx 数值一致）。
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))  # deeplearning/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))       # deploy/

from models import build_model              # noqa: E402
from scripts.config import load_config      # noqa: E402
from scripts.trainer import load_model_checkpoint  # noqa: E402


def build_lidar_backbone(cfg_path: str, ckpt_path: str) -> torch.nn.Module:
    """构建 SparseEncoder 并加载 ckpt 权重，返回 .half().eval().cuda() 的 backbone。

    复用训练侧 build_model（同一份模型定义），取 encoders.lidar.backbone。
    """
    cfg = load_config(cfg_path)
    model = build_model(cfg.model).cuda().eval()
    load_model_checkpoint(model, ckpt_path)
    backbone = model.encoders["lidar"]["backbone"].half().eval()
    print(f"[bf/sparse] SparseEncoder ready  sparse_shape={backbone.sparse_shape}")
    return backbone


def sparse_forward(backbone, voxel_features: torch.Tensor, coors: torch.Tensor, batch_size: int = 1) -> torch.Tensor:
    """voxel_features (M,C) float32（已均值化）+ coors (M,4) int32 [b,z,y,x]
    -> dense BEV (B, 256, 180, 180) half。
    """
    with torch.no_grad():
        return backbone(voxel_features.half().contiguous(), coors.int().contiguous(), batch_size)
