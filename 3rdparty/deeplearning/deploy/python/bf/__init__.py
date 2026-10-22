"""bf —— CUDA-BEVFusion 结构 Python 部署（镜像 deploy/cpp/）。

加载 4 个 .plan（build_trt_engine.sh 构建）+ 自定义 CUDA 算子（bevfusion_ops）
+ spconv.pytorch 稀疏编码器，数据流与 C++ bevfusion.cpp forward 对齐。

模块：
  runtime.py   4 个 .plan 加载与执行（TRT，IO fp16）
  cuda_ops.py  自定义 CUDA 算子封装（depth/geometry/bevpool/voxelize）
  sparse.py    spconv.pytorch SparseEncoder（ckpt 权重）
  infer.py     推理编排 + head decode
"""
from .infer import BevfusionBf
from . import cuda_ops

__all__ = ["BevfusionBf", "cuda_ops"]
