# TRT 部署（deploy/）

C++ 推理完全移植 NVIDIA CUDA-BEVFusion（deploy/cpp/），不再区分预处理/推理/后处理三层。
Python 侧负责模型导出（deploy/export_bf.py）。

## 目录结构

- cpp/：C++ 推理（CUDA-BEVFusion 移植，bevfusion 可执行）
- python/：Python 导出/校验（export_bf.py 入口，lean/ 为 SCN 导出移植）
- tools/：输入导出（export_example_data.py 生成 example-data）
- launch/：旧启动脚本（历史保留，新流程不依赖）
- export_bf.py：模型导出 + engine 构建主入口（见 python/README.md）

## 使用流程

1. 导出模型（Python，见 python/README.md）
2. 模型布局 + 构建 + 运行（C++，见 cpp/README.md）

## 产物

- outputs/onnx_bf/：5 个 onnx（camera.backbone / camera.vtransform / fuser /
  head.bbox / lidar.backbone.xyz）
- outputs/trt_bf/：4 个 .plan（camera.backbone / camera.vtransform / fuser / head.bbox）
