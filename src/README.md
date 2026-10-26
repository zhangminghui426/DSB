# src/

预留目录：后续 ROS2 集成代码（节点、launch 文件、接口封装）。

TRT 推理工程代码位于 `3rdparty/deeplearning/deploy/`（与模型库同属深度学习范畴）：

- `deploy/scripts/export.py` — 分模块 ONNX 导出（5 张图）
- `deploy/src/` — C++ 推理（CUDA-BEVFusion 移植）
- `deploy/scripts/` — export.py（ONNX 导出）/ example_data.py（输入）/ model_diff.py（对拍）/ vis_torch.py（可视化）
- `deploy/run_trt.sh` — 构建并运行 C++ 推理
- `deploy/export_trt.sh` — TRT engine 构建（head.bbox 用 fp32 层）
