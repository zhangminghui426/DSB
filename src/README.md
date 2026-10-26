# src/

预留目录：后续 ROS2 集成代码（节点、launch 文件、接口封装）。

TRT 推理工程代码位于 `3rdparty/deeplearning/deploy/`（与模型库同属深度学习范畴）：

- `deploy/tool/onnx_stage.py` — 分模块 ONNX 导出（5 张图）
- `deploy/src/` — C++ 推理（CUDA-BEVFusion 移植）
- `deploy/tool/` — torch_stage.py（输入）/ onnx_stage.py（导出）/ trt_stage.sh（运行）+ 对拍与可视化
- `deploy/build_trt_engine.sh` — TRT engine 构建（head.bbox 用 fp32 层）
