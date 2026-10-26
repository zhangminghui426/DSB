# src/

预留目录：后续 ROS2 集成代码（节点、launch 文件、接口封装）。

TRT 推理工程代码位于 `3rdparty/deeplearning/deploy/`（与模型库同属深度学习范畴）：

- `deploy/scripts/export.py` — 分模块 ONNX 导出（5 张图）
- `deploy/src/` — C++ 推理（CUDA-BEVFusion 移植）
- `deploy/tool/` — example-data 导出、三端对拍、环境/运行脚本
- `deploy/build_trt_engine.sh` — TRT engine 构建（head.bbox 用 fp32 层）
