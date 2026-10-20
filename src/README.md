# src/

预留目录：后续 ROS2 集成代码（节点、launch 文件、接口封装）。

TRT 推理工程代码位于 `3rdparty/deeplearning/deploy/`（与模型库同属深度学习范畴）：

- `deploy/python/` — Python 部署包（deploy.python，三层：preprocess/inference/postprocess）
- `deploy/cpp/` — C++ 推理（dsb_infer，三层 + ops）
- `deploy/tools/` — C++ 输入导出 + trt_perf 性能分析
- `deploy/launch/` — 部署启动脚本
