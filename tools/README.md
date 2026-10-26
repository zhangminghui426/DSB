# tools/

本目录预留：其他通用工程工具。

TRT 推理相关脚本位于 `3rdparty/deeplearning/deploy/`：
- `scripts/example_data.py` — torch 阶段：导出 C++ 推理输入 + 参考框
- `scripts/export.py` — onnx 阶段：分模块 ONNX 导出
- `run_trt.sh` — trt 阶段：构建并运行 C++ 推理
- `scripts/model_diff.py` / `scripts/vis_torch.py` — 三端对拍与可视化
