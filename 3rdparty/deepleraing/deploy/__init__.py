"""Deploy namespace root.

TRT 推理部署代码（属于深度学习/模型库范畴）：
  python/   Python 部署包（deploy.python：export/builder/runtime + 三层推理）
  cpp/      C++ 推理（dsb_infer：preprocess/inference/postprocess + ops）
  tools/    export_cpp_input.py（C++ 输入导出）+ trt_perf（性能分析）
  launch/   部署启动脚本（export/infer_trt/perf_trt/visualize_trt）

三层架构（python 与 cpp 结构对应）：
  preprocess   预处理：原始样本 -> 模型输入（深度投影 / 体素化）
  inference    推理：TRT engine 编排 + 自定义算子 -> 预测
  postprocess  后处理：预测 -> 检测框（解码 + NMS）
  自定义算子参考定义见 3rdparty/deepleraing/ops/（bev_pool_v2 / voxel / iou3d），
  C++ 侧为功能对齐的 CPU 实现（deploy/cpp/ops/）。
"""
