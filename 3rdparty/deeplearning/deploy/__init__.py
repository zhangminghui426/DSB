"""Deploy namespace root.

TRT 推理部署代码（属于深度学习/模型库范畴）：
  export.py  Python/C++ 推理共用的前置步骤：分模块 ONNX 导出 + trtexec 构建 engine
  python/    Python 推理（scripts/ 入口 + deploy.python 库，见 python/README.md）
  cpp/       C++ 推理（dsb_infer：preprocess/inference/postprocess + ops）
  tools/     export_cpp_input.py（C++ 输入导出）
  launch/    部署启动脚本（export/infer_trt/visualize_trt）

三层架构（python 与 cpp 结构对应）：
  preprocess   预处理：原始样本 -> 模型输入（深度投影 / 体素化）
  inference    推理：TRT engine 编排 + 自定义算子 -> 预测
  postprocess  后处理：预测 -> 检测框（解码 + NMS）
  自定义算子参考定义见 3rdparty/deeplearning/ops/（bev_pool_v2 / voxel / iou3d），
  C++ 侧为功能对齐的 CPU 实现（deploy/cpp/ops/）。
"""
