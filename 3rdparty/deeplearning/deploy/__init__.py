"""Deploy namespace root.

部署只走 C++（deploy/cpp/，CUDA-BEVFusion 完整移植）：
  export_bf.py        Python 导出前置：CUDA-BEVFusion 结构 ONNX 导出（+ verify）
  build_trt_engine.sh engine 构建（trtexec --fp16 + IO fp16:chw，Python/C++ 共用）
  python/             导出链依赖的 tensor 名常量（graphs），无推理代码
  cpp/                C++ 推理（src/bevfusion 平铺结构 + libspconv 稀疏卷积）
  lean/               SCN onnx 导出库（符号 trace / BN 融合）
  tools/              生成 example-data（C++ 输入）
  launch/             export.sh（ONNX -> engine 旧入口，可不用）
"""
