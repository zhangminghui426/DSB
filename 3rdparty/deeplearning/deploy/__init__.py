"""Deploy namespace root.

部署只走 C++（deploy/src/，CUDA-BEVFusion 完整移植）：
  export.py           Python 导出前置：CUDA-BEVFusion 结构 ONNX 导出（+ verify）
  build_trt_engine.sh engine 构建（trtexec --fp16 + IO fp16:chw，Python/C++ 共用）
  python/             导出链依赖的 tensor 名常量（graphs），无推理代码
  src/                C++ 推理源码（bevfusion/common/onnx/plugins + main.cpp）
  libs/               第三方库（libspconv/cuosd/spconv-onnx/stb）
  model/              引擎/ONNX（<name>/build/*.plan + lidar.backbone.xyz.onnx）
  tool/               工具（environment.sh/run.sh + model_diff.py 等 Python 工具）
  scripts/            导出链依赖常量包（graphs）
  lean/               SCN onnx 导出库（符号 trace / BN 融合）
  tools/              已并入 tool/
  lean/               SCN onnx 导出库（符号 trace / BN 融合）
  tools/              生成 example-data（C++ 输入）
"""
