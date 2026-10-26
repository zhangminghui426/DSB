# TRT 部署（deploy/）

C++ 推理完全移植 NVIDIA CUDA-BEVFusion（deploy/cpp/），不再区分预处理/推理/后处理三层，
直接采用参考实现 src/bevfusion/ 平铺结构。部署只走 C++；Python 侧仅负责模型导出
（deploy/export.py，按 CUDA-BEVFusion 模型结构导出 5 个 onnx）与 engine 构建前置。

## 目录结构

```
deploy/
├── README.md           # 本文件（部署总览）
├── build_trt_engine.sh # engine 构建脚本（CUDA-BEVFusion 原版，trtexec --fp16 + IO fp16:chw，
│                       #   输出 model/<name>/build/*.plan）
├── export.py           # 唯一 ONNX 导出脚本（det: BF 结构 5 图 + SCN；--verify 数值对拍）
├── python/             # 导出链依赖的 tensor 名常量（graphs），无推理代码
├── lean/               # SCN onnx 导出库（符号 trace / BN 融合 / inverse 支持）
├── cpp/                # C++ 推理（CUDA-BEVFusion 完整移植，见 cpp/README.md）
├── tools/
│   ├── export_example_data.py  # 生成 example-data（6 jpg + .tensor）
│   └── export_cpp_input.py     # 旧输入导出（历史保留）
└── launch/             # export.sh（ONNX -> engine 旧入口，可不用）
```

## 使用流程

1. 导出模型 onnx（Python，5 个 onnx → outputs/onnx_bf/）：
   ```bash
   cd 3rdparty/deeplearning
   python deploy/export.py configs/det_convfuser.py ckpts/bevfusion-det-bf.pth --outdir outputs/onnx --verify
   ```
2. 构建 engine（build_trt_engine.sh，输出 model/<name>/build/）：
   ```bash
   cd 3rdparty/deeplearning
   mkdir -p model/resnet50/build
   cp outputs/onnx_bf/camera.backbone.onnx outputs/onnx_bf/camera.vtransform.onnx outputs/onnx_bf/fuser.onnx outputs/onnx_bf/head.bbox.onnx model/resnet50/
   cp outputs/onnx_bf/lidar.backbone.xyz.onnx model/resnet50/
   DEBUG_MODEL=resnet50 DEBUG_PRECISION=fp16 bash deploy/build_trt_engine.sh
   ```
3. 按 main.cpp 布局放置模型：[cpp/README.md#模型导出与布局](cpp/README.md#模型导出与布局)
4. 生成 example-data 输入：[cpp/README.md#数据example-data](cpp/README.md#数据example-data)
5. 构建并运行 C++ 推理：[cpp/README.md](cpp/README.md)

## 产物

- outputs/onnx_bf/：5 个 onnx
  - camera.backbone.onnx：img+depth → camera_feature(HWC) + camera_depth_weights
  - camera.vtransform.onnx：feat_in(1,80,360,360) → feat_out(1,80,180,180)
  - fuser.onnx：camera+lidar → middle(1,512,180,180)
  - head.bbox.onnx：middle → score/rot/dim/reg/height/vel
  - lidar.backbone.xyz.onnx：稀疏编码器（libspconv 运行时解析，不走 TRT）
- model/<name>/build/：4 个 .plan（camera.backbone / camera.vtransform / fuser /
  head.bbox，--fp16 + IO fp16:chw）

## 自定义算子归属

bev_pool / 深度投影 / 体素化 / 稀疏卷积等自定义算子在 CUDA-BEVFusion 中都是
C++ 侧独立 CUDA kernel（src/bevfusion/*.cu）或 libspconv 运行时，不进入 TRT engine；
TRT engine 只覆盖 camera.backbone / vtransform / fuser / head.bbox 四个网络部分。
