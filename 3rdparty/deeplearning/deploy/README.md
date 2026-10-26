# TRT 部署（deploy/）

C++ 推理完全移植 NVIDIA CUDA-BEVFusion（deploy/src/），不再区分预处理/推理/后处理三层，
直接采用参考实现 src/bevfusion/ 平铺结构。Python 侧负责 ONNX 导出、输入生成、
三端对拍与可视化（deploy/scripts/）。

## 目录结构（扁平布局）

```
deploy/
├── README.md            # 本文件（部署总览）
├── CMakeLists.txt       # C++ 构建（mkdir -p build && cd build && cmake .. && make）
├── build_trt_engine.sh  # engine 构建脚本（CUDA-BEVFusion 原版，trtexec --fp16 + IO
│                        #   fp16:chw；输出 deploy/model/<name>/build/*.plan）
├── trt.sh               # 构建并运行 C++ 推理（. environment.sh + cmake/make + bevfusion）
├── environment.sh       # 推理环境变量（TensorRT/CUDA/libspconv 路径）
├── export.sh            # ONNX 导出旧入口（可不用）
├── env.sh               # 导出环境（PY/CUDA_HOME 等，export.sh source 用）
├── src/                 # C++ 源码（bevfusion/common/onnx/plugins + main.cpp）
├── libs/                # 第三方库（libspconv / cuosd / spconv-onnx / stb）
├── model/               # 引擎/ONNX（<name>/build/*.plan + <name>/lidar.backbone.xyz.onnx）
├── scripts/             # export.py（ONNX）/ example_data.py（输入）/ model_diff.py（对拍）/ vis_torch.py
└── lean/                # SCN onnx 导出库（符号 trace / BN+ReLU 融合 / inverse 支持）
```

## 使用流程

1. 导出模型 onnx（Python，5 个 onnx → outputs/onnx/）：
   ```bash
   cd 3rdparty/deeplearning
   python deploy/scripts/export.py configs/det_convfuser.py ckpts/bevfusion-det-bf.pth --outdir outputs/onnx --verify
   ```
2. 构建 engine（build_trt_engine.sh，输出 deploy/model/<name>/build/）：
   ```bash
   cd 3rdparty/deeplearning
   mkdir -p deploy/model/resnet50
   cp outputs/onnx/camera.backbone.onnx outputs/onnx/camera.vtransform.onnx \
      outputs/onnx/fuser.onnx outputs/onnx/head.bbox.onnx deploy/model/resnet50/
   cp outputs/onnx/lidar.backbone.xyz.onnx deploy/model/resnet50/
   DEBUG_MODEL=resnet50 DEBUG_PRECISION=fp16 bash deploy/build_trt_engine.sh
   ```
3. 生成 example-data 输入：
   ```bash
   cd 3rdparty/deeplearning
   python deploy/scripts/example_data.py configs/det_convfuser.py ckpts/bevfusion-det-bf.pth
   ```
4. 构建并运行 C++ 推理：
   ```bash
   cd 3rdparty/deeplearning/deploy && bash trt.sh
   ```

## 产物

- outputs/onnx/：5 个 onnx
  - camera.backbone.onnx：img+depth → camera_feature(HWC) + camera_depth_weights
  - camera.vtransform.onnx：feat_in(1,80,360,360) → feat_out(1,80,180,180)
  - fuser.onnx：camera+lidar → middle(1,512,180,180)
  - head.bbox.onnx：middle → score/rot/dim/reg/height/vel
  - lidar.backbone.xyz.onnx：稀疏编码器（libspconv 运行时解析，不走 TRT）
- deploy/model/<name>/build/：4 个 .plan（camera.backbone / camera.vtransform / fuser /
  head.bbox.fp32，IO 均为 fp16:chw；head.bbox 用 fp32 层以规避 TRT fp16
  数值 bug，构建脚本已自动处理）

## 三端对拍（torch / onnx / trt）

```bash
cd 3rdparty/deeplearning
python deploy/scripts/model_diff.py configs/det_convfuser.py ckpts/bevfusion-det-bf.pth \
    --backend all --dump-dir outputs/model_diff
```

产物：`benchmark.json`（逐模块延时）、`report.png`（延时 + 精度对比）、
`onnx_vis.jpg` / `trt_vis.jpg`（框叠加）。`--backend bench` / `report`
可单独重跑性能与可视化。

## SCN 导出注意（ReLU 必须融合进卷积）

libspconv 的 ONNX 引擎按 NVIDIA 导出的结构执行稀疏图：`SparseConvolution`
节点的 `activation` 属性带 ReLU。若导出时把 ReLU 留成图里的独立 `Relu`
节点（即只做 BN 融合、不做 ReLU 融合），libspconv 会算错稀疏邻域——完整
SCN 的 `lidar_bev` 与 torch 严重不符、检测框发散且同输入多次运行不稳定。

`export.py` 用 `funcs.layer_fusion_bn_relu`（对应 CUDA-BEVFusion 的
`funcs.layer_fusion`）同时融合 BN 与 ReLU，与 NVIDIA 的
`lidar.backbone.xyz.onnx` 结构一致。C++ 侧因此只需参考实现那套单 engine
调用（`src/bevfusion/lidar-scn.cpp`）。

## 自定义算子归属

bev_pool / 深度投影 / 体素化 / 稀疏卷积等自定义算子在 CUDA-BEVFusion 中都是
C++ 侧独立 CUDA kernel（src/bevfusion/*.cu）或 libspconv 运行时，不进入 TRT engine；
TRT engine 只覆盖 camera.backbone / vtransform / fuser / head.bbox 四个网络部分。
