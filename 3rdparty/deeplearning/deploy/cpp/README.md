# bevfusion（C++ 推理，3rdparty/deeplearning/deploy/cpp）

本目录是 [NVIDIA-AI-IOT/CUDA-BEVFusion](https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution/tree/master/CUDA-BEVFusion)
推理代码的完整移植：不再区分预处理/推理/后处理三层，直接采用参考实现
`src/bevfusion/` 平铺结构（bevfusion.cpp 主流程 + 各算子类）。

## 目录结构

```
deploy/cpp/
├── CMakeLists.txt          # 构建（bevfusion_core.so + bevfusion + custom_layernorm.so）
├── src/
│   ├── bevfusion/          # 主流程与算子（原样移植，保留 MIT license）
│   │   ├── bevfusion.cpp/.hpp      # Core 编排（normalize→backbone→bevpool→vtransform→fusion→head）
│   │   ├── camera-backbone.{cu,hpp}     # TRT：img + depth → camera_depth_weights + camera_feature
│   │   ├── camera-depth.{cu,hpp}        # CUDA：lidar 点云投影生成深度图
│   │   ├── camera-normalization.{cu,hpp}  # CUDA：图像归一化/缩放
│   │   ├── camera-geometry.{cu,hpp}     # CUDA：frustum 反投影 + 排名
│   │   ├── camera-bevpool.{cu,hpp}      # CUDA：depth_weights × feature 池化求和
│   │   ├── camera-vtransform.{cu,hpp}   # TRT：camera_bev → camera_bevfeat
│   │   ├── transfusion.{cu,hpp}         # TRT：cam_bevfeat + lidar_bev → decoder_feat
│   │   ├── head-transbbox.{cu,hpp}      # TRT：decoder_feat → 3D boxes（含后处理）
│   │   ├── lidar-voxelization.{cu,hpp}  # CUDA：点云体素化
│   │   ├── lidar-scn.{cpp,hpp}          # libspconv 稀疏卷积（onnx 权重解析）
│   │   └── lidar-scn-onnx-parser.{cpp,hpp}  # protobuf 解析 SCN onnx 权重
│   ├── common/             # tensor/tensorrt/check/timer/visualize 基础设施
│   ├── onnx/               # onnx protobuf（已生成，无需 protoc）
│   ├── plugins/            # custom_layernorm TRT 插件（编译为 .so）
│   └── main.cpp            # 入口：读 example-data（jpg + tensor），forward + visualize
├── libs/
│   ├── libspconv/          # 3DSparseConvolution：include + 预编译 .so
│   │   └── lib/x86_64_cuda12.8/libspconv.so
│   ├── cuosd/              # 可视化标注（head-transbbox 画框）
│   └── spconv-onnx/        # 3DSparseConvolution 的 onnx parser 参考
├── dependencies/
│   ├── stb/                # 图像解码（stb_image）
│   └── dlpack/
└── tool/
    ├── environment.sh      # 环境路径（TensorRT / CUDA / SPCONV 版本）
    ├── build_trt_engine.sh # trtexec 从 onnx 构建 engine
    └── run.sh              # 构建 + 运行
```

## 构建

前置依赖（用户环境已具备）：
- CUDA 12.8（`/usr/local/cuda-12.8`）、TensorRT 10.9（`/opt/TensorRT-10.9.0.34`）
- `libprotobuf-dev`（`apt install libprotobuf-dev`）
- libspconv 预编译库（仓库已含 `libs/libspconv/lib/x86_64_cuda12.8`，对应
  CUDA 12.x；11.x 用 `x86_64_cuda11.4`）

```bash
cd 3rdparty/deeplearning/deploy/cpp
. tool/environment.sh          # 设 TensorRT/CUDA/SPCONV 环境
bash tool/run.sh               # cmake + make + 运行（build/bevfusion）
```

产物：`build/bevfusion`、`build/bevfusion_core.so`、`build/custom_layernorm.so`
（main 内 dlopen 加载 layernorm 插件）。

## 模型与数据（CUDA-BEVFusion 结构）

参考实现要求以下模型（onnx → engine 构建，或直接放 engine）：

| 模型 | 输入 | 输出 |
| --- | --- | --- |
| `camera.backbone` | `img`(6×3×H×W) + `depth`(lidar 深度图) | `camera_depth_weights`(N×D×H×W)、`camera_feature` |
| `camera.vtransform` | `camera_bev` | `camera_bevfeat` |
| `fuser`（transfusion） | `camera_bevfeat` + `lidar_bev` | `decoder_feat` |
| `head.bbox` | `decoder_feat` | 3D boxes |
| `lidar.backbone.xyz.onnx` | 体素（voxels/coords） | sparse 特征（libspconv 推理） |

运行数据：`example-data/`（`camera2lidar.tensor`、`camera_intrinsics.tensor`、
`lidar2image.tensor`、`img_aug_matrix.tensor`、`points.tensor`、6 张相机 jpg）。

> 注意：此结构与旧版导出（`img→feats`、`feats+depth→depthnet` 分离）不兼容，
> 需要按上表重新导出模型（Python 侧 `deploy/python/scripts/builder.py` 改造见
> `../README.md` 与 `deploy/python/README.md`）。

## 与旧实现的关系

旧三层实现（`include/{preprocess,inference,postprocess}` + 自研 CUDA ops）已移除，
由 CUDA-BEVFusion 参考实现替代。自研实现的 git 历史保留在仓库中，如需回退：
`git log --oneline 3rdparty/deeplearning/deploy/cpp`。
