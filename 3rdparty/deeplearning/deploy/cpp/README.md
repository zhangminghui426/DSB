# bevfusion（deploy/cpp，CUDA-BEVFusion 移植）

本目录是 [NVIDIA-AI-IOT/CUDA-BEVFusion](https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution/tree/master/CUDA-BEVFusion)
推理代码的完整移植：src/bevfusion/ 原样拷贝（保留 MIT license），稀疏卷积使用
3DSparseConvolution 预编译 libspconv，ONNX protobuf 已生成（无需 protoc）。

## 目录结构

```
deploy/cpp/
├── CMakeLists.txt          # 构建：bevfusion_core.so + bevfusion + custom_layernorm.so
├── README.md               # 本文件
├── LICENSE-CUDA-BEVFusion.md  # 参考实现 MIT license
├── src/
│   ├── bevfusion/          # 主流程与算子（原样移植）
│   │   ├── bevfusion.cpp/.hpp        # Core 编排（normalize→backbone→bevpool→vtransform→fusion→head）
│   │   ├── camera-backbone.cu/.hpp   # TRT engine：img+depth → depth_weights + camera_feature
│   │   ├── camera-depth.cu/.hpp      # CUDA：lidar 点云投影生成深度图
│   │   ├── camera-normalization.cu/.hpp  # CUDA：图像归一化/缩放（704x256）
│   │   ├── camera-geometry.cu/.hpp   # CUDA：frustum 反投影 + 排名
│   │   ├── camera-bevpool.cu/.hpp    # CUDA：depth_weights × feature 池化求和
│   │   ├── camera-vtransform.cu/.hpp # TRT engine：camera_bev → camera_bevfeat
│   │   ├── transfusion.cu/.hpp       # TRT engine：cam_bevfeat + lidar_bev → decoder_feat
│   │   ├── head-transbbox.cu/.hpp    # TRT engine：decoder_feat → 3D boxes（decode kernel）
│   │   ├── lidar-voxelization.cu/.hpp # CUDA：点云体素化（max 160000 voxels）
│   │   ├── lidar-scn.cpp/.hpp        # libspconv 稀疏卷积推理
│   │   └── lidar-scn-onnx-parser.cpp/.hpp # protobuf 解析 SCN onnx 权重
│   ├── common/             # tensor / tensorrt / check / timer / visualize 基础设施
│   ├── onnx/               # onnx protobuf 生成代码（无需 protoc）
│   ├── plugins/            # custom_layernorm TRT 插件（编译为 .so，main 内 dlopen）
│   └── main.cpp            # 入口：读 example-data（jpg + tensor），forward + visualize
├── libs/
│   ├── libspconv/          # 3DSparseConvolution：include + 预编译 .so
│   │   └── lib/x86_64_cuda12.8/libspconv.so   # 真实二进制（仅 cuda12.8）
│   ├── cuosd/              # 可视化标注（BEV + 6 相机画框）
│   └── spconv-onnx/        # 3DSparseConvolution 的 onnx parser 参考
├── dependencies/
│   ├── stb/                # 图像解码（stb_image / stb_image_write）
│   └── dlpack/
└── tool/
    ├── environment.sh      # 环境变量（TensorRT / CUDA / SPCONV 路径）
    ├── build_trt_engine.sh # trtexec 从 onnx 构建 engine（参考脚本）
    └── run.sh              # cmake + make + 运行
```

## 前置依赖

- CUDA 12.8：/usr/local/cuda-12.8
- TensorRT 10.9：/opt/TensorRT-10.9.0.34
- libprotobuf-dev：apt install libprotobuf-dev
- libspconv 预编译库：libs/libspconv/lib/x86_64_cuda12.8/libspconv.so（真实二进制入库，
  仅支持 CUDA 12.x）

## 构建

```bash
cd 3rdparty/deeplearning/deploy/cpp
. tool/environment.sh
bash tool/run.sh
```

产物：
- build/bevfusion          可执行
- build/bevfusion_core.so  主库（链接 libnvinfer / libspconv / protobuf）
- build/custom_layernorm.so  layernorm 插件（main 内 dlopen 加载）

CMake 使用现代 CUDA 语言支持（enable_language(CUDA) + add_library，不依赖已废弃的
FindCUDA），sm80/86/89/90 四档 gencode，兼容 cmake 3.22+。

## 模型导出与布局

engine 构建见 [../python/README.md](../python/README.md)（deploy/export_bf.py）。main.cpp 按以下
布局读取（model/<name>/ 由运行时参数指定）：

model/<name>/build/camera.backbone.plan
model/<name>/build/camera.vtransform.plan
model/<name>/build/fuser.plan
model/<name>/build/head.bbox.plan
model/<name>/lidar.backbone.xyz.onnx     # libspconv 运行时解析，不走 TRT

```bash
cd 3rdparty/deeplearning
mkdir -p model/resnet50/build
cp outputs/trt_bf/*.plan model/resnet50/build/
cp outputs/onnx_bf/lidar.backbone.xyz.onnx model/resnet50/
```

## 数据（example-data）

输入契约（main.cpp 按固定文件名读取）：
- 0-FRONT.jpg ~ 5-BACK_RIGHT.jpg：6 张原始相机图（1600x900）
- camera2lidar.tensor：(6,4,4) float32 相机外参
- camera_intrinsics.tensor：(6,4,4) float32 相机内参
- lidar2image.tensor：(6,4,4) float32 投影矩阵（visualize 用）
- img_aug_matrix.tensor：(6,4,4) float32 图像增强矩阵
- points.tensor：(N,5) float16 点云（x,y,z,intensity,time）

生成：

```bash
cd 3rdparty/deeplearning && python deploy/tools/export_example_data.py configs/det_convfuser.py ckpts/bevfusion-det-bf.pth --sample-index 0 --out-dir outputs/example-data
```

## 运行

```bash
cd 3rdparty/deeplearning/deploy/cpp
./build/bevfusion ../../outputs/example-data resnet50 fp16
```

参数：<data 目录> <模型名> <precision：fp16|int8>。输出 dumped 中间结果
（cam_bev / lidar_bev / decoder_feat .bin）+ 检测可视化 jpg。
