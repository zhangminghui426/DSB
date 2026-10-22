# bevfusion（deploy/cpp，CUDA-BEVFusion 移植）

NVIDIA-AI-IOT/CUDA-BEVFusion 推理代码完整移植，稀疏卷积用 3DSparseConvolution
预编译 libspconv（libs/libspconv/lib/x86_64_cuda12.8）。

## 目录结构

- src/bevfusion/：主流程与算子（原样移植，保留 MIT license）
  - bevfusion.cpp/.hpp：Core 编排（normalize → backbone → bevpool → vtransform → fusion → head）
  - camera-backbone.cu/.hpp：TRT engine（img + depth → camera_depth_weights + camera_feature）
  - camera-depth.cu/.hpp：CUDA 算子（lidar 点云投影生成深度图）
  - camera-normalization.cu/.hpp：CUDA 算子（图像归一化/缩放）
  - camera-geometry.cu/.hpp：CUDA 算子（frustum 反投影 + 排名）
  - camera-bevpool.cu/.hpp：CUDA 算子（depth_weights × feature 池化求和）
  - camera-vtransform.cu/.hpp：TRT engine（camera_bev → camera_bevfeat）
  - transfusion.cu/.hpp：TRT engine（cam_bevfeat + lidar_bev → decoder_feat）
  - head-transbbox.cu/.hpp：TRT engine（decoder_feat → 3D boxes）
  - lidar-voxelization.cu/.hpp：CUDA 算子（点云体素化）
  - lidar-scn.cpp/.hpp：libspconv 稀疏卷积推理（onnx 权重解析）
  - lidar-scn-onnx-parser.cpp/.hpp：protobuf 解析 SCN onnx 权重
- src/common/：tensor / tensorrt / check / timer / visualize 基础设施
- src/onnx/：onnx protobuf（已生成，无需 protoc）
- src/plugins/：custom_layernorm TRT 插件
- src/main.cpp：入口（读 example-data，forward + visualize）
- libs/libspconv/：3DSparseConvolution（include + 预编译 .so，支持 cuda11.4/cuda12.8）
- libs/cuosd/：可视化标注库
- dependencies/：stb（图像解码）、dlpack
- tool/：environment.sh（环境变量）、build_trt_engine.sh（trtexec 构建）、run.sh（构建+运行）

## 前置依赖

- CUDA 12.8（/usr/local/cuda-12.8）、TensorRT 10.9（/opt/TensorRT-10.9.0.34）
- libprotobuf-dev：apt install libprotobuf-dev

## 构建 + 运行

cd 3rdparty/deeplearning/deploy/cpp
. tool/environment.sh        # 设 TensorRT/CUDA/SPCONV 路径
bash tool/run.sh             # cmake + make + 运行

产物：build/bevfusion、build/bevfusion_core.so、build/custom_layernorm.so
（main 内 dlopen 加载 layernorm 插件）。

## 模型布局

main.cpp 按 CUDA-BEVFusion 布局读取（engine 构建见 deploy/python/README.md）：

- model/<name>/build/camera.backbone.plan
- model/<name>/build/camera.vtransform.plan
- model/<name>/build/fuser.plan
- model/<name>/build/head.bbox.plan
- model/<name>/lidar.backbone.xyz.onnx（SCN 由 libspconv 运行时解析，不走 TRT）

cd 3rdparty/deeplearning
mkdir -p model/resnet50/build
cp outputs/trt_bf/*.plan model/resnet50/build/
cp outputs/onnx_bf/lidar.backbone.xyz.onnx model/resnet50/

## 数据

example-data/：6 张相机 jpg（0-FRONT.jpg ~ 5-BACK_RIGHT.jpg）+ camera2lidar.tensor、
camera_intrinsics.tensor、lidar2image.tensor、img_aug_matrix.tensor、points.tensor。

cd 3rdparty/deeplearning
python deploy/tools/export_example_data.py configs/det_convfuser.py ckpts/bevfusion-det-bf.pth \
    --sample-index 0 --out-dir outputs/example-data

## 运行

cd 3rdparty/deeplearning/deploy/cpp
./build/bevfusion ../../outputs/example-data resnet50 fp16
