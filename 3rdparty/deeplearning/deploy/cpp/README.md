# bevfusion（deploy/cpp，CUDA-BEVFusion 移植）

NVIDIA-AI-IOT/CUDA-BEVFusion 推理代码完整移植：src/bevfusion/（bevfusion.cpp 编排 +
camera-backbone/depth/normalization/geometry/bevpool/vtransform + transfusion +
head-transbbox + lidar-voxelization/lidar-scn + onnx-parser）、src/common/、
src/onnx/、src/plugins/、src/main.cpp。稀疏卷积用 3DSparseConvolution 预编译
libspconv（libs/libspconv/lib/x86_64_cuda12.8）。

## 构建 + 运行

前置：CUDA 12.8、TensorRT 10.9、libprotobuf-dev（apt install libprotobuf-dev）。

cd 3rdparty/deeplearning/deploy/cpp
. tool/environment.sh
bash tool/run.sh            # cmake + make + 运行

产物：build/bevfusion、build/bevfusion_core.so、build/custom_layernorm.so。

## 模型布局

main.cpp 读取 model/<name>/build/{camera.backbone,camera.vtransform,fuser,head.bbox}.plan
和 model/<name>/lidar.backbone.xyz.onnx（SCN 由 libspconv 运行时解析，不走 TRT）。

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
