#pragma once
// 相机分支 fold+BEV 池化 CUDA 算子（对齐 NVIDIA-AI-IOT/CUDA-BEVFusion
// camera-bevpool）：深度 softmax 折入特征 + 网格量化 + atomicAdd 求和。
// 合并 CPU 版 fold_depth_features + bev_pool，免 x 中间张量。
// 调用方负责设备内存与 stream；函数在传入 stream 上异步执行，调用方负责同步。
// 数值语义与 src/ops/ops.cpp CPU 参考一致（fold 后按 (x,y) rank 求和、Z 折叠）。

#include <cuda_runtime.h>

#include <cstdint>

namespace dsb {

// fold_bevpool kernel 输入：depthnet (N*(D+C)*area)、geom (N*D*fH*fW*3)
// （geom 由 ops/camera-geometry.cu cuda_compute_geometry 生成）
struct CudaFoldPoolParams {
  int N, D, C, area;  // area = fH*fW
  int nx0, nx1, nx2;
  float bx, by, bz, dx_, dy_, dz_;
};

// out 设备端 (C*nx0*nx1)；depthnet 设备端 (N*(D+C)*area)；geom 设备端
void cuda_fold_bevpool(const CudaFoldPoolParams &p, const float *depthnet,
                       const float *geom, float *out, cudaStream_t stream);

}  // namespace dsb
