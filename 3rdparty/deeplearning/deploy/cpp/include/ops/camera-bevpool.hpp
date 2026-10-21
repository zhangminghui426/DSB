#pragma once
// 相机分支 CUDA 算子（参考 NVIDIA-AI-IOT/CUDA-BEVFusion camera-bevpool.cu 组织）：
//   1. cuda_compute_geometry：frustum 点反投影到 lidar 系（每线程 1 点 x 6 相机）
//   2. cuda_fold_bevpool：深度 softmax 折入特征 + 网格量化 + atomicAdd 求和
//      （合并 CPU 版 fold_depth_features + bev_pool，免 x 中间张量）
// 调用方负责设备内存与 stream；函数在传入 stream 上异步执行，调用方负责同步。
// 数值语义与 src/ops/ops.cpp CPU 参考一致（fold 后按 rank 求和、无归一化）。

#include <cuda_runtime.h>

#include <cstdint>

namespace dsb {

// geometry kernel 输入：per_cam[c] 打包 24 floats：
//   [post_inv 9 | post_inv_t 3(=post_inv*post_t) | combine 9(=c2l*intr_inv) | c2l_t 3]
//   la[12] = [la_rot 9 | la_t 3]
struct CudaGeometryParams {
  int D, fH, fW, N;
  float image_w, image_h;
  float dbound0, dbound2;
  float bx, by, bz;     // x/y/z bound 起点
  float dx_, dy_, dz_;  // x/y/z bound 步长
  int nx0, nx1, nx2;
  float per_cam[6 * 24];
  float la[12];
};

// fold_bevpool kernel 输入：depthnet (N*(D+C)*area)、geom (N*D*fH*fW*3)
struct CudaFoldPoolParams {
  int N, D, C, area;  // area = fH*fW
  int nx0, nx1, nx2;
  float bx, by, bz, dx_, dy_, dz_;
};

// geom 设备端 (N*D*fH*fW*3)（布局与 CPU 版一致：i = n*D*area + d*area + hw）
void cuda_compute_geometry(const CudaGeometryParams &p, float *geom,
                           cudaStream_t stream);

// out 设备端 (C*nx0*nx1)；depthnet 设备端 (N*(D+C)*area)
void cuda_fold_bevpool(const CudaFoldPoolParams &p, const float *depthnet,
                       const float *geom, float *out, cudaStream_t stream);

}  // namespace dsb
