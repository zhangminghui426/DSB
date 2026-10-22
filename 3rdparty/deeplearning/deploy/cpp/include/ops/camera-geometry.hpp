#pragma once
// 相机分支几何 CUDA 算子（对齐 NVIDIA-AI-IOT/CUDA-BEVFusion camera-geometry）：
// frustum 网格点反投影到 lidar 系世界坐标。
// 调用方负责设备内存与 stream；函数在传入 stream 上异步执行，调用方负责同步。
// 数值语义与 src/ops/ops.cpp CPU 参考 compute_geometry 一致。

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

// geom 设备端 (N*D*fH*fW*3)（布局与 CPU 版一致：i = n*D*area + d*area + hw）
void cuda_compute_geometry(const CudaGeometryParams &p, float *geom,
                           cudaStream_t stream);

}  // namespace dsb
