#pragma once
// DSB 推理 CUDA 自定义算子（参考 NVIDIA-AI-IOT/CUDA-BEVFusion 的 kernel 组织）：
//   cuda_compute_geometry / cuda_fold_bevpool / cuda_downsample
// 调用方负责设备内存与 stream；函数在传入 stream 上异步执行，调用方负责同步。
// 数值语义与 src/ops/ops.cpp 的 CPU 参考实现一致（fold 后按 rank 求和、无归一化）。

#include <cuda_runtime.h>

namespace dsb {

struct CudaGeometryParams {
  int D, fH, fW, N;
  float image_w, image_h;
  float dbound0, dbound2;
  float bx, by, bz;     // x/y/z bound 起点
  float dx_, dy_, dz_;  // x/y/z bound 步长
  int nx0, nx1, nx2;
  // per_cam[c] 打包 24 floats：
  //   [post_inv 9 | post_inv_t 3(=post_inv*post_t) | combine 9(=c2l*intr_inv) | c2l_t 3]
  float per_cam[6 * 24];
  // lidar_aug: [la_rot 9 | la_t 3]
  float la[12];
};

struct CudaFoldPoolParams {
  int N, D, C, area;  // area = fH*fW
  int nx0, nx1, nx2;
  float bx, by, bz, dx_, dy_, dz_;
};

struct CudaDownsampleLayer {
  int in_c, out_c, k, s, p;
  long long w_off, w_len, b_len;
  float bn_eps;
};

// geom 设备端 (N*D*fH*fW*3)（布局与 CPU 版一致：i = n*D*area + d*area + hw）
void cuda_compute_geometry(const CudaGeometryParams &p, float *geom,
                           cudaStream_t stream);

// out 设备端 (C*nx0*nx1)；depthnet 设备端 (N*(D+C)*area)
void cuda_fold_bevpool(const CudaFoldPoolParams &p, const float *depthnet,
                       const float *geom, float *out, cudaStream_t stream);

// 逐层 3x3 conv+bn+relu：bev (1,C_in,H,W) 设备端 -> out (1,C_out,H_out,W_out)。
// layers 可位于主机端（kernel 每层 launch 时读取）；blob 为设备端权重。
// 内部用 tmp_a/tmp_b 作中间缓冲（n_layers>=2 时）。
void cuda_downsample(const CudaDownsampleLayer *layers, int n_layers,
                     const float *blob, float *bev, float *tmp_a, float *tmp_b,
                     float *out, int C_in, int H, int W, cudaStream_t stream);

}  // namespace dsb
