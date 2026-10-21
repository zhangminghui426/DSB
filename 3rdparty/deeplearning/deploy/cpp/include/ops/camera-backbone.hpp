#pragma once
// 相机 BEV 下采样 CUDA 算子（对应 CUDA-BEVFusion camera 分支 backbone 之后的
// BEV 处理）：逐层 3x3 conv + BN + ReLU，替代 CPU 版 downsample_conv。
// 调用方负责设备内存与 stream；函数在传入 stream 上异步执行，调用方负责同步。

#include <cuda_runtime.h>

#include <cstdint>

namespace dsb {

// 每层 conv 权重布局（blob，设备端连续）：
//   [w (out_c*k*k*in_c) | bias (out_c) | g (out_c) | b (out_c)
//    | mean (out_c) | var (out_c)]，自 w_off 起
struct CudaDownsampleLayer {
  int in_c, out_c, k, s, p;
  long long w_off, w_len, b_len;
  float bn_eps;
};

// 逐层 3x3 conv+bn+relu：bev (1,C_in,H,W) 设备端 -> out (1,C_out,H_out,W_out)。
// layers 可位于主机端（kernel 每层 launch 时读取）；blob 为设备端权重。
// 内部用 tmp_a/tmp_b 作中间缓冲（n_layers>=2 时）。
void cuda_downsample(const CudaDownsampleLayer *layers, int n_layers,
                     const float *blob, float *bev, float *tmp_a, float *tmp_b,
                     float *out, int C_in, int H, int W, cudaStream_t stream);

}  // namespace dsb
