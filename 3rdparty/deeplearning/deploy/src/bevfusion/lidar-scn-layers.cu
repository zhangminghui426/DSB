/*
 * Per-layer SCN kernels: ReLU, residual Add and final sparse->BEV fold.
 */
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "bevfusion/lidar-scn-layers.hpp"

namespace bevfusion {
namespace lidar {

static __global__ void relu_kernel(half* data, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float v = __half2float(data[i]);
  data[i] = __float2half(v > 0.0f ? v : 0.0f);
}

static __global__ void add_kernel(half* dst, const half* src, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  dst[i] = __float2half(__half2float(dst[i]) + __half2float(src[i]));
}

// feat (M, C) + idx (M, 4 = [b, x, y, z]) -> out (C*Z, X, Y)
// 与 SparseEncoder.forward 的 dense() + permute(0,1,4,2,3) + reshape 对齐：
// channel = c * Z + z，空间维为 (x, y)。
static __global__ void scatter_fold_kernel(const half* feat, const int* idx, half* out,
                                           int M, int C, int X, int Y, int Z) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M * C) return;
  int p = i / C;
  int c = i - p * C;
  int x = idx[p * 4 + 1];
  int y = idx[p * 4 + 2];
  int z = idx[p * 4 + 3];
  if (x < 0 || x >= X || y < 0 || y >= Y || z < 0 || z >= Z) return;
  int ch = c * Z + z;
  out[((size_t)ch * X + x) * Y + y] = feat[i];
}

void scn_relu_half(nvtype::half* data, size_t n, void* stream) {
  if (n == 0) return;
  int block = 256;
  int grid = (int)((n + block - 1) / block);
  relu_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(reinterpret_cast<half*>(data), n);
}

void scn_add_half(nvtype::half* dst, const nvtype::half* src, size_t n, void* stream) {
  if (n == 0) return;
  int block = 256;
  int grid = (int)((n + block - 1) / block);
  add_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
      reinterpret_cast<half*>(dst), reinterpret_cast<const half*>(src), n);
}

void scn_scatter_fold(const nvtype::half* feat, const int* idx, nvtype::half* out,
                      int M, int C, int X, int Y, int Z, void* stream) {
  cudaStream_t s = (cudaStream_t)stream;
  size_t total = (size_t)C * Z * X * Y;
  cudaMemsetAsync(out, 0, total * sizeof(half), s);
  int n = M * C;
  if (n <= 0) return;
  int block = 256;
  int grid = (int)((n + block - 1) / block);
  scatter_fold_kernel<<<grid, block, 0, s>>>(
      reinterpret_cast<const half*>(feat), idx, reinterpret_cast<half*>(out),
      M, C, X, Y, Z);
}

};  // namespace lidar
};  // namespace bevfusion
