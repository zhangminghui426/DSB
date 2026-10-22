// 相机分支 fold+BEV 池化 CUDA 算子（对齐 NVIDIA-AI-IOT/CUDA-BEVFusion
// camera-bevpool）。数值语义与 src/ops/ops.cpp CPU 参考逐位对齐（fp32；
// rank 按 (x,y) 求和、Z 维折叠、不归一化）。

#include "ops/camera-bevpool.hpp"

#include <math.h>

namespace dsb {

namespace {

// ------------------------------------------------ fold depth + bev_pool 合并
// 每线程一个几何点 i（= n*D*area + d*area + hw）：
//   softmax(depthnet 深度 logits) -> depth_p；out[c, (cx,cy)] += depth_p * feat[c]
// 网格坐标含 Z 仅作范围过滤，输出按 (cx,cy) 折叠（Dz=nx2=1，与 CPU 一致）。
__global__ void fold_bevpool_kernel(const CudaFoldPoolParams p,
                                    const float *dn, const float *geom,
                                    float *out) {
  const long long i =
      (long long)blockIdx.x * blockDim.x + threadIdx.x;
  const long long total = (long long)p.N * p.D * p.area;
  if (i >= total) return;
  const int n = (int)(i / ((long long)p.D * p.area));
  const long long rem = i % ((long long)p.D * p.area);
  const int d = (int)(rem / p.area);
  const int hw = (int)(rem % p.area);

  const float *logits = dn + (long long)n * ((p.D + p.C) * p.area) + hw;
  float mx = -1e30f;
  for (int dd = 0; dd < p.D; ++dd)
    mx = fmaxf(mx, logits[(long long)dd * p.area]);
  float sum = 0.0f;
  float e[128];  // D <= 128
  for (int dd = 0; dd < p.D; ++dd) {
    e[dd] = __expf(logits[(long long)dd * p.area] - mx);
    sum += e[dd];
  }
  const float dp = e[d] / sum;

  const float *g = geom + i * 3;
  const int cx = (int)floorf((g[0] - (p.bx - p.dx_ * 0.5f)) / p.dx_);
  const int cy = (int)floorf((g[1] - (p.by - p.dy_ * 0.5f)) / p.dy_);
  const int cz = (int)floorf((g[2] - (p.bz - p.dz_ * 0.5f)) / p.dz_);
  if (cx < 0 || cx >= p.nx0 || cy < 0 || cy >= p.nx1 || cz < 0 || cz >= p.nx2)
    return;
  // 输出 BEV 只保留 (x,y) 网格，Z 维仅做范围过滤后折叠累加
  // （与 CPU 参考 ops.cpp bev_pool 一致：rank 排序按 (cx,cy) 求和，Dz=1）。
  // 不能使用含 cz 的 rank：out 只分配了 C*nx0*nx1，cz>0 会越界写。
  const long long rank = (long long)cx * p.nx1 + cy;
  const long long nbev = (long long)p.nx0 * p.nx1;

  const float *feat = dn + (long long)n * ((p.D + p.C) * p.area) +
                      (long long)p.D * p.area + hw;
  for (int c = 0; c < p.C; ++c)
    atomicAdd(out + (long long)c * nbev + rank, dp * feat[(long long)c * p.area]);
}

}  // namespace

void cuda_fold_bevpool(const CudaFoldPoolParams &p, const float *depthnet,
                       const float *geom, float *out, cudaStream_t stream) {
  const long long total = (long long)p.N * p.D * p.area;
  const int threads = 256;
  const long long blocks = (total + threads - 1) / threads;
  fold_bevpool_kernel<<<(int)blocks, threads, 0, stream>>>(p, depthnet, geom,
                                                           out);
}

}  // namespace dsb
