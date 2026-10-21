// 相机 BEV 下采样 CUDA 算子实现（对应 CUDA-BEVFusion camera 分支 BEV 处理）。
// 数值语义与 src/ops/ops.cpp CPU 参考（downsample_conv）逐位对齐：bias 起步、
// 3x3 滑窗累加、BN(acc-mean)/sqrt(var+eps)*g+b、ReLU。

#include "ops/camera-backbone.hpp"

#include <math.h>

namespace dsb {

namespace {

// 每线程一个输出元素 (co, ho, wo)；3x3 conv + bn + relu。
__global__ void downsample_kernel(const float *cur, const float *wgt,
                                  const float *bias, const float *g,
                                  const float *bta, const float *mean,
                                  const float *var, float bn_eps, int C_in,
                                  int C_out, int H, int Wd, int k, int s, int p,
                                  float *nxt) {
  const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  const int Ho = (H - k + 2 * p) / s + 1, Wo = (Wd - k + 2 * p) / s + 1;
  if (idx >= (long long)C_out * Ho * Wo) return;
  const int wo = (int)(idx % Wo);
  const int ho = (int)((idx / Wo) % Ho);
  const int co = (int)(idx / ((long long)Wo * Ho));
  float acc = bias[co];
  const float *Wco = wgt + (long long)co * C_in * k * k;
  for (int ci = 0; ci < C_in; ++ci) {
    const float *cur_ci = cur + (long long)ci * H * Wd;
    const float *Wci = Wco + (long long)ci * k * k;
    for (int ki = 0; ki < k; ++ki) {
      const int hi = ho * s - p + ki;
      if (hi < 0 || hi >= H) continue;
      const float *row = cur_ci + (long long)hi * Wd;
      for (int kj = 0; kj < k; ++kj) {
        const int wi = wo * s - p + kj;
        if (wi < 0 || wi >= Wd) continue;
        acc += Wci[ki * k + kj] * row[wi];
      }
    }
  }
  const float s2 = var[co] + bn_eps;
  const float v = (acc - mean[co]) / sqrtf(s2) * g[co] + bta[co];
  nxt[idx] = fmaxf(v, 0.0f);  // ReLU
}

}  // namespace

void cuda_downsample(const CudaDownsampleLayer *layers, int n_layers,
                     const float *blob, float *bev, float *tmp_a, float *tmp_b,
                     float *out, int C_in, int H, int W, cudaStream_t stream) {
  const float *cur = bev;
  for (int li = 0; li < n_layers; ++li) {
    const CudaDownsampleLayer &ly = layers[li];
    const float *wgt = blob + ly.w_off;
    const float *bias = blob + ly.w_off + ly.w_len;
    const float *g = blob + ly.w_off + ly.w_len + ly.b_len;
    const float *bta = blob + ly.w_off + ly.w_len + 2 * ly.b_len;
    const float *mean = blob + ly.w_off + ly.w_len + 3 * ly.b_len;
    const float *var = blob + ly.w_off + ly.w_len + 4 * ly.b_len;
    const int Ho = (H - ly.k + 2 * ly.p) / ly.s + 1;
    const int Wo = (W - ly.k + 2 * ly.p) / ly.s + 1;
    float *dst = (li == n_layers - 1) ? out
                : (li == 0)           ? tmp_a
                : ((li % 2) == 1)     ? tmp_b
                                      : tmp_a;
    const long long total = (long long)ly.out_c * Ho * Wo;
    const int threads = 256;
    const long long blocks = (total + threads - 1) / threads;
    downsample_kernel<<<(int)blocks, threads, 0, stream>>>(
        cur, wgt, bias, g, bta, mean, var, ly.bn_eps, ly.in_c, ly.out_c, H, W,
        ly.k, ly.s, ly.p, dst);
    cur = dst;
    C_in = ly.out_c;
    H = Ho;
    W = Wo;
  }
}

}  // namespace dsb
