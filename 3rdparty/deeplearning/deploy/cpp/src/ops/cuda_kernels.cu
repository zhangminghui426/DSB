// DSB 推理 CUDA 自定义算子实现。
// 参考 NVIDIA-AI-IOT/Lidar_AI_Solution/CUDA-BEVFusion：
//   camera-geometry.cu（几何）、camera-bevpool.cu（BEV pooling）、
//   camera-vtransform.cu（depth*feat 折入）——本项目合并为两个 kernel。
// 数值语义与 src/ops/ops.cpp CPU 参考逐位对齐（fp32；rank 求和不归一化）。

#include "ops/cuda_kernels.hpp"

#include <cuda_fp16.h>
#include <math.h>

namespace dsb {

namespace {

// ------------------------------------------------------------------ geometry
// 每线程一个 frustum 点，内循环 6 相机；输出 (N, D, fH, fW, 3)。
__global__ void geometry_kernel(const CudaGeometryParams p, float *geom) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = p.fH * p.fW * p.D;
  if (tid >= total) return;
  const int wi = tid % p.fW;
  const int hi = (tid / p.fW) % p.fH;
  const int di = tid / (p.fW * p.fH);
  const float xs = (p.image_w - 1.0f) * wi / (p.fW - 1.0f);
  const float ys = (p.image_h - 1.0f) * hi / (p.fH - 1.0f);
  const float ds = p.dbound0 + p.dbound2 * di;

  for (int c = 0; c < p.N; ++c) {
    const float *pm = p.per_cam + c * 24;
    // u = post_inv * [xs, ys, ds] - post_inv_t
    const float u0 = pm[0] * xs + pm[1] * ys + pm[2] * ds - pm[9];
    const float u1 = pm[3] * xs + pm[4] * ys + pm[5] * ds - pm[10];
    const float u2 = pm[6] * xs + pm[7] * ys + pm[8] * ds - pm[11];
    // cam = [u0*u2, u1*u2, u2]
    const float cam0 = u0 * u2, cam1 = u1 * u2, cam2 = u2;
    // w = combine * cam + c2l_t
    const float *cm = pm + 12;
    const float w0 = cm[0] * cam0 + cm[1] * cam1 + cm[2] * cam2 + pm[21];
    const float w1 = cm[3] * cam0 + cm[4] * cam1 + cm[5] * cam2 + pm[22];
    const float w2 = cm[6] * cam0 + cm[7] * cam1 + cm[8] * cam2 + pm[23];
    // e = la_rot * w + la_t
    float *gout = geom + ((long long)c * total + tid) * 3;
    gout[0] = p.la[0] * w0 + p.la[1] * w1 + p.la[2] * w2 + p.la[9];
    gout[1] = p.la[3] * w0 + p.la[4] * w1 + p.la[5] * w2 + p.la[10];
    gout[2] = p.la[6] * w0 + p.la[7] * w1 + p.la[8] * w2 + p.la[11];
  }
}

// ------------------------------------------------ fold depth + bev_pool 合并
// 每线程一个几何点 i（= n*D*area + d*area + hw）：
//   softmax(depthnet 深度 logits) -> depth_p；out[c, rank(i)] += depth_p * feat[c]
// 一次 kernel 完成 CPU 版的 fold_depth_features + bev_pool（省 x 中间张量）。
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
  const long long nbev = (long long)p.nx0 * p.nx1;
  const long long rank = (long long)cz * nbev + (long long)cx * p.nx1 + cy;

  const float *feat = dn + (long long)n * ((p.D + p.C) * p.area) +
                      (long long)p.D * p.area + hw;
  for (int c = 0; c < p.C; ++c)
    atomicAdd(out + (long long)c * nbev + rank, dp * feat[(long long)c * p.area]);
}

// ---------------------------------------------------------------- downsample
// 每线程一个输出元素 (co, ho, wo)；3x3 conv + bn + relu。
__global__ void downsample_kernel(const float *cur, const float *W,
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
  const float *Wco = W + (long long)co * C_in * k * k;
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

// ------------------------------------------------------------------- public
void cuda_compute_geometry(const CudaGeometryParams &p, float *geom,
                           cudaStream_t stream) {
  const int total = p.fH * p.fW * p.D;
  const int threads = 256;
  const int blocks = (total + threads - 1) / threads;
  geometry_kernel<<<blocks, threads, 0, stream>>>(p, geom);
}

void cuda_fold_bevpool(const CudaFoldPoolParams &p, const float *depthnet,
                       const float *geom, float *out, cudaStream_t stream) {
  const long long total = (long long)p.N * p.D * p.area;
  const int threads = 256;
  const long long blocks = (total + threads - 1) / threads;
  fold_bevpool_kernel<<<(int)blocks, threads, 0, stream>>>(p, depthnet, geom,
                                                           out);
}

void cuda_downsample(const CudaDownsampleLayer *layers, int n_layers,
                     const float *blob, float *bev, float *tmp_a, float *tmp_b,
                     float *out, int C_in, int H, int W, cudaStream_t stream) {
  const float *cur = bev;
  for (int li = 0; li < n_layers; ++li) {
    const CudaDownsampleLayer &ly = layers[li];
    const float *W = blob + ly.w_off;
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
        cur, W, bias, g, bta, mean, var, ly.bn_eps, ly.in_c, ly.out_c, H, W,
        ly.k, ly.s, ly.p, dst);
    cur = dst;
    C_in = ly.out_c;
    H = Ho;
    W = Wo;
  }
}

}  // namespace dsb
