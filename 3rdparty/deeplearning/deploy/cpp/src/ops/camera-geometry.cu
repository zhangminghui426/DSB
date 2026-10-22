// 相机分支几何 CUDA 算子（对齐 NVIDIA-AI-IOT/CUDA-BEVFusion camera-geometry）。
// 数值语义与 src/ops/ops.cpp CPU 参考 compute_geometry 逐位对齐（fp32）。

#include "ops/camera-geometry.hpp"

#include <math.h>

namespace dsb {

namespace {

// 每线程一个 frustum 点，内循环 6 相机；输出 (N, D, fH, fW, 3)。
// 相机侧几何等价于 CPU：xs=(W-1)*i/(fW-1)、ys=(H-1)*i/(fH-1)、ds=dbound0+dbound2*d；
//   u = post_inv*[xs,ys,ds] - post_inv_t；cam = [u0*u2, u1*u2, u2]；
//   w = combine*cam + c2l_t；e = la_rot*w + la_t。
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
    const float u0 = pm[0] * xs + pm[1] * ys + pm[2] * ds - pm[9];
    const float u1 = pm[3] * xs + pm[4] * ys + pm[5] * ds - pm[10];
    const float u2 = pm[6] * xs + pm[7] * ys + pm[8] * ds - pm[11];
    const float cam0 = u0 * u2, cam1 = u1 * u2, cam2 = u2;
    const float *cm = pm + 12;
    const float w0 = cm[0] * cam0 + cm[1] * cam1 + cm[2] * cam2 + pm[21];
    const float w1 = cm[3] * cam0 + cm[4] * cam1 + cm[5] * cam2 + pm[22];
    const float w2 = cm[6] * cam0 + cm[7] * cam1 + cm[8] * cam2 + pm[23];
    float *gout = geom + ((long long)c * total + tid) * 3;
    gout[0] = p.la[0] * w0 + p.la[1] * w1 + p.la[2] * w2 + p.la[9];
    gout[1] = p.la[3] * w0 + p.la[4] * w1 + p.la[5] * w2 + p.la[10];
    gout[2] = p.la[6] * w0 + p.la[7] * w1 + p.la[8] * w2 + p.la[11];
  }
}

}  // namespace

void cuda_compute_geometry(const CudaGeometryParams &p, float *geom,
                           cudaStream_t stream) {
  const int total = p.fH * p.fW * p.D;
  const int threads = 256;
  const int blocks = (total + threads - 1) / threads;
  geometry_kernel<<<blocks, threads, 0, stream>>>(p, geom);
}

}  // namespace dsb
