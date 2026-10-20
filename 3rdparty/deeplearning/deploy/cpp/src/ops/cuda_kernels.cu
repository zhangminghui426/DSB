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

// ============================================================== sparse 链
// 与 CPU 参考（ops.cpp sparse_encoder_forward）同语义：
//   subm：输出体素 == 输入体素，查 hash 找邻居；
//   strided：out = (in + pad - offset)/stride（可整除维），输出需唯一化；
//   稠密化：(C*Z, X, Y) 布局 out[(c*Z+z)*XY + x*Y + y] = feat[voxel(x,y,z)][c]。

namespace {

// 坐标 key：+4096 偏移保证非负。subm 查询越界邻居坐标会为负，
// 若不偏移则负 key 与 hash 空哨兵（-1）冲突，lookup 会误判命中。
__device__ long long key3(int x, int y, int z) {
  return (((long long)(x + 4096)) << 40) |
         (((long long)(y + 4096)) << 20) | ((long long)(z + 4096));
}

// hash：pow2 大小，线性探测；keys 哨兵 -1（坐标 key 恒非负）。
__device__ int hash_slot(long long key, int hash_size) {
  return (int)((key * 2654435761ULL) & (hash_size - 1));
}

__device__ void hash_insert(long long *keys, int *idx, int hash_size,
                            long long key, int v) {
  int h = hash_slot(key, hash_size);
  while (true) {
    long long old = atomicCAS((long long *)&keys[h], -1LL, key);
    if (old == -1LL) {          // 插入成功
      idx[h] = v;
      return;
    }
    if (old == key) {           // 已存在（并发重复）
      idx[h] = v;
      return;
    }
    h = (h + 1) & (hash_size - 1);
  }
}

__device__ int hash_lookup(const long long *keys, const int *idx,
                           int hash_size, long long key) {
  int h = hash_slot(key, hash_size);
  while (keys[h] != key) {
    if (keys[h] == -1LL) return -1;
    h = (h + 1) & (hash_size - 1);
  }
  return idx[h];
}

// 每线程一个输入体素，写入 hash 表（输入体素坐标唯一）。
__global__ void sparse_build_hash_kernel(const int *coords, int nvox,
                                         long long *keys, int *idx,
                                         int hash_size) {
  const int u = blockIdx.x * blockDim.x + threadIdx.x;
  if (u >= nvox) return;
  const long long key =
      key3(coords[u * 3], coords[u * 3 + 1], coords[u * 3 + 2]);
  hash_insert(keys, idx, hash_size, key, u);
}

// subm conv：每线程 (vi, co)。acc = bias + sum_k sum_ci w[co,k,ci]*x[nbr,ci]。
__global__ void sparse_subm_kernel(const CudaSparseLayer ly,
                                   const int *offsets_all, const float *blob,
                                   const int *coords, const float *in_feats,
                                   int nvox, const long long *keys,
                                   const int *idx, int hash_size,
                                   float *out_feats) {
  const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (long long)nvox * ly.out_c) return;
  const int vi = (int)(t / ly.out_c);
  const int co = (int)(t % ly.out_c);
  const int *off = offsets_all + ly.off_off;
  const float *w = blob + ly.w_off;
  float acc = w[ly.w_len + co];  // bias
  for (int k = 0; k < ly.K; ++k) {
    const int nx = coords[vi * 3] + off[k * 3];
    const int ny = coords[vi * 3 + 1] + off[k * 3 + 1];
    const int nz = coords[vi * 3 + 2] + off[k * 3 + 2];
    const int nbr = hash_lookup(keys, idx, hash_size, key3(nx, ny, nz));
    if (nbr < 0) continue;
    const float *xf = in_feats + (long long)nbr * ly.in_c;
    const float *wf = w + ((long long)co * ly.K + k) * ly.in_c;
    for (int ci = 0; ci < ly.in_c; ++ci) acc += wf[ci] * xf[ci];
  }
  const float *mean = w + ly.w_len + ly.b_len;
  const float *var = w + ly.w_len + 2 * ly.b_len;
  const float *g = w + ly.w_len + 3 * ly.b_len;
  const float *b = w + ly.w_len + 4 * ly.b_len;
  const float v = (acc - mean[co]) / sqrtf(var[co] + ly.bn_eps) * g[co] + b[co];
  out_feats[(long long)vi * ly.out_c + co] = fmaxf(v, 0.0f);
}

// strided pass1：每 (u,k) 生成输出候选，hash 插入并记录 coords。
__global__ void sparse_strided_pass1_kernel(const CudaSparseLayer ly,
                                            const int *offsets_all,
                                            const int *coords_in, int nvox,
                                            long long *keys, int *idx,
                                            int hash_size, int *coords_out,
                                            int *counter) {
  const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  const long long total = (long long)nvox * ly.K;
  if (t >= total) return;
  const int u = (int)(t / ly.K);
  const int k = (int)(t % ly.K);
  const int *off = offsets_all + ly.off_off;
  int oc[3];
  for (int d = 0; d < 3; ++d) {
    const int pad = (d == 0) ? ly.px : ((d == 1) ? ly.py : ly.pz);
    const int st = (d == 0) ? ly.sx : ((d == 1) ? ly.sy : ly.sz);
    const int num = coords_in[u * 3 + d] + pad - off[k * 3 + d];
    if (num % st != 0) return;
    oc[d] = num / st;
  }
  const long long key = key3(oc[0], oc[1], oc[2]);
  const int h = hash_slot(key, hash_size);
  int hh = h;
  while (true) {
    long long old = atomicCAS((long long *)&keys[hh], -1LL, key);
    if (old == -1LL) {          // 新输出体素：分配序号并记录坐标
      const int oi = atomicAdd(counter, 1);
      idx[hh] = oi;
      coords_out[oi * 3] = oc[0];
      coords_out[oi * 3 + 1] = oc[1];
      coords_out[oi * 3 + 2] = oc[2];
      return;
    }
    if (old == key) return;     // 已存在（由其他 (u,k) 建立）
    hh = (hh + 1) & (hash_size - 1);
  }
}

// strided pass2：每 (u,k) 重算输出 idx，原子累加 conv 贡献（bn 前）。
__global__ void sparse_strided_pass2_kernel(const CudaSparseLayer ly,
                                            const int *offsets_all,
                                            const float *blob,
                                            const int *coords_in, int nvox,
                                            const float *in_feats,
                                            const long long *keys,
                                            const int *idx, int hash_size,
                                            float *raw) {
  const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  const long long total = (long long)nvox * ly.K;
  if (t >= total) return;
  const int u = (int)(t / ly.K);
  const int k = (int)(t % ly.K);
  const int *off = offsets_all + ly.off_off;
  int oc[3];
  for (int d = 0; d < 3; ++d) {
    const int pad = (d == 0) ? ly.px : ((d == 1) ? ly.py : ly.pz);
    const int st = (d == 0) ? ly.sx : ((d == 1) ? ly.sy : ly.sz);
    const int num = coords_in[u * 3 + d] + pad - off[k * 3 + d];
    if (num % st != 0) return;
    oc[d] = num / st;
  }
  const int oi = hash_lookup(keys, idx, hash_size, key3(oc[0], oc[1], oc[2]));
  if (oi < 0) return;
  const float *xf = in_feats + (long long)u * ly.in_c;
  const float *w = blob + ly.w_off;
  for (int co = 0; co < ly.out_c; ++co) {
    const float *wf = w + ((long long)co * ly.K + k) * ly.in_c;
    float s = 0.0f;
    for (int ci = 0; ci < ly.in_c; ++ci) s += wf[ci] * xf[ci];
    atomicAdd(raw + (long long)oi * ly.out_c + co, s);
  }
}

// bn + relu：每线程 (oi, co)。raw 为 conv 累加（不含 bias），bias 在此加入。
__global__ void sparse_bn_relu_kernel(const CudaSparseLayer ly,
                                      const float *blob, int n_out,
                                      const float *raw, float *out_feats) {
  const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (long long)n_out * ly.out_c) return;
  const int oi = (int)(t / ly.out_c);
  const int co = (int)(t % ly.out_c);
  const float *w = blob + ly.w_off;
  const float *bias = w + ly.w_len;
  const float *mean = w + ly.w_len + ly.b_len;
  const float *var = w + ly.w_len + 2 * ly.b_len;
  const float *g = w + ly.w_len + 3 * ly.b_len;
  const float *b = w + ly.w_len + 4 * ly.b_len;
  const float v = (raw[t] + bias[co] - mean[co]) /
                      sqrtf(var[co] + ly.bn_eps) * g[co] +
                  b[co];
  out_feats[t] = fmaxf(v, 0.0f);
}

// 稠密化：每线程 (x, y, z)（t = (z*X + x)*Y + y）查最后一层 hash，命中写 C 通道。
__global__ void sparse_densify_kernel(int C_final, int X, int Y, int Z,
                                      const int *coords, const long long *keys,
                                      const int *idx, int hash_size,
                                      const float *feats, float *bev) {
  const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  const long long total = (long long)X * Y * Z;
  if (t >= total) return;
  const int zx = (int)(t / Y);
  const int x = zx % X;
  const int z = zx / X;
  const int y = (int)(t % Y);
  const int v = hash_lookup(keys, idx, hash_size, key3(x, y, z));
  if (v < 0) return;
  const long long xy = (long long)X * Y;
  for (int c = 0; c < C_final; ++c)
    bev[(c * Z + z) * xy + x * Y + y] = feats[(long long)v * C_final + c];
}

}  // namespace

// 逐层：hash 重建 -> subm/strided conv；最后稠密化输出 (C*Z, X, Y)。
// nvox 由调用方提供（第 0 层输入体素数）；strided 层后通过 counter 同步读回
// 新体素数（strided 层数少，同步开销可忽略）。
void cuda_sparse_encode(
    const CudaSparseLayer *layers, int n_layers, const float *blob,
    const int *offsets_all, const float *cur, int nvox, int in_c0,
    const int *coords_in, float *lidar_bev, int X, int Y, int Z,
    float *cur_buf, float *nxt_buf, float *raw_buf, long long *hash_keys,
    int *hash_idx, long long *out_hash_keys, int *out_hash_idx,
    int hash_size, int *coords_buf, int *counter, cudaStream_t stream) {
  const int threads = 256;
  float *curf = const_cast<float *>(cur);
  float *nxtf = nxt_buf;
  const int *coords = coords_in;
  int n_out = nvox;

  for (int li = 0; li < n_layers; ++li) {
    const CudaSparseLayer &ly = layers[li];
    // 重建输入 hash 表
    cudaMemsetAsync(hash_keys, 0xFF, (size_t)hash_size * sizeof(long long),
                    stream);
    const int nb = (n_out + threads - 1) / threads;
    sparse_build_hash_kernel<<<nb, threads, 0, stream>>>(coords, n_out,
                                                         hash_keys, hash_idx,
                                                         hash_size);

    if (ly.type == 0) {
      // subm：输出体素数不变，coords 复用
      const long long total = (long long)n_out * ly.out_c;
      const long long blocks = (total + threads - 1) / threads;
      sparse_subm_kernel<<<(int)blocks, threads, 0, stream>>>(
          ly, offsets_all, blob, coords, curf, hash_keys, hash_idx, hash_size,
          nxtf);
      float *tmp = curf; curf = nxtf; nxtf = tmp;
    } else {
      // strided：输出坐标可与输入重合，须用独立输出 hash 表。
      // pass1 唯一化（counter 分配）-> 读回 n_out -> pass2 累加 -> bn+relu
      // -> coords 更新为输出。
      cudaMemsetAsync(counter, 0, sizeof(int), stream);
      cudaMemsetAsync(out_hash_keys, 0xFF,
                      (size_t)hash_size * sizeof(long long), stream);
      const long long total = (long long)n_out * ly.K;
      const long long blocks = (total + threads - 1) / threads;
      sparse_strided_pass1_kernel<<<(int)blocks, threads, 0, stream>>>(
          ly, offsets_all, coords, n_out, out_hash_keys, out_hash_idx,
          hash_size, coords_buf, counter);
      int n_out_new = 0;
      cudaMemcpy(&n_out_new, counter, sizeof(int), cudaMemcpyDeviceToHost);
      // 清零 raw（bn 前累加缓冲）
      cudaMemsetAsync(raw_buf, 0, (size_t)n_out_new * ly.out_c * sizeof(float),
                      stream);
      sparse_strided_pass2_kernel<<<(int)blocks, threads, 0, stream>>>(
          ly, offsets_all, blob, coords, n_out, curf, out_hash_keys,
          out_hash_idx, hash_size, raw_buf);
      const long long total2 = (long long)n_out_new * ly.out_c;
      const long long blocks2 = (total2 + threads - 1) / threads;
      sparse_bn_relu_kernel<<<(int)blocks2, threads, 0, stream>>>(
          ly, blob, n_out_new, raw_buf, nxtf);
      float *tmp = curf; curf = nxtf; nxtf = tmp;
      coords = coords_buf;
      n_out = n_out_new;
    }
  }

  // 最后一层 hash + 稠密化
  cudaMemsetAsync(hash_keys, 0xFF, (size_t)hash_size * sizeof(long long),
                  stream);
  const int nb2 = (n_out + threads - 1) / threads;
  sparse_build_hash_kernel<<<nb2, threads, 0, stream>>>(coords, n_out,
                                                        hash_keys, hash_idx,
                                                        hash_size);
  const int C_final = (n_layers > 0) ? layers[n_layers - 1].out_c : in_c0;
  const long long total3 = (long long)X * Y * Z;
  const long long blocks3 = (total3 + threads - 1) / threads;
  sparse_densify_kernel<<<(int)blocks3, threads, 0, stream>>>(
      C_final, X, Y, Z, coords, hash_keys, hash_idx, hash_size, curf,
      lidar_bev);
}

}  // namespace dsb
