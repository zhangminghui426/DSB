// lidar 稀疏编码 CUDA 算子实现（参考 CUDA-BEVFusion lidar-voxelization 组织）。
// 与 CPU 参考（ops.cpp sparse_encoder_forward）同语义：
//   subm：输出体素 == 输入体素，查 hash 找邻居；
//   strided：out = (in + pad - offset)/stride（可整除维），输出需唯一化；
//   稠密化：(C*Z, X, Y) 布局 out[(c*Z+z)*XY + x*Y + y] = feat[voxel(x,y,z)][c]。

#include "ops/lidar-sparse.hpp"

#include <math.h>

namespace dsb {

namespace {

// 坐标 key：+4096 偏移保证非负。subm 查询越界邻居坐标会为负，
// 若不偏移则负 key 与 hash 空哨兵（-1）冲突，lookup 会误判命中。
__device__ long long key3(int x, int y, int z) {
  return (((long long)(x + 4096)) << 40) |
         (((long long)(y + 4096)) << 20) | ((long long)(z + 4096));
}

// hash：pow2 大小，线性探测；keys 哨兵 -1（0xFFFFFFFFFFFFFFFF）。
// atomicCAS 只有 unsigned long long 重载，故以无符号形式读写。
__device__ int hash_slot(long long key, int hash_size) {
  return (int)(((unsigned long long)key * 2654435761ULL) & (hash_size - 1));
}

__device__ void hash_insert(long long *keys, int *idx, int hash_size,
                            long long key, int v) {
  const unsigned long long ukey = (unsigned long long)key;
  int h = hash_slot(key, hash_size);
  while (true) {
    unsigned long long old = atomicCAS((unsigned long long *)&keys[h],
                                       0xFFFFFFFFFFFFFFFFULL, ukey);
    if (old == 0xFFFFFFFFFFFFFFFFULL) {  // 插入成功
      idx[h] = v;
      return;
    }
    if (old == ukey) {  // 已存在（并发重复）
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
  const unsigned long long ukey = (unsigned long long)key3(oc[0], oc[1], oc[2]);
  int hh = hash_slot(key3(oc[0], oc[1], oc[2]), hash_size);
  while (true) {
    unsigned long long old = atomicCAS((unsigned long long *)&keys[hh],
                                       0xFFFFFFFFFFFFFFFFULL, ukey);
    if (old == 0xFFFFFFFFFFFFFFFFULL) {  // 新输出体素：分配序号并记录坐标
      const int oi = atomicAdd(counter, 1);
      idx[hh] = oi;
      coords_out[oi * 3] = oc[0];
      coords_out[oi * 3 + 1] = oc[1];
      coords_out[oi * 3 + 2] = oc[2];
      return;
    }
    if (old == ukey) return;  // 已存在（由其他 (u,k) 建立）
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
    float *nxt_buf, float *raw_buf, long long *hash_keys, int *hash_idx,
    long long *out_hash_keys, int *out_hash_idx, int hash_size,
    int *coords_buf, int *counter, cudaStream_t stream) {
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
          ly, offsets_all, blob, coords, curf, n_out, hash_keys, hash_idx,
          hash_size, nxtf);
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
