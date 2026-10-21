#pragma once
// lidar 稀疏编码 CUDA 算子（参考 NVIDIA-AI-IOT/CUDA-BEVFusion lidar-voxelization
// 组织方式）：逐层 subm/strided 稀疏卷积 + 最终稠密化，替代 CPU 版
// sparse_encoder_forward。调用方负责设备内存与 stream；异步执行，调用方同步。

#include <cuda_runtime.h>

#include <cstdint>

namespace dsb {

// 每层 conv 权重布局（blob，设备端连续）：
//   [w (out_c*K*in_c) | bias (out_c) | mean (out_c) | var (out_c)
//    | g (out_c) | b (out_c)]，自 w_off 起；offsets 在 offsets_all[off_off..]
struct CudaSparseLayer {
  int type;               // 0=subm, 1=strided
  int in_c, out_c, K;     // K = offsets 数
  long long w_off, w_len, b_len;  // conv.w 在 blob[w_off..w_off+w_len)，bias 在 +w_len
  float bn_eps;
  int sx, sy, sz;         // stride (x, y, z)
  int px, py, pz;         // pad
  long long off_off;      // 本层 offsets 起点（offsets_all 内，K*3 个 int：dx,dy,dz 连续）
};

// 逐层：hash 重建 -> subm/strided conv；最后稠密化输出 (C*Z, X, Y)。
// 输入：cur (nvox*in_c) 为体素均值特征（mean reduce 已在调用方做）；
//       coords_in (nvox*3) int 网格坐标；blob / offsets_all 设备端权重。
// 工作缓冲：nxt_buf/raw_buf (max_voxels*max_c)；
//          hash_keys/hash_idx 输入 hash，out_hash_keys/out_hash_idx 输出 hash
//          （strided 输出坐标可与输入重合，必须用独立表），均为 hash_size 大小；
//          coords_buf (max_voxels*3) int；counter 1 个 int。
void cuda_sparse_encode(
    const CudaSparseLayer *layers, int n_layers, const float *blob,
    const int *offsets_all, const float *cur, int nvox, int in_c0,
    const int *coords_in, float *lidar_bev, int X, int Y, int Z,
    float *nxt_buf, float *raw_buf, long long *hash_keys, int *hash_idx,
    long long *out_hash_keys, int *out_hash_idx, int hash_size,
    int *coords_buf, int *counter, cudaStream_t stream);

}  // namespace dsb
