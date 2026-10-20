#pragma once
// DSB 推理 CUDA 自定义算子（参考 NVIDIA-AI-IOT/CUDA-BEVFusion 的 kernel 组织）：
//   cuda_compute_geometry / cuda_fold_bevpool / cuda_downsample / cuda_sparse_encode
// 调用方负责设备内存与 stream；函数在传入 stream 上异步执行，调用方负责同步。
// 数值语义与 src/ops/ops.cpp 的 CPU 参考实现一致（fold 后按 rank 求和、无归一化）。

#include <cuda_runtime.h>

#include <cstdint>

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

struct CudaSparseLayer {
  int type;               // 0=subm, 1=strided
  int in_c, out_c, K;     // K = offsets 数
  long long w_off, w_len, b_len;  // conv.w 在 blob[w_off..w_off+w_len)，bias 在 +w_len
  float bn_eps;
  int sx, sy, sz;         // stride (x, y, z)
  int px, py, pz;         // pad
  long long off_off;      // 本层 offsets 起点（offsets_all 内，K*3 个 int：dx,dy,dz 连续）
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

// ---- lidar 稀疏编码（逐层 subm/strided 稀疏卷积 + 稠密化），全 device 执行 ----
// 输入：cur (nvox*in_c) 为体素均值特征（mean reduce 已在调用方做）；
//       coords_in (nvox*3) int 网格坐标；blob / offsets_all 设备端权重。
// 输出：lidar_bev (C_final*Z, X, Y)。
// 工作缓冲：cur_buf/nxt_buf/raw_buf (max_voxels*max_c)；
//          hash_keys/hash_idx 输入 hash，out_hash_keys/out_hash_idx 输出 hash
//          （strided 输出坐标可与输入重合，必须用独立表），均为 hash_size 大小；
//          coords_buf (max_voxels*3) int；counter 1 个 int。
void cuda_sparse_encode(
    const CudaSparseLayer *layers, int n_layers, const float *blob,
    const int *offsets_all, const float *cur, int nvox, int in_c0,
    const int *coords_in, float *lidar_bev, int X, int Y, int Z,
    float *cur_buf, float *nxt_buf, float *raw_buf, long long *hash_keys,
    int *hash_idx, long long *out_hash_keys, int *out_hash_idx,
    int hash_size, int *coords_buf, int *counter, cudaStream_t stream);

}  // namespace dsb
