#pragma once
// 模型推理层：TRT engine 编排 + CUDA/CPU 自定义算子，输入 -> 预测张量。
// 与 Python 侧 deploy/inference.py（Inferencer）逐层对应。
// 自定义算子（geometry / fold+bev_pool / downsample / voxelize / sparse）
// 参考 NVIDIA CUDA-BEVFusion 以 CUDA kernel 实现，与 TRT 共用 stream。

#include "config.hpp"
#include "preprocess/preprocess.hpp"
#include "inference/trt_engine.hpp"
#include "ops/cuda_kernels.hpp"

#include <cuda_runtime_api.h>

#include <memory>
#include <string>
#include <vector>

namespace dsb {

class Inferencer {
 public:
  explicit Inferencer(const Config &cfg);
  ~Inferencer();

  void load_engines(const std::string &engine_dir);
  void load_weights(const std::string &input_dir);

  // ---- 子图 / 算子执行 -------------------------------------------------
  // img (1,6,3,H,W) -> feats (1,6,256,32,88)。
  Tensor camera_backbone(const Tensor &img) const;

  // feats + depth (1,6,1,H,W) -> depthnet (1,6,D+C,32,88)。
  Tensor camera_vtransform(const Tensor &feats, const Tensor &depth) const;

  // depthnet -> cam_bev (1,80,180,180)：softmax fold + 几何 + bev_pool + 降采样。
  Tensor camera_bev(const Tensor &depthnet, const Preprocessor::Inputs &in) const;

  // 体素特征 -> lidar_bev (1,256,180,180)：mean reduce + 稀疏编码。
  Tensor lidar_branch(const Preprocessor::Inputs &in) const;

  // cam_bev + lidar_bev -> decoder_feat (1,512,180,180)。
  Tensor fuser(const Tensor &cam_bev, const Tensor &lidar_bev) const;

  // decoder_feat -> 9 个 head 预测（DET_HEAD_KEYS 顺序）。
  std::vector<Tensor> head(const Tensor &decoder_feat) const;

 private:
  const Config &cfg_;
  std::unique_ptr<TrtEngine> backbone_, vtransform_, fuser_, head_;
  std::vector<float> downsample_blob_, sparse_blob_;

  // ---- CUDA 自定义算子资源（与 TRT 引擎共用 stream） --------------------
  cudaStream_t stream_ = nullptr;
  float *geom_buf_ = nullptr;       // N*D*fH*fW*3
  float *depthnet_buf_ = nullptr;   // N*(D+C)*fH*fW
  float *bev_buf_ = nullptr;        // C*nx0*nx1（bev_pool 输出）
  float *ds_a_ = nullptr, *ds_b_ = nullptr, *ds_out_ = nullptr;
  float *blob_ds_ = nullptr;        // downsample 权重（设备端）
  CudaDownsampleLayer *layers_dev_ = nullptr;
  // lidar 稀疏链
  float *blob_sparse_ = nullptr;    // sparse 权重（设备端）
  CudaSparseLayer *sparse_layers_dev_ = nullptr;
  int *sparse_offsets_dev_ = nullptr;   // 所有层 offsets 打包（K*3 int/层）
  float *sparse_cur_ = nullptr, *sparse_nxt_ = nullptr, *sparse_raw_ = nullptr;
  float *lidar_bev_buf_ = nullptr;  // (C_final*Z, X, Y)
  int *sparse_coords_ = nullptr;    // (max_voxels*3) 输出 coords
  int *sparse_counter_ = nullptr;
  long long *sparse_hash_keys_ = nullptr;
  int *sparse_hash_idx_ = nullptr;
  long long *sparse_out_hash_keys_ = nullptr;  // strided 输出 hash（独立表）
  int *sparse_out_hash_idx_ = nullptr;
  int hash_size_ = 0;
};

}  // namespace dsb
