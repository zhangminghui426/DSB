#pragma once
// 模型推理层：TRT engine 编排 + CPU 自定义算子，输入 -> 预测张量。
// 与 Python 侧 deploy/inference.py（Inferencer）逐层对应。

#include "dsb_infer/config.hpp"
#include "dsb_infer/preprocess.hpp"
#include "dsb_infer/trt_engine.hpp"

#include <memory>
#include <string>
#include <vector>

namespace dsb {

class Inferencer {
 public:
  explicit Inferencer(const Config &cfg);

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
};

}  // namespace dsb
