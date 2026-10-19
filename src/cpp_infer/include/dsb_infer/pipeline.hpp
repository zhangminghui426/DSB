#pragma once
// End-to-end C++ inference pipeline (pipeline.cpp): init / load / infer /
// verify / dump, matching deploy/export.py + deploy/inferencer.py.

#include "dsb_infer/config.hpp"
#include "dsb_infer/ops.hpp"
#include "dsb_infer/trt_engine.hpp"

#include <memory>
#include <string>
#include <vector>

namespace dsb {

class InferencePipeline {
 public:
  // One val-sample worth of raw inputs (bin files from tools/export_cpp_input.py).
  struct Inputs {
    Tensor img;  // (B, N, 3, image_h, image_w), normalized
    Tensor points;  // (n, 5)
    std::vector<Mat4> camera2lidar, intrins, img_aug;  // 6 each
    Mat4 lidar_aug;
  };

  struct Outputs {
    Tensor cam_bev;      // (1, 80, 180, 180)
    Tensor lidar_bev;    // (1, 256, 180, 180)
    Tensor decoder_feat; // (1, 512, 180, 180)
    DetOutput det;       // decoded boxes / scores / labels
  };

  explicit InferencePipeline(const Config &cfg);

  Inputs load_inputs(const std::string &input_dir) const;
  void load_engines(const std::string &engine_dir);
  void load_weights(const std::string &input_dir);

  Outputs infer(const Inputs &in);

  // Prints decoder_feat vs torch ref (when <input_dir>/ref/decoder_feat.bin exists).
  void verify(const std::string &input_dir, const Outputs &out) const;
  void dump(const Outputs &out, const std::string &dump_dir) const;

 private:
  Config cfg_;
  std::unique_ptr<TrtEngine> backbone_, vtransform_, fuser_, head_;
  std::vector<float> downsample_blob_, sparse_blob_;
};

}  // namespace dsb
