#pragma once
// 后处理层：head 预测 -> 检测框（NMS + bbox coder + 过滤）。
// 与 Python 侧 deploy/postprocess.py（Postprocessor）逐层对应。

#include "config.hpp"
#include "tensor.hpp"

#include <vector>

namespace dsb {

struct DetOutput {
  Tensor boxes;   // (M, 9): x y z dx dy dz rot vx vy
  Tensor scores;  // (M,)
  Tensor labels;  // (M,) int
};

class Postprocessor {
 public:
  explicit Postprocessor(const Config &cfg);

  // preds 顺序：dense_heatmap, heatmap, query_heatmap_score, center,
  // height, dim, rot, vel, query_labels（与 deploy/graphs.py DET_HEAD_KEYS 一致）。
  DetOutput decode(const std::vector<Tensor> &preds) const;

 private:
  const Config &cfg_;
};

}  // namespace dsb
