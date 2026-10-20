// 后处理层实现：head 预测 -> 检测框（NMS + bbox coder + 过滤）。
#include "dsb_infer/postprocess.hpp"

#include <algorithm>
#include <cmath>

namespace dsb {

namespace {

struct Box {
  float score;
  int cls;
  std::array<float, 9> b;  // x y z dx dy dz rot vx vy
};

float iou2d(const std::array<float, 9> &a, const std::array<float, 9> &b) {
  // 轴对齐 2D BEV IoU（近似，旋转忽略）。
  const float ax1 = a[0] - a[3] / 2, ax2 = a[0] + a[3] / 2;
  const float ay1 = a[1] - a[4] / 2, ay2 = a[1] + a[4] / 2;
  const float bx1 = b[0] - b[3] / 2, bx2 = b[0] + b[3] / 2;
  const float by1 = b[1] - b[4] / 2, by2 = b[1] + b[4] / 2;
  const float iw = std::max(0.f, std::min(ax2, bx2) - std::max(ax1, bx1));
  const float ih = std::max(0.f, std::min(ay2, by2) - std::max(ay1, by1));
  const float inter = iw * ih;
  const float ua = a[3] * a[4] + b[3] * b[4] - inter;
  return ua > 0.f ? inter / ua : 0.f;
}

}  // namespace

Postprocessor::Postprocessor(const Config &cfg) : cfg_(cfg) {}

DetOutput Postprocessor::decode(const std::vector<Tensor> &preds) const {
  // preds：dense_heatmap, heatmap, query_heatmap_score, center, height,
  // dim, rot, vel, query_labels（float -> int）。
  const auto &qscore = preds[2];
  const auto &center = preds[3];
  const auto &height = preds[4];
  const auto &dim = preds[5];
  const auto &rot = preds[6];
  const auto &vel = preds[7];
  const float *ql = preds[8].data();
  const int M = qscore.dims()[0];

  std::vector<Box> cand;
  cand.reserve(M);
  for (int i = 0; i < M; ++i) {
    const float s = qscore.data()[i];
    if (s < cfg_.score_thr) continue;
    Box b;
    b.score = s;
    b.cls = static_cast<int>(std::round(ql[i]));
    // center / height / dim / rot / vel 按 (M, C) 行布局读。
    b.b[0] = center.data()[i * 2 + 0];
    b.b[1] = center.data()[i * 2 + 1];
    b.b[2] = height.data()[i];
    b.b[3] = dim.data()[i * 3 + 0];
    b.b[4] = dim.data()[i * 3 + 1];
    b.b[5] = dim.data()[i * 3 + 2];
    b.b[6] = rot.data()[i];
    b.b[7] = vel.data()[i * 2 + 0];
    b.b[8] = vel.data()[i * 2 + 1];
    cand.push_back(b);
  }

  // 类内 NMS（按 score 降序）。
  std::sort(cand.begin(), cand.end(),
            [](const Box &a, const Box &b) { return a.score > b.score; });
  std::vector<Box> keep;
  for (const auto &b : cand) {
    bool dup = false;
    for (const auto &k : keep) {
      if (k.cls != b.cls) continue;
      if (iou2d(k.b, b.b) > cfg_.nms_iou_threshold) { dup = true; break; }
    }
    if (!dup) {
      keep.push_back(b);
      if (static_cast<int>(keep.size()) >= cfg_.max_det) break;
    }
  }

  DetOutput out;
  const int K = static_cast<int>(keep.size());
  out.boxes = Tensor(std::vector<int>{K, 9}, Tensor::kFloat);
  out.scores = Tensor(std::vector<int>{K}, Tensor::kFloat);
  out.labels = Tensor(std::vector<int>{K}, Tensor::kInt32);
  for (int i = 0; i < K; ++i) {
    std::copy(keep[i].b.begin(), keep[i].b.end(), out.boxes.data() + i * 9);
    out.scores.data()[i] = keep[i].score;
    out.labels.data<int32_t>()[i] = keep[i].cls;
  }
  return out;
}

}  // namespace dsb
