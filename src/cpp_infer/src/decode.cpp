#include "dsb_infer/ops.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace dsb {

namespace {
// max pool 3x3 (stride 1, no padding) over (B, C, H, W).
void maxpool3(const Tensor &in, Tensor &out) {
  int B = (int)in.shape[0], C = (int)in.shape[1];
  int H = (int)in.shape[2], W = (int)in.shape[3];
  out.resize({B, C, H - 2, W - 2});
  for (int b = 0; b < B; ++b)
    for (int c = 0; c < C; ++c)
      for (int i = 0; i < H - 2; ++i)
        for (int j = 0; j < W - 2; ++j) {
          float m = -1e30f;
          for (int di = 0; di < 3; ++di)
            for (int dj = 0; dj < 3; ++dj)
              m = std::max(m, in.at({b, c, i + di, j + dj}));
          out.at({b, c, i, j}) = m;
        }
}
}  // namespace

// Mirrors TransFusionHead.get_bboxes (nms_type=None) + TransFusionBBoxCoder.decode.
DetOutput decode_det(const Config &cfg, const std::vector<Tensor> &preds) {
  // preds: dense_heatmap, heatmap, query_heatmap_score, center, height, dim,
  //        rot, vel, query_labels (float; class id per proposal, from the
  //        in-graph topk of head.bbox).
  const Tensor &dense_hm = preds[0];
  const Tensor &prop_hm = preds[1];
  const Tensor &qhm = preds[2];
  const Tensor &center = preds[3];
  const Tensor &height = preds[4];
  const Tensor &dim = preds[5];
  const Tensor &rot = preds[6];
  const Tensor &vel = preds[7];
  const Tensor &ql = preds[8];

  int B = (int)prop_hm.shape[0];
  int ncls = (int)prop_hm.shape[1];
  int nprop = (int)prop_hm.shape[2];

  // ---- NMS on dense heatmap (kernel 3, stride 1) -----------------------
  Tensor hm(dense_hm.shape);
  for (size_t i = 0; i < dense_hm.data.size(); ++i)
    hm.data[i] = 1.0f / (1.0f + std::exp(-dense_hm.data[i]));  // sigmoid
  Tensor pooled;
  maxpool3(hm, pooled);  // (B, C, H-2, W-2)
  int H = (int)hm.shape[2], W = (int)hm.shape[3];
  for (int b = 0; b < B; ++b)
    for (int c = 0; c < ncls; ++c)
      for (int i = 0; i < H; ++i)
        for (int j = 0; j < W; ++j) {
          // local_max = pooled at (i-1, j-1) inside [1, H-2]; 0 elsewhere
          bool inner = (i >= 1 && i <= H - 2 && j >= 1 && j <= W - 2);
          float lm = inner ? pooled.at({b, c, i - 1, j - 1}) : 0.0f;
          float v = hm.at({b, c, i, j});
          hm.at({b, c, i, j}) = (v == lm) ? v : 0.0f;
        }

  // ---- query_labels from the engine (same in-graph topk as
  // query_heatmap_score) ------------------------------------------------
  std::vector<std::vector<int>> query_labels(B, std::vector<int>(nprop));
  for (int b = 0; b < B; ++b)
    for (int j = 0; j < nprop; ++j)
      query_labels[b][j] = (int)std::lround(ql.at({b, j}));

  // ---- batch score = heatmap.sigmoid() * qhm * one_hot ------------------
  // (B, ncls, nprop)
  std::vector<std::vector<float>> score(B, std::vector<float>(ncls * nprop));
  for (int b = 0; b < B; ++b)
    for (int c = 0; c < ncls; ++c)
      for (int j = 0; j < nprop; ++j) {
        float s = 1.0f / (1.0f + std::exp(-prop_hm.at({b, c, j})));
        s *= qhm.at({b, c, j});
        if (query_labels[b][j] != c) s = 0.0f;
        score[b][c * nprop + j] = s;
      }

  // ---- decode to real-world boxes ---------------------------------------
  int code = 9;  // x y z dx dy dz rot vx vy
  float osf = (float)cfg.head_osf;
  float vx0 = cfg.head_voxel[0], vy0 = cfg.head_voxel[1];
  float px0 = cfg.head_pc[0], py0 = cfg.head_pc[1];
  std::vector<float> all_boxes(B * nprop * code);
  std::vector<float> all_scores(B * nprop);
  std::vector<int> all_labels(B * nprop);
  for (int b = 0; b < B; ++b) {
    for (int j = 0; j < nprop; ++j) {
      // best class
      int best_c = 0;
      float best_s = score[b][0 * nprop + j];
      for (int c = 1; c < ncls; ++c)
        if (score[b][c * nprop + j] > best_s) {
          best_s = score[b][c * nprop + j];
          best_c = c;
        }
      float cx = center.at({b, 0, j}) * osf * vx0 + px0;
      float cy = center.at({b, 1, j}) * osf * vy0 + py0;
      float d0 = std::exp(dim.at({b, 0, j}));
      float d1 = std::exp(dim.at({b, 1, j}));
      float d2 = std::exp(dim.at({b, 2, j}));
      float z = height.at({b, 0, j}) - d2 * 0.5f;
      float r = std::atan2(rot.at({b, 0, j}), rot.at({b, 1, j}));
      float *box = &all_boxes[(size_t)(b * nprop + j) * code];
      box[0] = cx; box[1] = cy; box[2] = z;
      box[3] = d0; box[4] = d1; box[5] = d2;
      box[6] = r;
      box[7] = vel.at({b, 0, j});
      box[8] = vel.at({b, 1, j});
      all_scores[b * nprop + j] = best_s;
      all_labels[b * nprop + j] = best_c;
    }
  }

  // ---- filter: post_center_range (+ score threshold if > 0) --------------
  const float *pc = cfg.head_post.data();  // [minx,miny,minz,maxx,maxy,maxz]
  DetOutput out;
  std::vector<float> obox, oscore;
  std::vector<float> olabel;
  for (int j = 0; j < nprop; ++j) {
    const float *box = &all_boxes[j * code];
    if (!(box[0] >= pc[0] && box[1] >= pc[1] && box[2] >= pc[2] &&
          box[0] <= pc[3] && box[1] <= pc[4] && box[2] <= pc[5]))
      continue;
    if (cfg.head_score_thr > 0.0f && all_scores[j] <= cfg.head_score_thr)
      continue;
    for (int d = 0; d < code; ++d) obox.push_back(box[d]);
    oscore.push_back(all_scores[j]);
    olabel.push_back((float)all_labels[j]);
  }
  out.boxes = Tensor({(int64_t)obox.size() / code, code});
  out.boxes.data = std::move(obox);
  out.scores = Tensor({(int64_t)oscore.size()});
  out.scores.data = std::move(oscore);
  out.labels = Tensor({(int64_t)olabel.size()});
  out.labels.data = std::move(olabel);
  return out;
}

}  // namespace dsb
