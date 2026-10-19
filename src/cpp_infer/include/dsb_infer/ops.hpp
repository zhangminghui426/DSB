#pragma once
// Inference ops declarations (CPU reference implementations).

#include "dsb_infer/config.hpp"
#include "dsb_infer/tensor.hpp"

#include <vector>

namespace dsb {

// ---------- CPU ops (ops.cpp) --------------------------------------------
// Lidar-depth projection for all cameras -> (1, N, 1, H, W) sparse depth.
Tensor project_depth(const Config &cfg, const Tensor &points,
                     const std::vector<Mat4> &lidar2image,
                     const std::vector<Mat4> &img_aug,
                     const Mat4 &lidar_aug);

// Camera frustum -> lidar-frame world coords -> (N, D, fH, fW, 3).
Tensor compute_geometry(const Config &cfg, const std::vector<Mat4> &camera2lidar,
                        const std::vector<Mat4> &intrins,
                        const std::vector<Mat4> &img_aug,
                        const Mat4 &lidar_aug);

// LSS bev pool: x (N, D, fH, fW, C) folded, geom (N, D, fH, fW, 3) world
// -> (C, nx[0], nx[1]) after Z collapse (Dz = nx[2] = 1).
Tensor bev_pool(const Config &cfg, const Tensor &geom, const Tensor &x);

// vtransform downsample: 3x (conv3x3+bn+relu, mid stride 2) -> (out_c, 180, 180).
Tensor downsample_conv(const Config &cfg, const Tensor &bev,
                       const std::vector<float> &blob);

// hard voxelize -> feats (M,5) / coords (M,4)=(b,x,y,z) / sizes (M).
void voxelize(const Tensor &points, const Config &cfg, Tensor &feats,
              Tensor &coords, Tensor &sizes);

// ---------- sparse encoder (sparse_conv.cpp) ------------------------------
Tensor sparse_encoder_forward(const Config &cfg, const Tensor &feats,
                              const Tensor &coords, const Tensor &sizes,
                              const std::vector<float> &blob);

// ---------- decode (decode.cpp) -------------------------------------------
struct DetOutput {
  Tensor boxes;   // (M, 9): x y z dx dy dz rot vx vy
  Tensor scores;  // (M,)
  Tensor labels;  // (M,) int
};
// preds order: dense_heatmap, heatmap, query_heatmap_score, center, height,
// dim, rot, vel (matches DET_HEAD_KEYS in deploy/graphs.py).
DetOutput decode_det(const Config &cfg, const std::vector<Tensor> &preds);

}  // namespace dsb
