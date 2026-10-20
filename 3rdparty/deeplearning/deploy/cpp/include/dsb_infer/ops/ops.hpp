#pragma once
// CPU 参考算子（推理层内部使用）：几何、bev_pool、降采样、稀疏编码。

#include "dsb_infer/config.hpp"
#include "dsb_infer/tensor.hpp"

#include <vector>

namespace dsb {

// Camera frustum -> lidar-frame world coords -> (N, D, fH, fW, 3)。
Tensor compute_geometry(const Config &cfg, const std::vector<Mat4> &camera2lidar,
                        const std::vector<Mat4> &intrins,
                        const std::vector<Mat4> &img_aug,
                        const Mat4 &lidar_aug);

// LSS bev pool：x (N, D, fH, fW, C)，geom (N, D, fH, fW, 3) 世界坐标
// -> (C, nx[0], nx[1])（Z 折叠，Dz = nx[2] = 1）。
Tensor bev_pool(const Config &cfg, const Tensor &geom, const Tensor &x);

// vtransform 降采样：3x (conv3x3+bn+relu, 中间 stride 2) -> (out_c, 180, 180)。
Tensor downsample_conv(const Config &cfg, const Tensor &bev,
                       const std::vector<float> &blob);

// 稀疏编码器前向：feats (M,5) / coords (M,4) / sizes (M,) -> (C*Z, X, Y)。
Tensor sparse_encoder_forward(const Config &cfg, const Tensor &feats,
                              const Tensor &coords, const Tensor &sizes,
                              const std::vector<float> &blob);

}  // namespace dsb
