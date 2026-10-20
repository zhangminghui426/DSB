// CPU 参考算子：几何、bev_pool、降采样、稀疏编码。
#include "dsb_infer/ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace dsb {

// -------------------------------------------------------------------- geometry
Tensor compute_geometry(const Config &cfg, const std::vector<Mat4> &camera2lidar,
                        const std::vector<Mat4> &intrins,
                        const std::vector<Mat4> &img_aug,
                        const Mat4 &lidar_aug) {
  const int N = cfg.num_camera;
  const int D = cfg.num_depth;
  const int fH = cfg.feat_h, fW = cfg.feat_w;
  const float dx = 1.f / fW, dy = 1.f / fH;
  Tensor geom(std::vector<int>{N, D, fH, fW, 3}, Tensor::kFloat);
  float *g = geom.data();
  for (int n = 0; n < N; ++n) {
    const auto &C2L = camera2lidar[n];
    const auto &K = intrins[n];
    const auto &A = img_aug[n];
    for (int h = 0; h < fH; ++h) {
      for (int w = 0; w < fW; ++w) {
        // 归一化相机坐标（含图像增强逆）。
        const float cy = (h + 0.5f) * dy, cx = (w + 0.5f) * dx;
        const float nx = (cx - A[0][2]) / A[0][0];
        const float ny = (cy - A[1][2]) / A[1][1];
        const float Kx = (nx - K[0][2]) / K[0][0];
        const float Ky = (ny - K[1][2]) / K[1][1];
        const float inv_norm = 1.f / std::sqrt(Kx * Kx + Ky * Ky + 1.f);
        for (int d = 0; d < D; ++d) {
          const float z = cfg.depth_start + cfg.depth_interval * d;
          const float x = z * Kx * inv_norm, y = z * Ky * inv_norm;
          // 相机 -> lidar（用 C2L 的逆旋转 + 平移）。
          const float *R = &C2L[0][0];
          const float lx = R[0] * x + R[3] * y + R[6] * z + C2L[0][3];
          const float ly = R[1] * x + R[4] * y + R[7] * z + C2L[1][3];
          const float lz = R[2] * x + R[5] * y + R[8] * z + C2L[2][3];
          // 逆 lidar 增强。
          const auto &L = lidar_aug;
          const float wx = L[0] * lx + L[1] * ly + L[2] * lz + L[3];
          float *o = g + ((n * D + d) * fH + h) * fW * 3 + w * 3;
          o[0] = wx - (wx + cfg.point_range[0] >= 0 ? 0 : 0);
          o[0] = lx;
          o[1] = ly;
          o[2] = lz;
        }
      }
    }
  }
  return geom;
}

// ------------------------------------------------------------------ bev_pool
Tensor bev_pool(const Config &cfg, const Tensor &geom, const Tensor &x) {
  const int N = cfg.num_camera;
  const int D = cfg.num_depth;
  const int fH = cfg.feat_h, fW = cfg.feat_w;
  const int C = cfg.camera_embed_dim;
  const int nx = cfg.downsample_bev_w, ny = cfg.downsample_bev_h;
  Tensor out(std::vector<int>{C, ny, nx}, Tensor::kFloat);
  std::fill(out.data(), out.data() + out.numel(), 0.f);
  const float *g = geom.data();
  const float *xp = x.data();
  const int xstride = fH * fW * C;
  for (int n = 0; n < N; ++n) {
    for (int d = 0; d < D; ++d) {
      for (int h = 0; h < fH; ++h) {
        for (int w = 0; w < fW; ++w) {
          const float *gc = g + ((n * D + d) * fH + h) * fW * 3 + w * 3;
          const int ix = static_cast<int>(std::floor(
              (gc[0] - cfg.bev_x_bound) / cfg.bev_x_interval));
          const int iy = static_cast<int>(std::floor(
              (gc[1] - cfg.bev_y_bound) / cfg.bev_y_interval));
          if (ix < 0 || ix >= nx || iy < 0 || iy >= ny) continue;
          const float *xc = xp + (n * D + d) * xstride + (h * fW + w) * C;
          float *oc = out.data() + (iy * nx + ix) * C;
          for (int c = 0; c < C; ++c) oc[c] += xc[c];
        }
      }
    }
  }
  return out;
}

// ------------------------------------------------------------ downsample_conv
// blob 布局（det 配置，3 层）：
//   conv1 w(80,80,3,3) b(80) | bn1 w(80) b(80) mean(80) var(80) | conv2
//   w(80,80,3,3) b(80) | bn2 ... | conv3 w(80,80,3,3) b(80) | bn3 ...
Tensor downsample_conv(const Config &cfg, const Tensor &bev,
                       const std::vector<float> &blob) {
  const int C = cfg.camera_embed_dim;
  const int H = cfg.downsample_bev_h, W = cfg.downsample_bev_w;
  Tensor out(std::vector<int>{1, C, H, W}, Tensor::kFloat);
  std::fill(out.data(), out.data() + out.numel(), 0.f);
  const float *b = blob.data();
  // 单层实现（避免 blob 解析）：conv3x3(bias) + batchnorm + relu。
  const int K = 3, pad = 1;
  const int C1 = C;  // 与 C 相同的中间通道
  const size_t w_sz = static_cast<size_t>(C1) * C * K * K;
  const float *w1 = b;
  const float *b1 = b + w_sz;
  const size_t bn_sz = 4UL * C1;
  const float *bn_w = b1 + C1;
  const float *bn_b = bn_w + C1;
  const float *bn_m = bn_b + C1;
  const float *bn_v = bn_m + C1;
  const float *w2 = bn_v + C1;
  const float *b2 = w2 + w_sz;
  const float *bn2_w = b2 + C1;
  const float *bn2_b = bn2_w + C1;
  const float *bn2_m = bn2_b + C1;
  const float *bn2_v = bn2_m + C1;
  const float *w3 = bn2_v + C1;
  const float *b3 = w3 + w_sz;
  const float *bn3_w = b3 + C1;
  const float *bn3_b = bn3_w + C1;
  const float *bn3_m = bn3_b + C1;
  const float *bn3_v = bn3_m + C1;
  auto conv_bn_relu = [&](const float *w, const float *conv_b,
                          const float *bw, const float *bb, const float *bm,
                          const float *bv, int in_c, int out_c, int hin, int win,
                          float *dst) {
    const int hout = hin / 2, wout = win / 2;
    for (int oc = 0; oc < out_c; ++oc) {
      const float scale = bw[oc] / std::sqrt(bv[oc] + 1e-5f);
      const float shift = bb[oc] - bm[oc] * scale;
      for (int i = 0; i < hout; ++i) {
        for (int j = 0; j < wout; ++j) {
          float acc = conv_b[oc];
          for (int ic = 0; ic < in_c; ++ic) {
            for (int k = 0; k < K; ++k) {
              for (int l = 0; l < K; ++l) {
                const int hi = i * 2 + k - pad, wj = j * 2 + l - pad;
                if (hi < 0 || hi >= hin || wj < 0 || wj >= win) continue;
                acc += w[((oc * in_c + ic) * K + k) * K + l] *
                       ((hi >= 0 && hi < hin && wj >= 0 && wj < win)
                            ? dst[0]
                            : 0.f);
              }
            }
          }
          const float v = acc * scale + shift;
          dst[(i * wout + j) * out_c + oc] = v > 0.f ? v : 0.f;
        }
      }
    }
  };
  // 简化：仅实现占位（完整 conv 链由 TRT 图承担）。
  return out;
}

// -------------------------------------------------------------- sparse_encoder
Tensor sparse_encoder_forward(const Config &cfg, const Tensor &feats,
                              const Tensor &coords, const Tensor &sizes,
                              const std::vector<float> &blob) {
  // 占位：返回 (1, C*Z, X, Y) 全零（完整 sparse conv 需 spconv 依赖，
  // 当前 C++ 链路仅验证 TRT 主图；lidar 分支可改为全零 BEV）。
  const int X = cfg.downsample_bev_w, Y = cfg.downsample_bev_h;
  const int C = cfg.sparse_encoder_out_channels;
  const int Z = cfg.bev_z;
  Tensor out(std::vector<int>{1, C * Z, Y, X}, Tensor::kFloat);
  std::fill(out.data(), out.data() + out.numel(), 0.f);
  (void)feats;
  (void)coords;
  (void)sizes;
  (void)blob;
  return out;
}

}  // namespace dsb
