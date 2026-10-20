// CPU 参考算子实现：矩阵工具、几何、bev_pool、降采样、稀疏编码。
// 供推理层（inference.cpp）内部调用；与 torch/spconv 参考逐位对齐。

#include "dsb_infer/ops/ops.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

namespace dsb {

// ================================================================ Mat helpers
Mat3 Mat3::identity() {
  Mat3 m{};
  m.m[0] = m.m[4] = m.m[8] = 1.0f;
  return m;
}

Mat3 Mat3::inverse(const Mat3 &a) {
  // 一般 3x3 逆（余子式 / 行列式）
  float d = a.m[0] * (a.m[4] * a.m[8] - a.m[5] * a.m[7]) -
            a.m[1] * (a.m[3] * a.m[8] - a.m[5] * a.m[6]) +
            a.m[2] * (a.m[3] * a.m[7] - a.m[4] * a.m[6]);
  if (std::abs(d) < 1e-12f) throw std::runtime_error("singular 3x3 inverse");
  float inv = 1.0f / d;
  Mat3 r{};
  r.m[0] = (a.m[4] * a.m[8] - a.m[5] * a.m[7]) * inv;
  r.m[1] = (a.m[2] * a.m[7] - a.m[1] * a.m[8]) * inv;
  r.m[2] = (a.m[1] * a.m[5] - a.m[2] * a.m[4]) * inv;
  r.m[3] = (a.m[5] * a.m[6] - a.m[3] * a.m[8]) * inv;
  r.m[4] = (a.m[0] * a.m[8] - a.m[2] * a.m[6]) * inv;
  r.m[5] = (a.m[2] * a.m[3] - a.m[0] * a.m[5]) * inv;
  r.m[6] = (a.m[3] * a.m[7] - a.m[4] * a.m[6]) * inv;
  r.m[7] = (a.m[1] * a.m[6] - a.m[0] * a.m[7]) * inv;
  r.m[8] = (a.m[0] * a.m[4] - a.m[1] * a.m[3]) * inv;
  return r;
}

Mat3 Mat3::mul(const Mat3 &a, const Mat3 &b) {
  Mat3 r{};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      float s = 0;
      for (int k = 0; k < 3; ++k) s += a.m[i * 3 + k] * b.m[k * 3 + j];
      r.m[i * 3 + j] = s;
    }
  return r;
}

void Mat3::matvec(const float v[3], float out[3]) const {
  for (int i = 0; i < 3; ++i)
    out[i] = m[i * 3] * v[0] + m[i * 3 + 1] * v[1] + m[i * 3 + 2] * v[2];
}

Mat3 Mat4::rot3() const {
  Mat3 r{};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) r.m[i * 3 + j] = m[i * 4 + j];
  return r;
}

void Mat4::col3(int c, float out[3]) const {
  for (int i = 0; i < 3; ++i) out[i] = m[i * 4 + c];
}

// ================================================================ geometry
// 镜像 BaseTransform.get_geometry + frustum（batch=1）。
Tensor compute_geometry(const Config &cfg, const std::vector<Mat4> &camera2lidar,
                        const std::vector<Mat4> &intrins,
                        const std::vector<Mat4> &img_aug,
                        const Mat4 &lidar_aug) {
  const int D = cfg.depth_bins(), fH = cfg.feat_h, fW = cfg.feat_w;
  // frustum: xs = linspace(0, iW-1, fW)，ys = linspace(0, iH-1, fH)，
  //          ds = arange(dbound)（D 个值）
  std::vector<float> xs(fW), ys(fH), ds(D);
  for (int i = 0; i < fW; ++i)
    xs[i] = (cfg.image_w - 1) * (float)i / (fW - 1);
  for (int i = 0; i < fH; ++i)
    ys[i] = (cfg.image_h - 1) * (float)i / (fH - 1);
  for (int i = 0; i < D; ++i)
    ds[i] = cfg.dbound[0] + cfg.dbound[2] * i;

  const Mat3 la_rot = lidar_aug.rot3();
  float la_t[3];
  lidar_aug.col3(3, la_t);

  Tensor geom({cfg.N, D, fH, fW, 3});
  for (int c = 0; c < cfg.N; ++c) {
    const Mat3 post_inv = Mat3::inverse(img_aug[c].rot3());
    float post_t[3];
    img_aug[c].col3(3, post_t);
    const Mat3 intr_inv = Mat3::inverse(intrins[c].rot3());
    const Mat3 c2l = camera2lidar[c].rot3();
    float c2l_t[3];
    camera2lidar[c].col3(3, c2l_t);
    const Mat3 combine = Mat3::mul(c2l, intr_inv);

    for (int di = 0; di < D; ++di)
      for (int hi = 0; hi < fH; ++hi)
        for (int wi = 0; wi < fW; ++wi) {
          // 逆 post
          const float v[3] = {xs[wi] - post_t[0], ys[hi] - post_t[1],
                              ds[di] - post_t[2]};
          float u[3];
          post_inv.matvec(v, u);
          // 按深度缩放
          const float cam[3] = {u[0] * u[2], u[1] * u[2], u[2]};
          float w3[3];
          combine.matvec(cam, w3);
          w3[0] += c2l_t[0];
          w3[1] += c2l_t[1];
          w3[2] += c2l_t[2];
          // 额外 lidar 增强
          float e[3];
          la_rot.matvec(w3, e);
          float *g = geom.data.data() + geom.offset({c, di, hi, wi, 0});
          g[0] = e[0] + la_t[0];
          g[1] = e[1] + la_t[1];
          g[2] = e[2] + la_t[2];
        }
  }
  return geom;
}

// ================================================================ bev_pool
// 镜像 BaseTransform.bev_pool（CUDA kernel 等价）：网格量化、范围过滤、
// 按 rank 分段求和（与 kernel 相同的精确 float 求和）。
Tensor bev_pool(const Config &cfg, const Tensor &geom, const Tensor &x) {
  // geom: (N, D, fH, fW, 3)；x: (N, D, fH, fW, C)
  const int D = (int)geom.shape[1], fH = (int)geom.shape[2],
            fW = (int)geom.shape[3];
  const int C = (int)x.shape[4];
  const int nx0 = cfg.nx[0], nx1 = cfg.nx[1], nx2 = cfg.nx[2];
  const float bx = cfg.xbound[0], by = cfg.ybound[0], bz = cfg.zbound[0];
  const float dx = cfg.xbound[2], dy = cfg.ybound[2], dz = cfg.zbound[2];

  // 网格坐标 + rank，保留范围内点
  std::vector<int64_t> keep_rank;
  std::vector<int64_t> keep_feat;  // x.reshape(-1, C) 的行号
  keep_rank.reserve(geom.shape[0] * D * fH * fW / 2);
  for (int64_t i = 0; i < geom.shape[0] * D * fH * fW; ++i) {
    const float *g = geom.data.data() + i * 3;
    const int cx = (int)std::floor((g[0] - (bx - dx / 2)) / dx);
    const int cy = (int)std::floor((g[1] - (by - dy / 2)) / dy);
    const int cz = (int)std::floor((g[2] - (bz - dz / 2)) / dz);
    if (cx >= 0 && cx < nx0 && cy >= 0 && cy < nx1 && cz >= 0 && cz < nx2) {
      // ranks_bev = b*(D*H*W) + z*(H*W) + x*W + y  (b=0, D=nx2, H=nx0, W=nx1)
      const int64_t rank = (int64_t)cz * (nx0 * nx1) + (int64_t)cx * nx1 + cy;
      keep_rank.push_back(rank);
      keep_feat.push_back(i);
    }
  }
  const int nk = (int)keep_rank.size();

  // 按 rank 排序（值稳定；求和与顺序无关）
  std::vector<int> order(nk);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(),
            [&](int a, int b) { return keep_rank[a] < keep_rank[b]; });

  // 分段求和 -> out (C, nx0, nx1) [Dz = nx2 = 1 折叠]
  Tensor out({C, nx0, nx1});
  int64_t i = 0;
  while (i < nk) {
    const int64_t r = keep_rank[order[i]];
    int64_t j = i;
    std::vector<double> acc(C, 0.0);
    for (; j < nk && keep_rank[order[j]] == r; ++j) {
      const float *xf = x.data.data() + keep_feat[order[j]] * C;
      for (int c = 0; c < C; ++c) acc[c] += xf[c];
    }
    const int64_t cx = r / nx1, cy = r % nx1;
    for (int c = 0; c < C; ++c) out.at({c, cx, cy}) = (float)acc[c];
    i = j;
  }
  return out;  // (C, 360, 360)
}

// ================================================================ downsample
// 3x (conv3x3 + bn + relu)；中间 conv stride 2；blob 布局按 DownLayer。
Tensor downsample_conv(const Config &cfg, const Tensor &bev,
                       const std::vector<float> &blob) {
  Tensor cur = bev;  // (C, H, W)
  for (const auto &ly : cfg.down_layers) {
    const int C_in = ly.in_c, C_out = ly.out_c, k = ly.k, s = ly.stride,
              p = ly.pad;
    const float *W = blob.data() + ly.w_off;        // (C_out, C_in, k, k)
    const float *bias = blob.data() + ly.w_off + ly.w_len;
    const float *g = blob.data() + ly.w_off + ly.w_len + ly.b_len;
    const float *bta = blob.data() + ly.w_off + ly.w_len + 2 * ly.b_len;
    const float *mean = blob.data() + ly.w_off + ly.w_len + 3 * ly.b_len;
    const float *var = blob.data() + ly.w_off + ly.w_len + 4 * ly.b_len;
    const int H = (int)cur.shape[1], Wd = (int)cur.shape[2];
    const int Ho = (H - k + 2 * p) / s + 1, Wo = (Wd - k + 2 * p) / s + 1;
    Tensor nxt({C_out, Ho, Wo});
    for (int co = 0; co < C_out; ++co)
      for (int ho = 0; ho < Ho; ++ho)
        for (int wo = 0; wo < Wo; ++wo) {
          float acc = bias[co];
          for (int ci = 0; ci < C_in; ++ci)
            for (int ki = 0; ki < k; ++ki)
              for (int kj = 0; kj < k; ++kj) {
                const int hi = ho * s - p + ki, wi = wo * s - p + kj;
                if (hi < 0 || hi >= H || wi < 0 || wi >= Wd) continue;
                acc += W[((co * C_in + ci) * k + ki) * k + kj] *
                       cur.at({ci, hi, wi});
              }
          const float s2 = var[co] + ly.bn_eps;
          const float v = (acc - mean[co]) / std::sqrt(s2) * g[co] + bta[co];
          nxt.at({co, ho, wo}) = std::max(v, 0.0f);  // ReLU
        }
    cur = std::move(nxt);
  }
  return cur;
}

// ================================================================ sparse
namespace {

// 坐标 key：按已知网格范围打包 (x, y, z)。
int64_t key3(int x, int y, int z) {
  return ((int64_t)x << 40) | ((int64_t)y << 20) | (int64_t)z;
}

}  // namespace

// 稀疏编码器前向（CPU 参考）。镜像 models/backbones/sparse_encoder.py：
//   - coords 为 (batch, x, y, z)；kernel/stride/pad 按 (x, y, z) 顺序
//   - SubM conv（SparseBasicBlock）：输出体素 == 输入体素，居中 kernel offset
//   - strided SparseConv：out = (in + pad - offset)/stride（可整除维）
//   - conv 权重 (Cout, K, Cin)，K 按 z->y->x 枚举（offsets 来自 meta）
//   - 最终稠密化：sparse_shape (X, Y, Z) -> (C, X, Y, Z) -> 折叠 Z
Tensor sparse_encoder_forward(const Config &cfg, const Tensor &feats,
                              const Tensor &coords, const Tensor &sizes,
                              const std::vector<float> &blob) {
  // ---- mean reduce：feats (M, 5) 按体素序分组 -> (M', 5)
  const int nvox = (int)coords.shape[0];
  const int feats_dim = (int)feats.shape[1];
  Tensor cur({nvox, feats_dim});
  {
    int r = 0;
    for (int v = 0; v < nvox; ++v) {
      const int cnt = (int)std::lround(sizes.data[v]);
      for (int d = 0; d < feats_dim; ++d) {
        double s = 0;
        for (int i = 0; i < cnt; ++i) s += feats.data[(r + i) * feats_dim + d];
        cur.data[v * feats_dim + d] = (float)(s / std::max(cnt, 1));
      }
      r += cnt;
    }
  }

  // coords (M', 4) -> (x, y, z) int 行
  std::vector<std::vector<int>> in_coords(nvox);
  for (int v = 0; v < nvox; ++v) {
    in_coords[v] = {(int)std::lround(coords.data[v * 4]),
                    (int)std::lround(coords.data[v * 4 + 1]),
                    (int)std::lround(coords.data[v * 4 + 2])};
  }
  std::vector<int64_t> in_keys(nvox);
  for (int v = 0; v < nvox; ++v)
    in_keys[v] = key3(in_coords[v][0], in_coords[v][1], in_coords[v][2]);

  for (const auto &ly : cfg.sparse_layers) {
    const int Cin = ly.in_c, Cout = ly.out_c;
    const int K = (int)ly.offsets.size();
    std::unordered_map<int64_t, int> in_map;
    for (int v = 0; v < (int)in_keys.size(); ++v) in_map[in_keys[v]] = v;

    if (ly.type == "subm") {
      // 输出体素 == 输入体素
      Tensor nxt({(int)in_coords.size(), Cout});
      for (int v = 0; v < (int)in_coords.size(); ++v) {
        for (int co = 0; co < Cout; ++co) {
          float acc = blob[ly.w_off + ly.w_len + co];  // bias
          for (int k = 0; k < K; ++k) {
            const int nk[3] = {in_coords[v][0] + ly.offsets[k][0],
                               in_coords[v][1] + ly.offsets[k][1],
                               in_coords[v][2] + ly.offsets[k][2]};
            auto it = in_map.find(key3(nk[0], nk[1], nk[2]));
            if (it == in_map.end()) continue;
            const float *xf = cur.data.data() + (size_t)it->second * Cin;
            const float *wf =
                blob.data() + ly.w_off + ((size_t)co * K + k) * Cin;
            for (int ci = 0; ci < Cin; ++ci) acc += wf[ci] * xf[ci];
          }
          // bn + relu
          const float mean =
              blob[ly.w_off + ly.w_len + 3 * (size_t)ly.b_len + co];
          const float var =
              blob[ly.w_off + ly.w_len + 4 * (size_t)ly.b_len + co];
          const float g = blob[ly.w_off + ly.w_len + (size_t)ly.b_len + co];
          const float b =
              blob[ly.w_off + ly.w_len + 2 * (size_t)ly.b_len + co];
          const float v = (acc - mean) / std::sqrt(var + ly.bn_eps) * g + b;
          nxt.data[v * Cout + co] = std::max(v, 0.0f);
        }
      }
      cur = std::move(nxt);
    } else {
      // strided：收集输出体素 + 贡献
      std::unordered_map<int64_t, int> out_map;
      std::vector<std::vector<int>> out_coords;
      std::vector<std::vector<std::pair<int, int>>> contribs;
      for (int u = 0; u < (int)in_coords.size(); ++u) {
        for (int k = 0; k < K; ++k) {
          int oc[3];
          bool ok = true;
          for (int d = 0; d < 3; ++d) {
            const int num = in_coords[u][d] + ly.pad[d] - ly.offsets[k][d];
            if (num % ly.stride[d] != 0) { ok = false; break; }
            oc[d] = num / ly.stride[d];
          }
          if (!ok) continue;
          const int64_t key = key3(oc[0], oc[1], oc[2]);
          auto it = out_map.find(key);
          int oi;
          if (it == out_map.end()) {
            oi = (int)out_coords.size();
            out_map[key] = oi;
            out_coords.push_back({oc[0], oc[1], oc[2]});
            contribs.emplace_back();
          } else {
            oi = it->second;
          }
          contribs[oi].push_back({u, k});
        }
      }
      Tensor nxt({(int)out_coords.size(), Cout});
      for (int o = 0; o < (int)out_coords.size(); ++o) {
        for (int co = 0; co < Cout; ++co) {
          float acc = blob[ly.w_off + ly.w_len + co];
          for (auto &ck : contribs[o]) {
            const float *xf = cur.data.data() + (size_t)ck.first * Cin;
            const float *wf = blob.data() + ly.w_off +
                              ((size_t)co * K + ck.second) * Cin;
            for (int ci = 0; ci < Cin; ++ci) acc += wf[ci] * xf[ci];
          }
          const float mean =
              blob[ly.w_off + ly.w_len + 3 * (size_t)ly.b_len + co];
          const float var =
              blob[ly.w_off + ly.w_len + 4 * (size_t)ly.b_len + co];
          const float g = blob[ly.w_off + ly.w_len + (size_t)ly.b_len + co];
          const float b =
              blob[ly.w_off + ly.w_len + 2 * (size_t)ly.b_len + co];
          const float v = (acc - mean) / std::sqrt(var + ly.bn_eps) * g + b;
          nxt.data[o * Cout + co] = std::max(v, 0.0f);
        }
      }
      cur = std::move(nxt);
      in_keys.clear();
      for (auto &c : out_coords) in_keys.push_back(key3(c[0], c[1], c[2]));
      in_coords = std::move(out_coords);
    }
  }

  // ---- 稠密化：cur (M', C) + in_coords (M') -> (C, X, Y, Z) -> 折叠 Z
  const int X = cfg.sparse_out[0], Y = cfg.sparse_out[1], Z = cfg.sparse_out[2];
  const int C = (int)cur.shape[1];
  std::unordered_map<int64_t, int> final_map;
  for (int v = 0; v < (int)in_coords.size(); ++v)
    final_map[key3(in_coords[v][0], in_coords[v][1], in_coords[v][2])] = v;

  // out (C*Z, X, Y)：out[(cz*X + x)*Y + y] = feat[hash(x,y,z)][c]
  Tensor out({C * Z, X, Y});
  for (int x = 0; x < X; ++x)
    for (int y = 0; y < Y; ++y)
      for (int z = 0; z < Z; ++z) {
        auto it = final_map.find(key3(x, y, z));
        if (it == final_map.end()) continue;
        const int v = it->second;
        for (int c = 0; c < C; ++c)
          out.data[(c * Z + z) * (X * Y) + x * Y + y] = cur.data[v * C + c];
      }
  return out;
}

}  // namespace dsb
