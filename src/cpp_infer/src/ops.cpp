#include "dsb_infer/ops.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>
#include <unordered_map>

namespace dsb {

// ================================================================ Mat helpers
Mat3 Mat3::identity() {
  Mat3 m{};
  m.m[0] = m.m[4] = m.m[8] = 1.0f;
  return m;
}

Mat3 Mat3::inverse(const Mat3 &a) {
  // general 3x3 inverse (cofactor / det)
  float d = a.m[0] * (a.m[4] * a.m[5] - a.m[5] * a.m[7]) -
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

// ================================================================ load_config
static std::vector<std::string> split(const std::string &s) {
  std::istringstream iss(s);
  std::vector<std::string> v;
  std::string tok;
  while (iss >> tok) v.push_back(tok);
  return v;
}

static std::vector<float> parse_floats(const std::string &s) {
  auto toks = split(s);
  std::vector<float> v;
  for (auto &t : toks) v.push_back(std::stof(t));
  return v;
}
static std::vector<int> parse_ints(const std::string &s) {
  auto toks = split(s);
  std::vector<int> v;
  for (auto &t : toks) v.push_back(std::stoi(t));
  return v;
}

Config load_config(const std::string &path) {
  Config cfg;
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open meta: " + path);
  std::string line;
  std::vector<std::string> raw_lines;
  while (std::getline(f, line)) raw_lines.push_back(line);

  size_t i = 0;
  auto next = [&]() -> std::string {
    for (; i < raw_lines.size(); ++i) {
      std::string l = raw_lines[i];
      size_t c = l.find('#');
      if (c != std::string::npos) l = l.substr(0, c);
      if (!l.empty()) return l;
    }
    return "";
  };

  std::vector<std::string> table_lines;
  for (; i < raw_lines.size(); ++i) {
    std::string l = raw_lines[i];
    size_t c = l.find('#');
    if (c != std::string::npos) l = l.substr(0, c);
    if (l.empty()) continue;
    auto toks = split(l);
    if (toks[0] == "sparse_layer_N" || toks[0] == "downsample_N") {
      int n = std::stoi(toks[1]);
      if (toks[0] == "sparse_layer_N") cfg.sparse_layers.clear();
      else cfg.down_layers.clear();
      for (int k = 0; k < n; ++k) {
        std::string l2 = next();
        auto t2 = split(l2);
        if (t2.empty()) throw std::runtime_error("bad table line");
        if (t2[0] == "subm" || t2[0] == "strided") {
          SparseLayer ly;
          ly.type = t2[0];
          ly.in_c = std::stoi(t2[1]);
          ly.out_c = std::stoi(t2[2]);
          ly.kernel = {std::stoi(t2[3]), std::stoi(t2[4]), std::stoi(t2[5])};
          ly.stride = {std::stoi(t2[6]), std::stoi(t2[7]), std::stoi(t2[8])};
          ly.pad = {std::stoi(t2[9]), std::stoi(t2[10]), std::stoi(t2[11])};
          int noff = std::stoi(t2[12]);
          ly.w_off = std::stoull(t2[13]);
          ly.w_len = std::stoull(t2[14]);
          ly.b_len = std::stoi(t2[15]);
          ly.bn_eps = std::stof(t2[16]);
          for (int o = 0; o < noff; ++o) {
            std::string l3 = next();
            auto t3 = split(l3);
            ly.offsets.push_back({std::stoi(t3[0]), std::stoi(t3[1]),
                                  std::stoi(t3[2])});
          }
          cfg.sparse_layers.push_back(std::move(ly));
        } else {
          DownLayer ly;
          ly.in_c = std::stoi(t2[0]);
          ly.out_c = std::stoi(t2[1]);
          ly.k = std::stoi(t2[2]);
          ly.stride = std::stoi(t2[3]);
          ly.pad = std::stoi(t2[4]);
          ly.w_off = std::stoull(t2[5]);
          ly.w_len = std::stoull(t2[6]);
          ly.b_len = std::stoi(t2[7]);
          ly.bn_eps = std::stof(t2[8]);
          cfg.down_layers.push_back(std::move(ly));
        }
      }
      continue;
    }
    table_lines.push_back(l);
  }

  for (auto &l : table_lines) {
    auto toks = split(l);
    const std::string &k = toks[0];
    if (k == "B") cfg.B = std::stoi(toks[1]);
    else if (k == "N") cfg.N = std::stoi(toks[1]);
    else if (k == "image_h") cfg.image_h = std::stoi(toks[1]);
    else if (k == "image_w") cfg.image_w = std::stoi(toks[1]);
    else if (k == "feat_h") cfg.feat_h = std::stoi(toks[1]);
    else if (k == "feat_w") cfg.feat_w = std::stoi(toks[1]);
    else if (k == "xbound") cfg.xbound = parse_floats(l);
    else if (k == "ybound") cfg.ybound = parse_floats(l);
    else if (k == "zbound") cfg.zbound = parse_floats(l);
    else if (k == "dbound") cfg.dbound = parse_floats(l);
    else if (k == "nx") cfg.nx = parse_ints(l);
    else if (k == "lidar_pcr") cfg.lidar_pcr = parse_floats(l);
    else if (k == "lidar_voxel") cfg.lidar_voxel = parse_floats(l);
    else if (k == "lidar_grid") cfg.lidar_grid = parse_ints(l);
    else if (k == "lidar_max_pts") cfg.lidar_max_pts = std::stoi(toks[1]);
    else if (k == "lidar_max_voxels") cfg.lidar_max_voxels = std::stoi(toks[1]);
    else if (k == "sparse_out") cfg.sparse_out = parse_ints(l);
    else if (k == "vt_out_c") cfg.vt_out_c = std::stoi(toks[1]);
    else if (k == "head_nclass") cfg.head_nclass = std::stoi(toks[1]);
    else if (k == "head_nprop") cfg.head_nprop = std::stoi(toks[1]);
    else if (k == "head_nms_k") cfg.head_nms_k = std::stoi(toks[1]);
    else if (k == "head_bev") cfg.head_bev = std::stoi(toks[1]);
    else if (k == "head_pc") cfg.head_pc = parse_floats(l);
    else if (k == "head_osf") cfg.head_osf = std::stoi(toks[1]);
    else if (k == "head_voxel") cfg.head_voxel = parse_floats(l);
    else if (k == "head_post") cfg.head_post = parse_floats(l);
    else if (k == "head_score_thr") cfg.head_score_thr = std::stof(toks[1]);
  }
  return cfg;
}

// ================================================================ project_depth
// Mirrors deploy/preprocess.py project_depth (batch=1, all cameras).
Tensor project_depth(const Config &cfg, const Tensor &points,
                     const std::vector<Mat4> &lidar2image,
                     const std::vector<Mat4> &img_aug,
                     const Mat4 &lidar_aug) {
  int ncam = cfg.N, H = cfg.image_h, W = cfg.image_w;
  Tensor depth({ncam, 1, H, W});
  const int n = (int)points.shape[0];

  // inverse lidar aug
  Mat3 la_rot = lidar_aug.rot3();
  float la_t[3];
  lidar_aug.col3(3, la_t);
  Mat3 la_rot_inv = Mat3::inverse(la_rot);

  for (int c = 0; c < ncam; ++c) {
    Mat3 l2i = lidar2image[c].rot3();
    float l2i_t[3];
    lidar2image[c].col3(3, l2i_t);
    Mat3 ia_rot = img_aug[c].rot3();
    float ia_t[3];
    img_aug[c].col3(3, ia_t);

    for (int i = 0; i < n; ++i) {
      const float *p = points.data.data() + i * points.shape[1];
      float v[3] = {p[0] - la_t[0], p[1] - la_t[1], p[2] - la_t[2]};
      float world[3];
      la_rot_inv.matvec(v, world);
      // lidar2image
      float c0[3];
      l2i.matvec(world, c0);
      c0[0] += l2i_t[0];
      c0[1] += l2i_t[1];
      c0[2] += l2i_t[2];
      float dist = c0[2];
      float z = std::clamp(c0[2], 1e-5f, 1e5f);
      float px = c0[0] / z, py = c0[1] / z;
      // img aug
      float x = ia_rot.m[0] * px + ia_rot.m[1] * py + ia_t[0];
      float y = ia_rot.m[3] * px + ia_rot.m[4] * py + ia_t[1];
      float gx = y, gy = x;  // grid-sample (y, x) order swap
      if (gx >= 0 && gx < H && gy >= 0 && gy < W) {
        int64_t ix = (int64_t)gx, iy = (int64_t)gy;
        depth.at({c, 0, ix, iy}) = dist;
      }
    }
  }
  return depth;
}

// ================================================================ geometry
// Mirrors BaseTransform.get_geometry + frustum (batch=1).
Tensor compute_geometry(const Config &cfg, const std::vector<Mat4> &camera2lidar,
                        const std::vector<Mat4> &intrins,
                        const std::vector<Mat4> &img_aug,
                        const Mat4 &lidar_aug) {
  int D = cfg.depth_bins(), fH = cfg.feat_h, fW = cfg.feat_w;
  // frustum: xs = linspace(0, iW-1, fW), ys = linspace(0, iH-1, fH),
  //          ds = arange(dbound) (D values)
  std::vector<float> xs(fW), ys(fH), ds(D);
  for (int i = 0; i < fW; ++i)
    xs[i] = (cfg.image_w - 1) * (float)i / (fW - 1);
  for (int i = 0; i < fH; ++i)
    ys[i] = (cfg.image_h - 1) * (float)i / (fH - 1);
  for (int i = 0; i < D; ++i)
    ds[i] = cfg.dbound[0] + cfg.dbound[2] * i;

  Mat3 la_rot = lidar_aug.rot3();
  float la_t[3];
  lidar_aug.col3(3, la_t);

  Tensor geom({cfg.N, D, fH, fW, 3});
  for (int c = 0; c < cfg.N; ++c) {
    Mat3 post_inv = Mat3::inverse(img_aug[c].rot3());
    float post_t[3];
    img_aug[c].col3(3, post_t);
    Mat3 intr_inv = Mat3::inverse(intrins[c].rot3());
    Mat3 c2l = camera2lidar[c].rot3();
    float c2l_t[3];
    camera2lidar[c].col3(3, c2l_t);
    Mat3 combine = Mat3::mul(c2l, intr_inv);

    for (int di = 0; di < D; ++di)
      for (int hi = 0; hi < fH; ++hi)
        for (int wi = 0; wi < fW; ++wi) {
          // undo post
          float v[3] = {xs[wi] - post_t[0], ys[hi] - post_t[1],
                        ds[di] - post_t[2]};
          float u[3];
          post_inv.matvec(v, u);
          // scale by depth
          float cam[3] = {u[0] * u[2], u[1] * u[2], u[2]};
          float w3[3];
          combine.matvec(cam, w3);
          w3[0] += c2l_t[0];
          w3[1] += c2l_t[1];
          w3[2] += c2l_t[2];
          // extra lidar aug
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
// Mirrors BaseTransform.bev_pool (CUDA kernel equivalent): fold depth prob
// into features is done by the caller; here we do cell quantization, range
// filter, sort-by-rank segment sum (exact float sum like the kernel).
Tensor bev_pool(const Config &cfg, const Tensor &geom, const Tensor &x) {
  // geom: (N, D, fH, fW, 3); x: (N, D, fH, fW, C)
  int D = (int)geom.shape[1], fH = (int)geom.shape[2], fW = (int)geom.shape[3];
  int C = (int)x.shape[4];
  int nx0 = cfg.nx[0], nx1 = cfg.nx[1], nx2 = cfg.nx[2];
  float bx = cfg.xbound[0], by = cfg.ybound[0], bz = cfg.zbound[0];
  float dx = cfg.xbound[2], dy = cfg.ybound[2], dz = cfg.zbound[2];

  // cell coords + rank, keep in-range points
  std::vector<int64_t> keep_rank;
  std::vector<int64_t> keep_feat;  // row index into x.reshape(-1, C)
  keep_rank.reserve(x.shape[0] * D * fH * fW / 2);
  for (int64_t i = 0; i < geom.shape[0] * D * fH * fW; ++i) {
    const float *g = geom.data.data() + i * 3;
    int cx = (int)std::floor((g[0] - (bx - dx / 2)) / dx);
    int cy = (int)std::floor((g[1] - (by - dy / 2)) / dy);
    int cz = (int)std::floor((g[2] - (bz - dz / 2)) / dz);
    if (cx >= 0 && cx < nx0 && cy >= 0 && cy < nx1 && cz >= 0 && cz < nx2) {
      // ranks_bev = b*(D*H*W) + z*(H*W) + x*W + y  (b=0, D=nx2, H=nx0, W=nx1)
      int64_t rank = (int64_t)cz * (nx0 * nx1) + (int64_t)cx * nx1 + cy;
      keep_rank.push_back(rank);
      keep_feat.push_back(i);
    }
  }
  int nk = (int)keep_rank.size();

  // sort by rank (value-stable; sum is order-independent)
  std::vector<int> order(nk);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(),
            [&](int a, int b) { return keep_rank[a] < keep_rank[b]; });

  // segment sum -> out (C, nx0, nx1) [Dz = nx2 = 1 collapsed]
  Tensor out({C, nx0, nx1});
  int64_t i = 0;
  while (i < nk) {
    int64_t r = keep_rank[order[i]];
    int64_t j = i;
    // accumulate
    std::vector<double> acc(C, 0.0);
    for (; j < nk && keep_rank[order[j]] == r; ++j) {
      const float *xf = x.data.data() + keep_feat[order[j]] * C;
      for (int c = 0; c < C; ++c) acc[c] += xf[c];
    }
    int64_t cx = r / nx1, cy = r % nx1;
    for (int c = 0; c < C; ++c) out.at({c, cx, cy}) = (float)acc[c];
    i = j;
  }
  return out;  // (C, 360, 360)
}

// ================================================================ downsample
// 3x (conv3x3 + bn + relu); mid conv stride 2; blob layout per DownLayer.
Tensor downsample_conv(const Config &cfg, const Tensor &bev,
                       const std::vector<float> &blob) {
  Tensor cur = bev;  // (C, H, W)
  for (const auto &ly : cfg.down_layers) {
    int C_in = ly.in_c, C_out = ly.out_c, k = ly.k, s = ly.stride, p = ly.pad;
    const float *W = blob.data() + ly.w_off;        // (C_out, C_in, k, k)
    const float *bias = blob.data() + ly.w_off + ly.w_len;
    const float *g = blob.data() + ly.w_off + ly.w_len + ly.b_len;
    const float *bta = blob.data() + ly.w_off + ly.w_len + 2 * ly.b_len;
    const float *mean = blob.data() + ly.w_off + ly.w_len + 3 * ly.b_len;
    const float *var = blob.data() + ly.w_off + ly.w_len + 4 * ly.b_len;
    int H = (int)cur.shape[1], Wd = (int)cur.shape[2];
    int Ho = (H - k + 2 * p) / s + 1, Wo = (Wd - k + 2 * p) / s + 1;
    Tensor nxt({C_out, Ho, Wo});
    for (int co = 0; co < C_out; ++co)
      for (int ho = 0; ho < Ho; ++ho)
        for (int wo = 0; wo < Wo; ++wo) {
          float acc = bias[co];
          for (int ci = 0; ci < C_in; ++ci)
            for (int ki = 0; ki < k; ++ki)
              for (int kj = 0; kj < k; ++kj) {
                int hi = ho * s - p + ki, wi = wo * s - p + kj;
                if (hi < 0 || hi >= H || wi < 0 || wi >= Wd) continue;
                acc += W[((co * C_in + ci) * k + ki) * k + kj] *
                       cur.at({ci, hi, wi});
              }
          float s2 = var[co] + ly.bn_eps;
          float v = (acc - mean[co]) / std::sqrt(s2) * g[co] + bta[co];
          nxt.at({co, ho, wo}) = std::max(v, 0.0f);  // ReLU
        }
    cur = std::move(nxt);
  }
  return cur;
}

// ================================================================ voxelize
// Mirrors ops/voxel/src/voxelization.cpp hard_voxelize (CPU), NDim=3.
void voxelize(const Tensor &points, const Config &cfg, Tensor &feats,
              Tensor &coords, Tensor &sizes) {
  const int n = (int)points.shape[0];
  const int nDim = 3;
  std::vector<float> vs = cfg.lidar_voxel;
  std::vector<float> cr = cfg.lidar_pcr;  // [minx,miny,minz,maxx,maxy,maxz]
  std::vector<int> grid = cfg.lidar_grid;
  int max_points = cfg.lidar_max_pts;
  int max_voxels = cfg.lidar_max_voxels;

  // key = ((x*grid_y + y)*grid_z + z) packed; map key -> voxel index
  std::unordered_map<int64_t, int> hash;
  std::vector<std::vector<float>> voxel_pts;  // per voxel: up to max_points*(5)
  std::vector<std::vector<int>> voxel_coords;
  std::vector<int> voxel_counts;

  int n_voxels = 0;
  for (int i = 0; i < n; ++i) {
    const float *p = points.data.data() + i * points.shape[1];
    int coord[3];
    bool ok = true;
    for (int d = 0; d < nDim; ++d) {
      float c = std::floor((p[d] - cr[d]) / vs[d]);
      if (c < 0 || c >= grid[d]) { ok = false; break; }
      coord[d] = (int)c;
    }
    if (!ok) continue;
    int64_t key =
        ((int64_t)coord[0] * grid[1] + coord[1]) * grid[2] + coord[2];
    auto it = hash.find(key);
    int vidx;
    if (it == hash.end()) {
      if (n_voxels >= max_voxels) continue;
      vidx = n_voxels++;
      hash[key] = vidx;
      voxel_pts.emplace_back();
      voxel_coords.push_back({coord[0], coord[1], coord[2]});
      voxel_counts.push_back(0);
    } else {
      vidx = it->second;
      if (voxel_counts[vidx] >= max_points) continue;
    }
    for (int d = 0; d < (int)points.shape[1]; ++d)
      voxel_pts[vidx].push_back(p[d]);
    voxel_counts[vidx]++;
  }

  // flatten: feats (M,5), coords (M,4), sizes (M)
  int M = 0;
  for (int c : voxel_counts) M += c;
  feats.resize({M, (int)points.shape[1]});
  coords.resize({M, 4});
  sizes.resize({n_voxels});
  int r = 0;
  for (int v = 0; v < n_voxels; ++v) {
    sizes.data[v] = (float)voxel_counts[v];
    for (int d = 0; d < nDim; ++d)
      coords.data[v * 4 + d] = (float)voxel_coords[v][d];
    coords.data[v * 4 + 3] = 0.0f;  // batch
    for (int i = 0; i < voxel_counts[v]; ++i)
      for (int d = 0; d < (int)points.shape[1]; ++d)
        feats.data[r * points.shape[1] + d] =
            voxel_pts[v][i * points.shape[1] + d];
    ++r;
  }
}

}  // namespace dsb
