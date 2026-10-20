// 数据预处理层实现：加载导出输入、深度投影、点云体素化。
#include "dsb_infer/preprocess.hpp"

#include <cstring>
#include <fstream>
#include <numeric>

#include "dsb_infer/ops.hpp"

namespace dsb {

namespace {

void read_bytes(const std::string &path, void *dst, size_t nbytes) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  f.read(reinterpret_cast<char *>(dst), static_cast<std::streamsize>(nbytes));
}

}  // namespace

Preprocessor::Preprocessor(const Config &cfg) : cfg_(cfg) {}

Preprocessor::Inputs Preprocessor::load_inputs(
    const std::string &input_dir) const {
  Inputs in;
  const auto idir = input_dir + "/inputs";

  // img.bin：直接读入 (1,6,3,H,W) 连续内存。
  const size_t n_img = 1UL * cfg_.num_camera * 3 * cfg_.image_h * cfg_.image_w;
  in.img = Tensor(std::vector<int>{1, cfg_.num_camera, 3, cfg_.image_h,
                                   cfg_.image_w},
                  Tensor::kFloat);
  read_bytes(idir + "/img.bin", in.img.data(), n_img * sizeof(float));

  // points.bin：int32 n + (n,5) float32。
  int32_t n = 0;
  read_bytes(idir + "/points.bin", &n, sizeof(int32_t));
  in.points = Tensor(std::vector<int>{n, 5}, Tensor::kFloat);
  read_bytes(idir + "/points.bin", in.points.data(),
             static_cast<size_t>(n) * 5 * sizeof(float));

  // matrices.bin：19 个 4x4 float32（6 cam2lidar + 6 intrins + 6 img_aug + 1 lidar_aug）。
  const int N = cfg_.num_camera;
  in.camera2lidar.assign(N, Mat4{});
  in.intrins.assign(N, Mat4{});
  in.img_aug.assign(N, Mat4{});
  std::vector<float> raw(19 * 16);
  read_bytes(idir + "/matrices.bin", raw.data(), raw.size() * sizeof(float));
  auto read_mat4_block = [&raw](int start, int n) {
    std::vector<Mat4> out(n);
    for (int i = 0; i < n; ++i)
      std::memcpy(out[i].data(), &raw[(start + i) * 16], 16 * sizeof(float));
    return out;
  };
  in.camera2lidar = read_mat4_block(0, N);
  in.intrins = read_mat4_block(6, N);
  in.img_aug = read_mat4_block(12, N);
  std::memcpy(in.lidar_aug.data(), &raw[18 * 16], 16 * sizeof(float));
  return in;
}

Tensor Preprocessor::project_depth(const Inputs &in) const {
  const int N = cfg_.num_camera;
  const int H = cfg_.image_h, W = cfg_.image_w;
  const int n = static_cast<int>(in.points.dims()[0]);
  Tensor depth(std::vector<int>{N, 1, H, W}, Tensor::kFloat);
  std::fill(depth.data(), depth.data() + depth.numel(), 0.f);

  // 逆 lidar 增强：xyz' = R^-1 (xyz - t)。
  const auto &L = in.lidar_aug;
  Mat4 L_inv{};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) L_inv[i][j] = L[j][i];
  for (int i = 0; i < 3; ++i) {
    float t = 0.f;
    for (int j = 0; j < 3; ++j) t += L_inv[i][j] * L[j][3];
    L_inv[i][3] = -t;
  }
  L_inv[3][3] = 1.f;

  for (int i = 0; i < n; ++i) {
    const float *p = in.points.data() + i * 5;
    // lidar 帧 -> 世界（逆增强）。
    float w[4] = {0, 0, 0, 1};
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 3; ++c) w[r] += L_inv[r][c] * p[c];
    for (int r = 0; r < 3; ++r) w[r] += L_inv[r][3];
    const float d = std::sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);

    for (int c = 0; c < N; ++c) {
      const auto &T = in.camera2lidar[c];
      // 世界 -> 相机：T 即 lidar2camera（在相机坐标中乘）。
      float u[3];
      for (int r = 0; r < 3; ++r)
        u[r] = T[r][3];  // 平移分量
      // 完整投影用第 3 列乘积，按 BEVFusion 语义：
      // lidar2image = lidar2camera * camera intrins；这里 T 为 cam2lidar，
      // 因此用其逆旋转。简化实现：
      float cam[3] = {0, 0, 0};
      for (int r = 0; r < 3; ++r)
        for (int c2 = 0; c2 < 3; ++c2) cam[r] += T[c2][r] * w[c2];
      for (int r = 0; r < 3; ++r) cam[r] = -cam[r] + u[r];
      const float z = cam[2] > 1e-3f ? cam[2] : 1e-3f;
      const float x = cam[0] / z, y = cam[1] / z;
      const auto &K = in.intrins[c];
      const float px = K[0][0] * x + K[0][1] * y + K[0][2];
      const float py = K[1][0] * x + K[1][1] * y + K[1][2];
      // 图像增强（缩放 + 平移）。
      const auto &A = in.img_aug[c];
      const float ax = A[0][0] * px + A[0][1] * py + A[0][3];
      const float ay = A[1][0] * px + A[1][1] * py + A[1][3];
      const int ux = static_cast<int>(std::round(ay));  // (row) y
      const int uy = static_cast<int>(std::round(ax));  // (col) x
      if (ux >= 0 && ux < H && uy >= 0 && uy < W) {
        float &cur = depth.data()[c * H * W + ux * W + uy];
        if (cur == 0.f || d < cur) cur = d;
      }
    }
  }
  return depth;
}

void Preprocessor::voxelize(const Inputs &in, Tensor &feats, Tensor &coords,
                            Tensor &sizes) const {
  const auto &r = cfg_.point_range;
  const auto &v = cfg_.voxel_size;
  const int max_vox = cfg_.max_voxels;
  const int max_pts = cfg_.max_points_per_voxel;
  const int nx = static_cast<int>(std::ceil((r[3] - r[0]) / v));
  const int ny = static_cast<int>(std::ceil((r[4] - r[1]) / v));
  const int nz = static_cast<int>(std::ceil((r[5] - r[2]) / v));

  std::map<std::array<int, 3>, int> hash;
  std::vector<std::vector<int>> point_lists;
  const int n = static_cast<int>(in.points.dims()[0]);
  for (int i = 0; i < n; ++i) {
    const float *p = in.points.data() + i * 5;
    if (p[0] < r[0] || p[0] >= r[3] || p[1] < r[1] || p[1] >= r[4] ||
        p[2] < r[2] || p[2] >= r[5])
      continue;
    std::array<int, 3> key{
        static_cast<int>(std::floor((p[0] - r[0]) / v)),
        static_cast<int>(std::floor((p[1] - r[1]) / v)),
        static_cast<int>(std::floor((p[2] - r[2]) / v))};
    auto it = hash.find(key);
    if (it == hash.end()) {
      if (static_cast<int>(point_lists.size()) >= max_vox) continue;
      hash[key] = static_cast<int>(point_lists.size());
      point_lists.push_back({});
      it = hash.find(key);
    }
    auto &lst = point_lists[it->second];
    if (static_cast<int>(lst.size()) < max_pts) lst.push_back(i);
  }

  const int M = static_cast<int>(point_lists.size());
  feats = Tensor(std::vector<int>{M, 5}, Tensor::kFloat);
  coords = Tensor(std::vector<int>{M, 4}, Tensor::kInt32);
  sizes = Tensor(std::vector<int>{M}, Tensor::kInt32);
  int m = 0;
  for (const auto &kv : hash) {
    const auto &key = kv.first;
    const auto &lst = point_lists[kv.second];
    // mean-reduce 点特征（x,y,z,reflectivity + 体素中心偏移）。
    float acc[5] = {0, 0, 0, 0, 0};
    for (int idx : lst) {
      const float *p = in.points.data() + idx * 5;
      for (int c = 0; c < 5; ++c) acc[c] += p[c];
    }
    const float denom = static_cast<float>(lst.size());
    float *f = feats.data() + m * 5;
    for (int c = 0; c < 5; ++c) f[c] = acc[c] / denom;
    int32_t *cd = coords.data<int32_t>() + m * 4;
    cd[0] = 0;  // batch
    cd[1] = key[0];
    cd[2] = key[1];
    cd[3] = key[2];
    sizes.data<int32_t>()[m] = static_cast<int32_t>(lst.size());
    ++m;
  }
}

}  // namespace dsb
