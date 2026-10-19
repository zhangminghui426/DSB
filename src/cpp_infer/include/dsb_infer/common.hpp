#pragma once
// Common tensor / config structures for the DSB C++ inference runtime.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace dsb {

// Minimal float tensor (CPU). Row-major contiguous storage.
struct Tensor {
  std::vector<float> data;
  std::vector<int64_t> shape;

  Tensor() = default;
  explicit Tensor(std::vector<int64_t> s) : shape(std::move(s)) {
    data.resize(numel());
  }

  int64_t numel() const {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    return n;
  }
  float *ptr() { return data.data(); }
  const float *ptr() const { return data.data(); }

  int64_t stride(int dim) const {
    int64_t s = 1;
    for (int i = (int)shape.size() - 1; i > dim; --i) s *= shape[i];
    return s;
  }
  int64_t offset(const std::vector<int64_t> &idx) const {
    int64_t o = 0;
    for (size_t d = 0; d < idx.size(); ++d) o += idx[d] * stride((int)d);
    return o;
  }
  float &at(const std::vector<int64_t> &idx) { return data[offset(idx)]; }
  const float &at(const std::vector<int64_t> &idx) const {
    return data[offset(idx)];
  }

  void reshape(std::vector<int64_t> s) {
    int64_t n = 1;
    for (int64_t d : s) n *= d;
    if (n != numel()) throw std::runtime_error("reshape size mismatch");
    shape = std::move(s);
  }
  void resize(std::vector<int64_t> s) {
    shape = std::move(s);
    data.assign(numel(), 0.0f);
  }
};

// mat3 / mat4 helpers (row-major).
struct Mat3 {
  float m[9];
  static Mat3 identity();
  static Mat3 inverse(const Mat3 &a);
  static Mat3 mul(const Mat3 &a, const Mat3 &b);
  void matvec(const float v[3], float out[3]) const;
};

struct Mat4 {
  float m[16];
  Mat3 rot3() const;
  void col3(int c, float out[3]) const;  // first 3 elems of column c
};

inline int64_t prod(const std::vector<int64_t> &v) {
  int64_t n = 1;
  for (int64_t x : v) n *= x;
  return n;
}

// ------------------------------------------------------------------ config
// key = space-separated values, '#' comments. Parsed from tools/export_cpp_input.py
// meta.txt; sparse/downsample layer tables are appended as structured lines.
struct SparseLayer {
  std::string type;         // "subm" | "strided"
  int in_c, out_c;
  std::vector<int> kernel;  // (kx, ky, kz)
  std::vector<int> stride;
  std::vector<int> pad;
  std::vector<std::vector<int>> offsets;  // per kernel index (dx, dy, dz)
  size_t w_off, w_len;                    // into the weight blob
  int b_len;
  float bn_eps;
};

struct DownLayer {
  int in_c, out_c, k, stride, pad;
  size_t w_off, w_len;
  int b_len;
  float bn_eps;
};

struct Config {
  int B = 1, N = 6;
  int image_h = 256, image_w = 704;
  int feat_h = 32, feat_w = 88;
  std::vector<float> xbound{ -54, 54, 0.3f }, ybound{ -54, 54, 0.3f };
  std::vector<float> zbound{ -10, 10, 20 }, dbound{ 1, 60, 0.5f };
  std::vector<int> nx{ 360, 360, 1 };
  // lidar
  std::vector<float> lidar_pcr{ -54, -54, -5, 54, 54, 3 };
  std::vector<float> lidar_voxel{ 0.075f, 0.075f, 0.2f };
  std::vector<int> lidar_grid{ 1440, 1440, 41 };
  int lidar_max_pts = 10, lidar_max_voxels = 160000;
  std::vector<int> sparse_out{ 180, 180, 2 };  // (X, Y, Z) after conv_out
  int vt_out_c = 80;
  // head
  int head_nclass = 10, head_nprop = 200, head_nms_k = 3, head_bev = 180 * 180;
  std::vector<float> head_pc{ -54, -54 };
  int head_osf = 8;
  std::vector<float> head_voxel{ 0.075f, 0.075f };
  std::vector<float> head_post{ -61.2f, -61.2f, -10, 61.2f, 61.2f, 10 };
  float head_score_thr = 0.0f;
  // layer tables
  std::vector<SparseLayer> sparse_layers;
  std::vector<DownLayer> down_layers;

  int depth_bins() const {  // torch.arange(dbound) length
    float lo = dbound[0], hi = dbound[1], step = dbound[2];
    return (int)std::ceil((hi - lo) / step - 1e-6);
  }
};

Config load_config(const std::string &path);

// ------------------------------------------------------------------- io
inline std::vector<float> read_bin(const std::string &path, size_t n) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::vector<float> v(n);
  f.read(reinterpret_cast<char *>(v.data()), n * sizeof(float));
  if (f.gcount() != (std::streamsize)(n * sizeof(float)))
    throw std::runtime_error("short read " + path);
  return v;
}

template <typename T>
void write_bin(const std::string &path, const std::vector<T> &v) {
  std::ofstream f(path, std::ios::binary);
  f.write(reinterpret_cast<const char *>(v.data()), v.size() * sizeof(T));
}

}  // namespace dsb
