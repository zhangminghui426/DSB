#pragma once
// Minimal tensor / matrix structures for the DSB C++ inference runtime.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>
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
