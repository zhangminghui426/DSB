// 数据预处理层实现：加载导出输入 + 深度投影 + 体素化。
// 镜像 Python 侧 deploy/preprocess.py（project_depth / voxelize）。

#include "preprocess/preprocess.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <unordered_map>

namespace dsb {

namespace {

std::vector<Mat4> read_mat4_block(const std::vector<float> &all, int idx) {
  std::vector<Mat4> v;
  for (int k = 0; k < idx; ++k) {
    Mat4 m{};
    std::memcpy(m.m, all.data() + k * 16, 16 * sizeof(float));
    v.push_back(m);
  }
  return v;
}

}  // namespace

Preprocessor::Preprocessor(const Config &cfg) : cfg_(cfg) {}

Preprocessor::Inputs Preprocessor::load_inputs(
    const std::string &input_dir) const {
  Inputs in;
  in.img = Tensor({cfg_.B, cfg_.N, 3, cfg_.image_h, cfg_.image_w});
  in.img.data = read_bin(input_dir + "/inputs/img.bin", in.img.data.size());

  std::ifstream f(input_dir + "/inputs/points.bin", std::ios::binary);
  if (!f) throw std::runtime_error("cannot open inputs/points.bin");
  int32_t n = 0;
  f.read(reinterpret_cast<char *>(&n), sizeof(n));
  in.points = Tensor({n, 5});
  f.read(reinterpret_cast<char *>(in.points.data.data()),
         (std::streamsize)(n * 5 * sizeof(float)));
  std::printf("[dsb] points: %d\n", n);

  std::vector<float> mats =
      read_bin(input_dir + "/inputs/matrices.bin", 19 * 16);
  in.camera2lidar = read_mat4_block(mats, 6);  // idx 0..5
  in.intrins = read_mat4_block(
      std::vector<float>(mats.begin() + 6 * 16, mats.end()), 6);
  in.img_aug = read_mat4_block(
      std::vector<float>(mats.begin() + 12 * 16, mats.end()), 6);
  std::memcpy(in.lidar_aug.m, mats.data() + 18 * 16, 16 * sizeof(float));
  return in;
}

Tensor Preprocessor::project_depth(const Inputs &in) const {
  const int ncam = cfg_.N, H = cfg_.image_h, W = cfg_.image_w;
  const int n = (int)in.points.shape[0];
  Tensor depth({ncam, 1, H, W});

  // 逆 lidar 增强
  const Mat3 la_rot = in.lidar_aug.rot3();
  float la_t[3];
  in.lidar_aug.col3(3, la_t);
  const Mat3 la_rot_inv = Mat3::inverse(la_rot);

  for (int c = 0; c < ncam; ++c) {
    // lidar -> camera：camera2lidar 是 camera->lidar（sensor2lidar），取逆
    const Mat3 c2l_rot = in.camera2lidar[c].rot3();
    float c2l_t[3];
    in.camera2lidar[c].col3(3, c2l_t);
    const Mat3 c2l_inv = Mat3::inverse(c2l_rot);
    // intrins 内参（row-major 4x4：fx=m[0], fy=m[5], cx=m[2], cy=m[6]）
    const float fx = in.intrins[c].m[0], fy = in.intrins[c].m[5];
    const float cx = in.intrins[c].m[2], cy = in.intrins[c].m[6];
    // img 增强完整 3x3 仿射（z=1）
    const Mat3 ia_rot = in.img_aug[c].rot3();
    float ia_t[3];
    in.img_aug[c].col3(3, ia_t);

    for (int i = 0; i < n; ++i) {
      const float *p = in.points.data.data() + i * in.points.shape[1];
      const float v[3] = {p[0] - la_t[0], p[1] - la_t[1], p[2] - la_t[2]};
      float world[3];
      la_rot_inv.matvec(v, world);
      // lidar -> camera 刚体
      const float vc[3] = {world[0] - c2l_t[0], world[1] - c2l_t[1],
                           world[2] - c2l_t[2]};
      float cam[3];
      c2l_inv.matvec(vc, cam);
      const float dist = cam[2];
      const float z = std::clamp(cam[2], 1e-5f, 1e5f);
      const float px = fx * cam[0] / z + cx;
      const float py = fy * cam[1] / z + cy;
      // img 增强（与 torch cur_img_aug_matrix.matmul 一致，含第三列*1）
      const float x = ia_rot.m[0] * px + ia_rot.m[1] * py + ia_rot.m[2] + ia_t[0];
      const float y = ia_rot.m[3] * px + ia_rot.m[4] * py + ia_rot.m[5] + ia_t[1];
      const float gx = y, gy = x;  // grid-sample (y, x) 顺序交换
      if (gx >= 0 && gx < H && gy >= 0 && gy < W) {
        const int64_t ix = (int64_t)gx, iy = (int64_t)gy;
        depth.at({c, 0, ix, iy}) = dist;
      }
    }
  }
  return depth;
}

void Preprocessor::voxelize(const Inputs &in, Tensor &feats, Tensor &coords,
                            Tensor &sizes) const {
  const int n = (int)in.points.shape[0];
  const int n_dim = 3;
  const std::vector<float> &vs = cfg_.lidar_voxel;
  const std::vector<float> &cr = cfg_.lidar_pcr;  // [minx,miny,minz,maxx,...]
  const std::vector<int> &grid = cfg_.lidar_grid;
  const int max_points = cfg_.lidar_max_pts;
  const int max_voxels = cfg_.lidar_max_voxels;

  // key = ((x*grid_y + y)*grid_z + z) 打包；hash key -> 体素序号
  std::unordered_map<int64_t, int> hash;
  std::vector<std::vector<float>> voxel_pts;  // 每体素：至多 max_points*5
  std::vector<std::vector<int>> voxel_coords;
  std::vector<int> voxel_counts;

  int n_voxels = 0;
  for (int i = 0; i < n; ++i) {
    const float *p = in.points.data.data() + i * in.points.shape[1];
    int coord[3];
    bool ok = true;
    for (int d = 0; d < n_dim; ++d) {
      const float c = std::floor((p[d] - cr[d]) / vs[d]);
      if (c < 0 || c >= grid[d]) { ok = false; break; }
      coord[d] = (int)c;
    }
    if (!ok) continue;
    const int64_t key =
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
    for (int d = 0; d < (int)in.points.shape[1]; ++d)
      voxel_pts[vidx].push_back(p[d]);
    voxel_counts[vidx]++;
  }

  // 展平：feats (M,5)，coords (M,4)，sizes (M)
  int M = 0;
  for (int c : voxel_counts) M += c;
  feats.resize({M, (int)in.points.shape[1]});
  coords.resize({M, 4});
  sizes.resize({n_voxels});
  int r = 0;
  for (int v = 0; v < n_voxels; ++v) {
    sizes.data[v] = (float)voxel_counts[v];
    for (int d = 0; d < n_dim; ++d)
      coords.data[v * 4 + d] = (float)voxel_coords[v][d];
    coords.data[v * 4 + 3] = 0.0f;  // batch
    for (int i = 0; i < voxel_counts[v]; ++i)
      for (int d = 0; d < (int)in.points.shape[1]; ++d)
        feats.data[r * in.points.shape[1] + d] =
            voxel_pts[v][i * in.points.shape[1] + d];
    ++r;
  }
}

}  // namespace dsb
