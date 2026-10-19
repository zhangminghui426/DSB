#pragma once
// Deployment config parsed from tools/export_cpp_input.py meta.txt.
// key = space-separated values, '#' comments; sparse/downsample layer
// tables are appended as structured lines.

#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

namespace dsb {

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

}  // namespace dsb
