#pragma once
// 数据预处理层：加载导出输入并转换为模型输入（深度投影 / 体素化）。
// 与 Python 侧 deploy/preprocess.py（Preprocessor）逐层对应。

#include "config.hpp"
#include "tensor.hpp"

#include <string>
#include <vector>

namespace dsb {

class Preprocessor {
 public:
  // 一份导出样本的原始输入（tools/export_cpp_input.py 产出）。
  struct Inputs {
    Tensor img;             // (B, N, 3, image_h, image_w)，已归一化
    Tensor points;          // (n, 5)
    std::vector<Mat4> camera2lidar;  // (N, 4, 4)
    std::vector<Mat4> intrins;       // (N, 4, 4)
    std::vector<Mat4> img_aug;       // (N, 4, 4)
    Mat4 lidar_aug;                  // (4, 4)
  };

  explicit Preprocessor(const Config &cfg);

  // 从导出目录读取原始输入（inputs/img.bin / points.bin / matrices.bin）。
  Inputs load_inputs(const std::string &input_dir) const;

  // lidar 点 -> 每相机稀疏深度图 (N, 1, H, W)。
  Tensor project_depth(const Inputs &in) const;

  // 点云硬体素化 -> (feats (M,5), coords (M,4)=(b,x,y,z), sizes (M,))。
  void voxelize(const Inputs &in, Tensor &feats, Tensor &coords,
                Tensor &sizes) const;

 private:
  const Config &cfg_;
};

}  // namespace dsb
