/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "lidar-scn.hpp"

#include <spconv/engine.hpp>
#include "lidar-scn-onnx-parser.hpp"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "common/check.hpp"
#include "bevfusion/lidar-scn-layers.hpp"

namespace bevfusion {
namespace lidar {

// 调试：BEVFUSION_DUMP_DIR 时落盘 voxelize 输出（coords int32 / feats half，
// save_tensor 同格式），供 model_diff --backend trt-full 隔离 voxelize vs SCN。
static void dump_voxelize(Voxelization* v, void* stream) {
  const char* dir = getenv("BEVFUSION_DUMP_DIR");
  if (!dir) return;
  cudaStream_t s = (cudaStream_t)stream;
  int nv = (int)v->num_voxels(), id = (int)v->indices_dim(), fd = (int)v->voxel_dim();
  if (nv <= 0 || fd <= 0 || id <= 0) return;
  std::vector<int> h_idx((size_t)nv * id);
  std::vector<nvtype::half> h_feat((size_t)nv * fd);
  checkRuntime(cudaMemcpyAsync(h_idx.data(), v->indices(), (size_t)nv * id * sizeof(int),
                               cudaMemcpyDeviceToHost, s));
  checkRuntime(cudaMemcpyAsync(h_feat.data(), v->features(), (size_t)nv * fd * sizeof(nvtype::half),
                               cudaMemcpyDeviceToHost, s));
  checkRuntime(cudaStreamSynchronize(s));

  FILE* f = fopen((std::string(dir) + "/trt_lidar_coords").c_str(), "wb");
  int hdr[3] = {0x33FF1101, 2, 1};  // 1 = int32
  fwrite(hdr, sizeof(int), 3, f);
  int dims[2] = {nv, id};
  fwrite(dims, sizeof(int), 2, f);
  fwrite(h_idx.data(), sizeof(int), (size_t)nv * id, f);
  fclose(f);
  f = fopen((std::string(dir) + "/trt_lidar_feats").c_str(), "wb");
  hdr[2] = 2;  // 2 = float16
  fwrite(hdr, sizeof(int), 3, f);
  int fdims[2] = {nv, fd};  // feats 是 (nv, feat_dim)，不是 indices_dim
  fwrite(fdims, sizeof(int), 2, f);
  fwrite(h_feat.data(), sizeof(nvtype::half), (size_t)nv * fd, f);
  fclose(f);
  printf("[dump] trt_lidar_coords/feats num_voxels=%d indices_dim=%d feat_dim=%d\n", nv, id, fd);
}

class SCNImplement : public SCN {
 public:
  ~SCNImplement() {
    if (override_coords_dev_) checkRuntime(cudaFree(override_coords_dev_));
    if (override_feats_dev_) checkRuntime(cudaFree(override_feats_dev_));
  }

  bool init(const SCNParameter& param) {
    this->param_ = param;
    voxelization_ = create_voxelization(param_.voxelization);
    if (voxelization_ == nullptr) return false;

    native_scn_ = spconv::load_engine_from_onnx(
        param_.model, static_cast<spconv::Precision>(param_.precision));
    return native_scn_ != nullptr;
  }

  virtual const nvtype::half* forward(const nvtype::half* points, unsigned int num_points, void* stream) override {
    if (!try_load_override_(stream)) {
      // 常规路径：C++ CUDA voxelize + 引擎
      voxelization_->forward(points, num_points, stream, param_.order);
      dump_voxelize(voxelization_.get(), stream);
      native_scn_->input(0)->features().reference((void*)voxelization_->features(),
                                                  std::vector<int64_t>{voxelization_->num_voxels(), voxelization_->voxel_dim()},
                                                  spconv::DataType::Float16);
      native_scn_->input(0)->indices().reference((void*)voxelization_->indices(),
                                                 std::vector<int64_t>{voxelization_->num_voxels(), voxelization_->indices_dim()},
                                                 spconv::DataType::Int32);
    } else {
      // 隔离实验：BEVFUSION_DUMP_DIR/torch_lidar_coords(+feats) 存在时，用 torch 的
      // voxelize 输出喂引擎，定位"voxelize 差异 vs SCN 引擎差异"。
      native_scn_->input(0)->features().reference((void*)override_feats_dev_,
                                                  std::vector<int64_t>{override_nv_, 5}, spconv::DataType::Float16);
      native_scn_->input(0)->indices().reference((void*)override_coords_dev_,
                                                 std::vector<int64_t>{override_nv_, 4}, spconv::DataType::Int32);
    }
    native_scn_->input(0)->set_grid_size(voxelization_->grid_size());
    native_scn_->forward(stream);
    if (getenv("BEVFUSION_SCN_DEBUG")) {
      auto s = shape();
      printf("[scn] output shape:");
      for (auto v : s) printf(" %lld", (long long)v);
      printf("\n");
    }
    return native_scn_->output(0)->features().ptr<nvtype::half>();
  }

  virtual std::vector<int64_t> shape() override {
    return native_scn_->output(0)->features().shape;
  }

 private:
  // 读 BEVFUSION_DUMP_DIR/torch_lidar_coords(+feats)（save_tensor 格式）。
  // 本仓库 sparse encoder 的 coords 已是 [batch,x,y,z]（与 C++ XYZ voxelize
  // 一致），直接喂引擎，不再做轴变换。
  bool try_load_override_(void* stream) {
    const char* dir = getenv("BEVFUSION_DUMP_DIR");
    if (!dir) return false;
    std::string cpath = std::string(dir) + "/torch_lidar_coords";
    std::string fpath = std::string(dir) + "/torch_lidar_feats";
    FILE* fc = fopen(cpath.c_str(), "rb");
    FILE* ff = fopen(fpath.c_str(), "rb");
    if (!fc || !ff) {
      if (fc) fclose(fc);
      if (ff) fclose(ff);
      return false;
    }
    int hdr[3];
    if (fread(hdr, sizeof(int), 3, fc) != 3 || hdr[1] != 2) { fclose(fc); fclose(ff); return false; }
    int dims[2];
    if (fread(dims, sizeof(int), 2, fc) != 2) { fclose(fc); fclose(ff); return false; }
    int M = dims[0], Dc = dims[1];
    std::vector<int> hcoords((size_t)M * Dc);
    fread(hcoords.data(), sizeof(int), (size_t)M * Dc, fc);
    fclose(fc);
    int fdims[2];
    if (fread(hdr, sizeof(int), 3, ff) != 3 || fread(fdims, sizeof(int), 2, ff) != 2) { fclose(ff); return false; }
    int Mf = fdims[0], Df = fdims[1];
    std::vector<nvtype::half> hfeats((size_t)Mf * Df);
    fread(hfeats.data(), sizeof(nvtype::half), (size_t)Mf * Df, ff);
    fclose(ff);
    if (M != Mf || Df != 5 || Dc != 4) {
      printf("[override] 尺寸不符 coords=%dx%d feats=%dx%d，跳过\n", M, Dc, Mf, Df);
      return false;
    }
    // coords 已是 [batch,x,y,z]（本仓库 sparse encoder 约定），默认原样使用。
    // 调试可用 BEVFUSION_COORD_PERM=zyx 等 3 字符排列做轴变换（第 j 位输出
    // 取原第 perm[j] 轴），用于定位引擎坐标序约定。
    const char* perm = getenv("BEVFUSION_COORD_PERM");
    if (perm != nullptr && strlen(perm) == 3 &&
        perm[0] >= '0' && perm[0] <= '2' && perm[1] >= '0' && perm[1] <= '2' &&
        perm[2] >= '0' && perm[2] <= '2') {
      for (int i = 0; i < M; ++i) {
        int c0 = hcoords[i * 4 + 1 + (perm[0] - '0')];
        int c1 = hcoords[i * 4 + 1 + (perm[1] - '0')];
        int c2 = hcoords[i * 4 + 1 + (perm[2] - '0')];
        hcoords[i * 4 + 1] = c0;
        hcoords[i * 4 + 2] = c1;
        hcoords[i * 4 + 3] = c2;
      }
    }
    cudaStream_t s = (cudaStream_t)stream;
    if (!override_coords_dev_) checkRuntime(cudaMalloc(&override_coords_dev_, (size_t)M * 4 * sizeof(int)));
    if (!override_feats_dev_) checkRuntime(cudaMalloc(&override_feats_dev_, (size_t)M * 5 * sizeof(nvtype::half)));
    checkRuntime(cudaMemcpyAsync(override_coords_dev_, hcoords.data(), (size_t)M * 4 * sizeof(int),
                                 cudaMemcpyHostToDevice, s));
    checkRuntime(cudaMemcpyAsync(override_feats_dev_, hfeats.data(), (size_t)M * 5 * sizeof(nvtype::half),
                                 cudaMemcpyHostToDevice, s));
    checkRuntime(cudaStreamSynchronize(s));
    override_nv_ = M;
    printf("[override] 用 torch voxelize 输出喂 SCN：num_voxels=%d feat_dim=%d\n", M, Df);
    return true;
  }

  SCNParameter param_;
  std::shared_ptr<Voxelization> voxelization_;
  std::shared_ptr<spconv::Engine> native_scn_;
  int* override_coords_dev_ = nullptr;
  nvtype::half* override_feats_dev_ = nullptr;
  int override_nv_ = 0;
};

// ---------------------------------------------------------------------------
// Per-layer SCN：libspconv 在单个 graph 里串联多个 SparseConvolution 时，对稀疏
// 邻域（缺失邻居）会算错；逐 conv 建 engine + 主机侧 ReLU/Add 与 torch 对齐
// （lidar_bev corr ~0.99996，单 graph 仅 ~0.13）。
// 层序列对应 configs/det_convfuser.py 的 SparseEncoder；需要
// <scn_dir>/layers/conv{0..20}.onnx（deploy/scripts/export.py 自动拆分）。
// ---------------------------------------------------------------------------
static std::string scn_layer_dir(const std::string& model_path) {
  const char* env = getenv("BEVFUSION_SCN_LAYERS_DIR");
  if (env != nullptr && env[0] != '\0') return env;
  auto pos = model_path.find_last_of("/\\");
  std::string dir = (pos == std::string::npos) ? std::string(".") : model_path.substr(0, pos);
  return dir + "/layers";
}

static bool scn_file_exists(const std::string& path) {
  FILE* f = fopen(path.c_str(), "rb");
  if (f == nullptr) return false;
  fclose(f);
  return true;
}

class SCNPerLayer : public SCN {
 public:
  ~SCNPerLayer() override {
    if (folded_ != nullptr) checkRuntime(cudaFree(folded_));
  }

  bool init(const SCNParameter& param) {
    param_ = param;
    voxelization_ = create_voxelization(param_.voxelization);
    if (voxelization_ == nullptr) return false;

    // 层序列/网格按 configs/det_convfuser.py（5 维特征、±54m、0.075m）硬编码；
    // 配置不符时回退单 graph，避免静默算错。
    auto gs = voxelization_->grid_size();
    if (param_.voxelization.num_feature != 5 || param_.order != CoordinateOrder::XYZ ||
        gs.size() != 3 || gs[0] != 1440 || gs[1] != 1440 || gs[2] != 41) {
      printf("[scn] per-layer: unsupported config (features=%d grid=%d,%d,%d order=%d)\n",
             param_.voxelization.num_feature, gs.size() > 0 ? gs[0] : -1,
             gs.size() > 1 ? gs[1] : -1, gs.size() > 2 ? gs[2] : -1,
             (int)param_.order);
      return false;
    }

    layers_dir_ = scn_layer_dir(param_.model);
    engines_.resize(kNumConv);
    for (int i = 0; i < kNumConv; ++i) {
      std::string path = layers_dir_ + "/conv" + std::to_string(i) + ".onnx";
      if (!scn_file_exists(path)) {
        printf("[scn] per-layer: missing %s\n", path.c_str());
        return false;
      }
      engines_[i] = spconv::load_engine_from_onnx(
          path, static_cast<spconv::Precision>(param_.precision));
      if (engines_[i] == nullptr) {
        printf("[scn] per-layer: failed to load %s\n", path.c_str());
        return false;
      }
    }
    checkRuntime(cudaMalloc(&folded_, (size_t)kOutC * kOutZ * kOutX * kOutY *
                                           sizeof(nvtype::half)));
    printf("[scn] per-layer mode: %d engines from %s\n", kNumConv, layers_dir_.c_str());
    return true;
  }

  const nvtype::half* forward(const nvtype::half* points, unsigned int num_points,
                              void* stream) override {
    voxelization_->forward(points, num_points, stream, param_.order);
    int nv = (int)voxelization_->num_voxels();
    if (nv <= 0) return nullptr;
    tensors_[0] = {(const nvtype::half*)voxelization_->features(),
                   (const int*)voxelization_->indices(), nv,
                   (int)voxelization_->voxel_dim()};

    // conv / ReLU / residual Add 顺序与 ONNX trace 一致；ReLU 原地做，
    // 因此残差 identity 直接复用前一个 conv 输出 buffer。
    run_conv(0, 0, 1, 1440, 1440, 41, stream); relu(1, stream);
    run_conv(1, 1, 3, 1440, 1440, 41, stream); relu(3, stream);
    run_conv(2, 3, 5, 1440, 1440, 41, stream); add(5, 1, stream); relu(5, stream);
    run_conv(3, 5, 8, 1440, 1440, 41, stream); relu(8, stream);
    run_conv(4, 8, 10, 1440, 1440, 41, stream); add(10, 5, stream); relu(10, stream);
    run_conv(5, 10, 13, 1440, 1440, 41, stream); relu(13, stream);
    run_conv(6, 13, 15, 720, 720, 21, stream); relu(15, stream);
    run_conv(7, 15, 17, 720, 720, 21, stream); add(17, 13, stream); relu(17, stream);
    run_conv(8, 17, 20, 720, 720, 21, stream); relu(20, stream);
    run_conv(9, 20, 22, 720, 720, 21, stream); add(22, 17, stream); relu(22, stream);
    run_conv(10, 22, 25, 720, 720, 21, stream); relu(25, stream);
    run_conv(11, 25, 27, 360, 360, 11, stream); relu(27, stream);
    run_conv(12, 27, 29, 360, 360, 11, stream); add(29, 25, stream); relu(29, stream);
    run_conv(13, 29, 32, 360, 360, 11, stream); relu(32, stream);
    run_conv(14, 32, 34, 360, 360, 11, stream); add(34, 29, stream); relu(34, stream);
    run_conv(15, 34, 37, 360, 360, 11, stream); relu(37, stream);
    run_conv(16, 37, 39, 180, 180, 5, stream); relu(39, stream);
    run_conv(17, 39, 41, 180, 180, 5, stream); add(41, 37, stream); relu(41, stream);
    run_conv(18, 41, 44, 180, 180, 5, stream); relu(44, stream);
    run_conv(19, 44, 46, 180, 180, 5, stream); add(46, 41, stream); relu(46, stream);
    run_conv(20, 46, 49, 180, 180, 5, stream); relu(49, stream);

    const auto& last = tensors_[49];
    scn_scatter_fold(last.feat, last.idx, folded_, last.m, last.c,
                     kOutX, kOutY, kOutZ, stream);
    return folded_;
  }

  std::vector<int64_t> shape() override { return {1, kOutC * kOutZ, kOutX, kOutY}; }

 private:
  static constexpr int kNumConv = 21;
  static constexpr int kOutC = 128;
  static constexpr int kOutX = 180;
  static constexpr int kOutY = 180;
  static constexpr int kOutZ = 2;

  struct TensorRef {
    const nvtype::half* feat = nullptr;
    const int* idx = nullptr;
    int m = 0;
    int c = 0;
  };

  void run_conv(int conv, int in_id, int out_id, int gx, int gy, int gz, void* stream) {
    auto& e = engines_[conv];
    const auto& in = tensors_[in_id];
    e->input(0)->features().reference((void*)in.feat, {in.m, in.c},
                                      spconv::DataType::Float16);
    e->input(0)->indices().reference((void*)in.idx, {in.m, 4},
                                     spconv::DataType::Int32);
    e->input(0)->set_grid_size({gx, gy, gz});
    e->forward(stream);
    auto fs = e->output(0)->features().shape;
    tensors_[out_id] = {(const nvtype::half*)e->output(0)->features().ptr<nvtype::half>(),
                        (const int*)e->output(0)->indices().ptr<int>(),
                        (int)fs[0], (int)fs[1]};
  }

  void relu(int id, void* stream) {
    auto& t = tensors_[id];
    scn_relu_half((nvtype::half*)t.feat, (size_t)t.m * t.c, stream);
  }

  void add(int id, int ident, void* stream) {
    auto& t = tensors_[id];
    const auto& o = tensors_[ident];
    Assertf(t.m == o.m && t.c == o.c, "SCN add shape mismatch: %d x %d vs %d x %d",
            t.m, t.c, o.m, o.c);
    scn_add_half((nvtype::half*)t.feat, o.feat, (size_t)t.m * t.c, stream);
  }

  SCNParameter param_;
  std::string layers_dir_;
  std::shared_ptr<Voxelization> voxelization_;
  std::vector<std::shared_ptr<spconv::Engine>> engines_;
  std::array<TensorRef, 54> tensors_;
  nvtype::half* folded_ = nullptr;
};

std::shared_ptr<SCN> create_scn(const SCNParameter& param) {
  const char* mode = getenv("BEVFUSION_SCN_PER_LAYER");
  bool use_per_layer = false;
  if (mode != nullptr) {
    use_per_layer = (strcmp(mode, "0") != 0);
  } else {
    use_per_layer = scn_file_exists(scn_layer_dir(param.model) + "/conv0.onnx");
  }
  if (use_per_layer) {
    std::shared_ptr<SCNPerLayer> instance(new SCNPerLayer());
    if (instance->init(param)) return instance;
    printf("[scn] per-layer init failed, fallback to monolithic engine\n");
  }
  std::shared_ptr<SCNImplement> instance(new SCNImplement());
  if (!instance->init(param)) {
    instance.reset();
  }
  return instance;
}

};  // namespace lidar
};  // namespace bevfusion