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

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace bevfusion {
namespace lidar {

// 调试：BEVFUSION_DUMP_DIR 时落盘 voxelize 输出（coords int32 / feats half，
// save_tensor 同格式），供 model_diff --backend trt-full 隔离 voxelize vs SCN。
static void dump_voxelize(const Voxelization* v, void* stream) {
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
  fwrite(dims, sizeof(int), 2, f);
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

    native_scn_ = spconv::load_engine_from_onnx(param_.model, static_cast<spconv::Precision>(param_.precision));
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
    return native_scn_->output(0)->features().ptr<nvtype::half>();
  }

  virtual std::vector<int64_t> shape() override {
    return native_scn_->output(0)->features().shape;
  }

 private:
  // 读 BEVFUSION_DUMP_DIR/torch_lidar_coords(+feats)（save_tensor 格式）。
  // torch coords 为 [batch,z,y,x]（mmcv 惯例），引擎期望 [batch,x,y,z]，这里转换。
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
    // [batch,z,y,x] -> [batch,x,y,z]
    for (int i = 0; i < M; ++i) {
      int b = hcoords[i * 4 + 0], z = hcoords[i * 4 + 1], y = hcoords[i * 4 + 2], x = hcoords[i * 4 + 3];
      hcoords[i * 4 + 0] = b; hcoords[i * 4 + 1] = x; hcoords[i * 4 + 2] = y; hcoords[i * 4 + 3] = z;
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

std::shared_ptr<SCN> create_scn(const SCNParameter& param) {
  std::shared_ptr<SCNImplement> instance(new SCNImplement());
  if (!instance->init(param)) {
    instance.reset();
  }
  return instance;
}

};  // namespace lidar
};  // namespace bevfusion