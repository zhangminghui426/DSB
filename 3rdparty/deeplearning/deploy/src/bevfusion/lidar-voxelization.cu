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

#include <cuda_fp16.h>
#include <thrust/binary_search.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include "common/check.hpp"
#include "common/launch.cuh"
#include "lidar-voxelization.hpp"

namespace bevfusion {
namespace lidar {

// 输入顺序 voxelize（与 mmcv hard_voxelize deterministic 对齐）：
// 按 voxel key 稳定排序，组内保持输入点序，取前 max_points 个点算均值。
// pass1：按输入 index 原位写 keys/idxs（无效点写哨兵），供稳定排序使用。
__global__ void voxel_key_kernel(size_t num_points, const half* points, VoxelizationParameter param,
                                 uint64_t* keys, int* idxs, unsigned int* valid) {
  int i = cuda_linear_index;
  if (i >= (int)num_points) return;
  keys[i] = 0xFFFFFFFFFFFFFFFFull;  // 无效点哨兵（排序后落在末尾）
  idxs[i] = i;
  const half* p = points + (size_t)i * param.num_feature;
  float px = __half2float(p[0]), py = __half2float(p[1]), pz = __half2float(p[2]);
  if (px < param.min_range.x || px >= param.max_range.x || py < param.min_range.y || py >= param.max_range.y ||
      pz < param.min_range.z || pz >= param.max_range.z)
    return;
  int x = (int)floorf((px - param.min_range.x) / param.voxel_size.x);
  int y = (int)floorf((py - param.min_range.y) / param.voxel_size.y);
  int z = (int)floorf((pz - param.min_range.z) / param.voxel_size.z);
  if (x < 0 || x >= param.grid_size.x || y < 0 || y >= param.grid_size.y || z < 0 || z >= param.grid_size.z) return;
  keys[i] = (uint64_t)z * (uint64_t)param.grid_size.y * (uint64_t)param.grid_size.x +
            (uint64_t)y * (uint64_t)param.grid_size.x + (uint64_t)x;
  atomicAdd(valid, 1u);
}

// pass3：start[i] = (i==0) ? 0 : ends[i-1]
__global__ void voxel_start_kernel(size_t num_voxels, const int* ends, int* starts) {
  int i = cuda_linear_index;
  if (i >= (int)num_voxels) return;
  starts[i] = (i == 0) ? 0 : ends[i - 1];
}

// pass4：每 voxel 取组内前 max_points 个点（输入顺序）算均值 + 写 indices
__global__ void voxel_reduce_kernel(size_t num_voxels, const int* starts, const int* ends,
                                    const uint64_t* sorted_keys, const int* sorted_idxs, const half* points,
                                    VoxelizationParameter param, int max_points, half* voxel_features,
                                    unsigned int* voxel_indices, int xyz_order) {
  int v = cuda_linear_index;
  if (v >= (int)num_voxels) return;
  int s = starts[v], e = ends[v];
  int n = min(e - s, max_points);
  int64_t k = (int64_t)sorted_keys[s];
  int z = (int)(k / ((int64_t)param.grid_size.y * param.grid_size.x));
  int rem = (int)(k % ((int64_t)param.grid_size.y * param.grid_size.x));
  int y = rem / param.grid_size.x, x = rem % param.grid_size.x;
  if (xyz_order == 1) {  // XYZ（BEVFusion）
    ((uint4*)voxel_indices)[v] = make_uint4(0, (unsigned)x, (unsigned)y, (unsigned)z);
  } else {  // ZYX（CenterPoint）
    ((uint4*)voxel_indices)[v] = make_uint4(0, (unsigned)z, (unsigned)y, (unsigned)x);
  }
  if (n <= 0) return;
  float sum[8] = {0.0f};
  for (int j = 0; j < n; ++j) {
    const half* p = points + (size_t)sorted_idxs[s + j] * param.num_feature;
    for (int f = 0; f < param.num_feature; ++f) sum[f] += __half2float(p[f]);
  }
  for (int f = 0; f < param.num_feature; ++f)
    voxel_features[(size_t)v * param.num_feature + f] = __float2half(sum[f] / n);
}

nvtype::Int3 VoxelizationParameter::compute_grid_size(const nvtype::Float3& max_range,
                                                      const nvtype::Float3& min_range,
                                                      const nvtype::Float3& voxel_size) {
  nvtype::Int3 size;
  size.x = static_cast<int>(std::round((max_range.x - min_range.x) / voxel_size.x));
  size.y = static_cast<int>(std::round((max_range.y - min_range.y) / voxel_size.y));
  size.z = static_cast<int>(std::round((max_range.z - min_range.z) / voxel_size.z));
  return size;
}

class VoxelizationImplement : public Voxelization {
 public:
  virtual ~VoxelizationImplement() {
    if (d_keys_) checkRuntime(cudaFree(d_keys_));
    if (d_idxs_) checkRuntime(cudaFree(d_idxs_));
    if (d_voxel_keys_) checkRuntime(cudaFree(d_voxel_keys_));
    if (d_ends_) checkRuntime(cudaFree(d_ends_));
    if (d_starts_) checkRuntime(cudaFree(d_starts_));
    if (d_valid_) checkRuntime(cudaFree(d_valid_));
    if (h_valid_) checkRuntime(cudaFreeHost(h_valid_));
    if (d_voxel_features_) checkRuntime(cudaFree(d_voxel_features_));
    if (d_voxel_indices_) checkRuntime(cudaFree(d_voxel_indices_));
  }

  bool init(VoxelizationParameter param) {
    this->param_ = param;
    this->output_grid_size_ = {(int)param_.grid_size.x, (int)param_.grid_size.y, (int)param_.grid_size.z + 1};
    checkRuntime(cudaMalloc(&d_keys_, (size_t)param_.max_points * sizeof(uint64_t)));
    checkRuntime(cudaMalloc(&d_idxs_, (size_t)param_.max_points * sizeof(int)));
    checkRuntime(cudaMalloc(&d_voxel_keys_, (size_t)param_.max_voxels * sizeof(uint64_t)));
    checkRuntime(cudaMalloc(&d_ends_, (size_t)param_.max_voxels * sizeof(int)));
    checkRuntime(cudaMalloc(&d_starts_, (size_t)param_.max_voxels * sizeof(int)));
    checkRuntime(cudaMalloc(&d_valid_, sizeof(unsigned int)));
    checkRuntime(cudaMallocHost(&h_valid_, sizeof(unsigned int)));
    checkRuntime(cudaMalloc(&d_voxel_features_, (size_t)param_.max_voxels * param_.num_feature * sizeof(half)));
    checkRuntime(cudaMalloc(&d_voxel_indices_, (size_t)param_.max_voxels * 4 * sizeof(unsigned int)));
    return true;
  }

  // points 为 half device 指针；内部走默认流，结束前同步（SCN 引擎随后读取）。
  virtual void forward(const nvtype::half* points, int num_points, void* stream,
                       CoordinateOrder output_order) override {
    const half* _points = reinterpret_cast<const half*>(points);
    if (output_order == CoordinateOrder::XYZ) {
      this->output_grid_size_ = {(int)param_.grid_size.x, (int)param_.grid_size.y, (int)param_.grid_size.z + 1};
    } else if (output_order == CoordinateOrder::ZYX) {
      this->output_grid_size_ = {(int)param_.grid_size.z + 1, (int)param_.grid_size.y, (int)param_.grid_size.x};
    } else {
      Assertf(false, "Invalid output_order: %d", static_cast<int>(output_order));
    }

    checkRuntime(cudaMemsetAsync(d_valid_, 0, sizeof(unsigned int), 0));
    cuda_linear_launch(voxel_key_kernel, 0, num_points, _points, param_, d_keys_, d_idxs_, d_valid_);
    checkRuntime(cudaStreamSynchronize(0));
    checkRuntime(cudaMemcpy(h_valid_, d_valid_, sizeof(unsigned int), cudaMemcpyDeviceToHost));

    real_num_voxels_ = 0;
    unsigned int valid = *h_valid_;
    if (valid == 0) return;

    // 稳定排序：同一 voxel 内保持输入点序
    thrust::device_ptr<uint64_t> dk(d_keys_);
    thrust::device_ptr<int> di(d_idxs_);
    thrust::stable_sort_by_key(dk, dk + num_points, di);

    // 唯一 key + 每 voxel 组范围 [starts, ends)
    int num_voxels = (int)(thrust::unique_copy(dk, dk + valid, thrust::device_ptr<uint64_t>(d_voxel_keys_)) -
                           thrust::device_ptr<uint64_t>(d_voxel_keys_));
    if (num_voxels <= 0) return;
    if (num_voxels > param_.max_voxels) num_voxels = param_.max_voxels;
    thrust::upper_bound(dk, dk + valid, thrust::device_ptr<uint64_t>(d_voxel_keys_),
                        thrust::device_ptr<uint64_t>(d_voxel_keys_) + num_voxels,
                        thrust::device_ptr<int>(d_ends_));
    cuda_linear_launch(voxel_start_kernel, 0, num_voxels, d_ends_, d_starts_);
    int xyz_order = (output_order == CoordinateOrder::XYZ) ? 1 : 0;
    cuda_linear_launch(voxel_reduce_kernel, 0, num_voxels, d_starts_, d_ends_, d_keys_, d_idxs_, _points, param_,
                       param_.max_points_per_voxel, d_voxel_features_, d_voxel_indices_, xyz_order);
    checkRuntime(cudaStreamSynchronize(0));
    real_num_voxels_ = num_voxels;
    (void)stream;
  }

  virtual unsigned int num_voxels() override { return real_num_voxels_; }

  virtual unsigned int voxel_dim() override { return param_.num_feature; }

  virtual unsigned int indices_dim() override { return 4; }

  virtual std::vector<int> grid_size() override { return output_grid_size_; }

  virtual const void* indices() override { return d_voxel_indices_; }

  virtual const void* features() override { return d_voxel_features_; }

  virtual CoordinateOrder order() override { return order_; }

 private:
  CoordinateOrder order_ = CoordinateOrder::NoneOrder;
  VoxelizationParameter param_;
  unsigned int real_num_voxels_ = 0;
  std::vector<int> output_grid_size_;

  uint64_t* d_keys_ = nullptr;
  int* d_idxs_ = nullptr;
  uint64_t* d_voxel_keys_ = nullptr;
  int* d_ends_ = nullptr;
  int* d_starts_ = nullptr;
  unsigned int* d_valid_ = nullptr;
  unsigned int* h_valid_ = nullptr;
  half* d_voxel_features_ = nullptr;
  unsigned int* d_voxel_indices_ = nullptr;
};

std::shared_ptr<Voxelization> create_voxelization(VoxelizationParameter parameter) {
  std::shared_ptr<VoxelizationImplement> instance(new VoxelizationImplement());
  if (!instance->init(parameter)) {
    instance.reset();
  }
  return instance;
}

};  // namespace lidar
};  // namespace bevfusion
