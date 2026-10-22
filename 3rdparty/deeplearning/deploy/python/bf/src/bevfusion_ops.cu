// bevfusion_ops.cu —— CUDA-BEVFusion 参考算子 Python 移植（torch extension）
//
// 算子与 deploy/cpp/src/bevfusion/ 对应：
//   camera-depth.cu        compute_depth    lidar 点云投影 -> 6 相机深度图
//   camera-geometry.cu     compute_geometry frustum 反投影 -> geometry 索引 + ranks
//   camera-bevpool.cu      bevpool          depth_weights x camera_feature 区间求和
//   lidar-voxelization.cu  voxel_offset + voxel_mean 体素化（offset 计算 / 桶内均值累加）
// 排序 / 分桶等离散索引操作在 Python 侧用 torch 完成（stable argsort 语义与 thrust 一致）。
//
// 编译：python setup.py build_ext --inplace（deploy/python/bf/ 下）
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <vector>

namespace bfops {

// ------------------------------------------------------------------ helpers
// 与参考 camera-depth.cu / camera-geometry.cu 相同
static __forceinline__ __device__ float project(const float4& T, const float3& p) {
  return T.x * p.x + T.y * p.y + T.z * p.z + T.w;
}
static __forceinline__ __device__ float dot(const float4& T, const float3& p) {
  return T.x * p.x + T.y * p.y + T.z * p.z;
}
static __forceinline__ __device__ float3 inverse_project(const float4* T, const float3& p) {
  float3 r;
  r.x = p.x - T[0].w;
  r.y = p.y - T[1].w;
  r.z = p.z - T[2].w;
  return make_float3(dot(T[0], r), dot(T[1], r), dot(T[2], r));
}
static __forceinline__ __device__ float clampf(float x, float lower, float upper) {
  return x < lower ? lower : (x > upper ? upper : x);
}

// ------------------------------------------------------------------ compute_depth
// points: (N,5) half，前 3 通道 xyz；lidar2image / img_aug_matrix: (6,4,4) float32
// 输出 depth: (num_camera, image_height, image_width) half
__global__ void compute_depth_kernel(unsigned int num_points, const half* points, const float4* img_aug_matrix,
                                     const float4* lidar2image, unsigned int points_channel, int num_camera,
                                     unsigned int image_width, unsigned int image_height, half* depth_out) {
  int tid = (blockIdx.x * blockDim.x + threadIdx.x);
  if (tid >= num_points) return;

  float3 point = make_float3(__half2float(points[points_channel * tid]),
                             __half2float(points[points_channel * tid + 1]),
                             __half2float(points[points_channel * tid + 2]));
  for (int icamera = 0; icamera < num_camera; ++icamera) {
    float dist = clampf(project(lidar2image[4 * icamera + 2], point), 1e-5, 1e5);
    float3 projed = make_float3(project(lidar2image[4 * icamera + 0], point) / dist,
                                project(lidar2image[4 * icamera + 1], point) / dist, dist);
    float x = project(img_aug_matrix[4 * icamera + 0], projed);
    float y = project(img_aug_matrix[4 * icamera + 1], projed);
    if (x >= 0 && x < image_width && y >= 0 && y < image_height) {
      int ix = static_cast<int>(x);
      int iy = static_cast<int>(y);
      depth_out[(icamera * image_height + iy) * image_width + ix] = __float2half(dist);
    }
  }
}

torch::Tensor compute_depth(torch::Tensor points, torch::Tensor lidar2image, torch::Tensor img_aug_matrix,
                            int64_t image_width, int64_t image_height) {
  const at::cuda::OptionalCUDAGuard guard(device_of(points));
  int num_points = points.size(0);
  int num_camera = lidar2image.size(0);
  int points_channel = points.size(1);
  auto depth = torch::zeros({num_camera, image_height, image_width},
                            points.options().dtype(torch::kHalf));
  auto stream = at::cuda::getCurrentCUDAStream();
  unsigned int block = 256;
  unsigned int grid = (num_points + block - 1) / block;
  compute_depth_kernel<<<grid, block, 0, stream>>>(
      num_points, (const half*)points.data_ptr(),
      (const float4*)lidar2image.data_ptr(), (const float4*)img_aug_matrix.data_ptr(),
      points_channel, num_camera, image_width, image_height,
      (half*)depth.data_ptr());
  return depth;
}

// ------------------------------------------------------------------ compute_geometry
// frustum: (numel,3) float32（Python 按 D/fH/fW 生成，参考 create_frustum_kernel）
// camera2lidar / camera_intrinsics_inv / img_aug_matrix_inv: (6,4,4) float32
// bx/dx: (3,) float32；nx/geometry_dim: (3,) int64；num_camera
// 输出 geometry / ranks: (num_camera*numel,) int32；keep_count: int32
__global__ void compute_geometry_kernel(unsigned int numel_frustum, const float3* frustum, const float4* camera2lidar,
                                        const float4* camera_intrins_inv, const float4* img_aug_matrix_inv,
                                        float3 bx, float3 dx, int3 nx, unsigned int* keep_count,
                                        int* ranks, int3 geometry_dim, unsigned int num_camera,
                                        int* geometry_out) {
  int tid = (blockIdx.x * blockDim.x + threadIdx.x);
  if (tid >= numel_frustum) return;

  float3 point = frustum[tid];
  for (int icamera = 0; icamera < (int)num_camera; ++icamera) {
    float3 projed = inverse_project(img_aug_matrix_inv + 4 * icamera, point);
    projed.x *= projed.z;
    projed.y *= projed.z;
    projed = make_float3(dot(camera_intrins_inv[4 * icamera + 0], projed),
                         dot(camera_intrins_inv[4 * icamera + 1], projed),
                         dot(camera_intrins_inv[4 * icamera + 2], projed));
    projed = make_float3(project(camera2lidar[4 * icamera + 0], projed),
                         project(camera2lidar[4 * icamera + 1], projed),
                         project(camera2lidar[4 * icamera + 2], projed));

    int _pid = icamera * numel_frustum + tid;
    int3 coords;
    coords.x = int((projed.x - (bx.x - dx.x / 2.0)) / dx.x);
    coords.y = int((projed.y - (bx.y - dx.y / 2.0)) / dx.y);
    coords.z = int((projed.z - (bx.z - dx.z / 2.0)) / dx.z);
    geometry_out[_pid] = (coords.z * geometry_dim.z * geometry_dim.y + coords.x) * geometry_dim.x + coords.y;

    bool kept = coords.x >= 0 && coords.y >= 0 && coords.z >= 0 &&
                coords.x < nx.x && coords.y < nx.y && coords.z < nx.z;
    if (!kept) {
      ranks[_pid] = 0;
    } else {
      atomicAdd(keep_count, 1);
      ranks[_pid] = (coords.x * nx.y + coords.y) * nx.z + coords.z;
    }
  }
}

std::vector<torch::Tensor> compute_geometry(torch::Tensor frustum, torch::Tensor camera2lidar,
                                            torch::Tensor camera_intrinsics_inv, torch::Tensor img_aug_matrix_inv,
                                            torch::Tensor bx, torch::Tensor dx, torch::Tensor nx,
                                            torch::Tensor geometry_dim, int64_t num_camera) {
  const at::cuda::OptionalCUDAGuard guard(device_of(frustum));
  int numel = frustum.size(0);
  int numel_geometry = num_camera * numel;
  auto opts32 = frustum.options().dtype(torch::kInt32);
  auto geometry = torch::empty({numel_geometry}, opts32);
  auto ranks = torch::empty({numel_geometry}, opts32);
  auto keep_count = torch::zeros({1}, opts32);
  auto stream = at::cuda::getCurrentCUDAStream();
  unsigned int block = 256;
  unsigned int grid = (numel + block - 1) / block;
  float3 bx_ = {bx[0].item<float>(), bx[1].item<float>(), bx[2].item<float>()};
  float3 dx_ = {dx[0].item<float>(), dx[1].item<float>(), dx[2].item<float>()};
  int3 nx_ = {nx[0].item<int>(), nx[1].item<int>(), nx[2].item<int>()};
  int3 gdim_ = {geometry_dim[0].item<int>(), geometry_dim[1].item<int>(), geometry_dim[2].item<int>()};
  compute_geometry_kernel<<<grid, block, 0, stream>>>(
      numel, (const float3*)frustum.data_ptr(), (const float4*)camera2lidar.data_ptr(),
      (const float4*)camera_intrinsics_inv.data_ptr(), (const float4*)img_aug_matrix_inv.data_ptr(),
      bx_, dx_, nx_, (unsigned int*)keep_count.data_ptr(), (int*)ranks.data_ptr(),
      gdim_, num_camera, (int*)geometry.data_ptr());
  return {geometry, ranks, keep_count};
}

// ------------------------------------------------------------------ bevpool
// camera_feature: (BN, farea, C) half（HWC）；depth_weights: (BN*D*farea,) half
// intervals: (M,3) int32（x=区间起 y=区间止 z=目标 voxel 索引）；indices: (K,) int32（排序后原始索引）
// 输出 bev: (C, out_h, out_w) half（输出偏移 interval.z + c*out_h*out_w，同参考）
__global__ void bevpool_kernel(const half* camera_feature, const half* depth_weights, unsigned int nchannel,
                               const int3* intervals, unsigned int n_intervals, const unsigned int* indices,
                               unsigned int out_h, unsigned int out_w, unsigned int ndepth, unsigned int farea,
                               half* output_bevfeat) {
  int interval_index = blockIdx.y * blockDim.y + threadIdx.y;
  int channel = blockIdx.x * blockDim.x + threadIdx.x;
  if (interval_index >= n_intervals || channel >= nchannel) return;

  int3 interval = intervals[interval_index];
  float accumulate = 0.0f;
  for (int i = interval.x; i < interval.y; i++) {
    int indice = indices[i];
    int camera_index = indice / (ndepth * farea);
    int fm_inner_index = indice % farea;
    float depth_weight = __half2float(depth_weights[indice]);
    float feature = __half2float(camera_feature[(camera_index * farea + fm_inner_index) * nchannel + channel]);
    accumulate = fmaf(feature, depth_weight, accumulate);
  }
  unsigned int output_offset = interval.z + channel * out_h * out_w;
  output_bevfeat[output_offset] = __float2half(accumulate);
}

torch::Tensor bevpool(torch::Tensor camera_feature, torch::Tensor depth_weights, torch::Tensor intervals,
                      torch::Tensor indices, int64_t out_h, int64_t out_w, int64_t ndepth, int64_t farea,
                      int64_t nchannel) {
  const at::cuda::OptionalCUDAGuard guard(device_of(camera_feature));
  int n_intervals = intervals.size(0);
  if (n_intervals == 0) {
    return torch::zeros({nchannel, out_h, out_w}, camera_feature.options().dtype(torch::kHalf));
  }
  auto bev = torch::zeros({nchannel, out_h, out_w}, camera_feature.options().dtype(torch::kHalf));
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 block(16, 16, 1);
  dim3 grid((nchannel + block.x - 1) / block.x, (n_intervals + block.y - 1) / block.y, 1);
  bevpool_kernel<<<grid, block, 0, stream>>>(
      (const half*)camera_feature.data_ptr(), (const half*)depth_weights.data_ptr(),
      nchannel, (const int3*)intervals.data_ptr(), n_intervals,
      (const unsigned int*)indices.data_ptr(), out_h, out_w, ndepth, farea,
      (half*)bev.data_ptr());
  return bev;
}

// ------------------------------------------------------------------ voxelization
// points: (N, nf) half；min_range / voxel_size: (3,) float32；grid_size: (3,) int64
// 输出 offsets: (N,) int32（无效点 = -1）、valid: (N,) uint8
__global__ void voxel_offset_kernel(unsigned int num_points, const half* points, float3 min_range, float3 voxel_size,
                                    int3 grid_size, int* offsets, unsigned char* valid) {
  int tid = (blockIdx.x * blockDim.x + threadIdx.x);
  if (tid >= num_points) return;

  float px = __half2float(points[tid * 3 + 0]);
  float py = __half2float(points[tid * 3 + 1]);
  float pz = __half2float(points[tid * 3 + 2]);

  int voxel_idx = int(floorf((px - min_range.x) / voxel_size.x));
  int voxel_idy = int(floorf((py - min_range.y) / voxel_size.y));
  int voxel_idz = int(floorf((pz - min_range.z) / voxel_size.z));
  bool kept = voxel_idx >= 0 && voxel_idy >= 0 && voxel_idz >= 0 &&
              voxel_idx < grid_size.x && voxel_idy < grid_size.y && voxel_idz < grid_size.z;
  if (!kept) {
    offsets[tid] = -1;
    valid[tid] = 0;
    return;
  }
  offsets[tid] = (voxel_idz * grid_size.y + voxel_idy) * grid_size.x + voxel_idx;
  valid[tid] = 1;
}

// 桶内均值累加：points 按 sel 顺序取；每点属于 bucket_id（截断 max_points 后）
// 输出 features: (num_buckets, nf) float32（未归一，调用方除 counts）
//      counts:   (num_buckets,) int32
__global__ void voxel_mean_kernel(unsigned int num_sel, const half* points, const int* sel, const int* bucket_id,
                                  unsigned int num_buckets, unsigned int num_feature,
                                  float* features, int* counts) {
  int tid = (blockIdx.x * blockDim.x + threadIdx.x);
  if (tid >= num_sel) return;

  int b = bucket_id[tid];
  if (b < 0 || (unsigned int)b >= num_buckets) return;
  int src = sel[tid];
  for (unsigned int j = 0; j < num_feature; ++j) {
    atomicAdd(features + b * num_feature + j, __half2float(points[src * num_feature + j]));
  }
  atomicAdd(counts + b, 1);
}

std::vector<torch::Tensor> voxel_offset(torch::Tensor points, torch::Tensor min_range, torch::Tensor voxel_size,
                                        torch::Tensor grid_size) {
  const at::cuda::OptionalCUDAGuard guard(device_of(points));
  int num_points = points.size(0);
  auto opts32 = points.options().dtype(torch::kInt32);
  auto offsets = torch::empty({num_points}, opts32);
  auto valid = torch::zeros({num_points}, points.options().dtype(torch::kByte));
  auto stream = at::cuda::getCurrentCUDAStream();
  unsigned int block = 256;
  unsigned int grid = (num_points + block - 1) / block;
  float3 min_ = {min_range[0].item<float>(), min_range[1].item<float>(), min_range[2].item<float>()};
  float3 vs_ = {voxel_size[0].item<float>(), voxel_size[1].item<float>(), voxel_size[2].item<float>()};
  int3 gs_ = {grid_size[0].item<int>(), grid_size[1].item<int>(), grid_size[2].item<int>()};
  voxel_offset_kernel<<<grid, block, 0, stream>>>(
      num_points, (const half*)points.data_ptr(), min_, vs_, gs_,
      (int*)offsets.data_ptr(), (unsigned char*)valid.data_ptr());
  return {offsets, valid};
}

torch::Tensor voxel_mean(torch::Tensor points, torch::Tensor sel, torch::Tensor bucket_id, int64_t num_buckets) {
  const at::cuda::OptionalCUDAGuard guard(device_of(points));
  int num_sel = sel.size(0);
  int num_feature = points.size(1);
  auto opts32 = points.options().dtype(torch::kInt32);
  auto features = torch::zeros({num_buckets, num_feature}, points.options().dtype(torch::kFloat));
  auto counts = torch::zeros({num_buckets}, opts32);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (num_sel == 0) {
    return features;
  }
  unsigned int block = 256;
  unsigned int grid = (num_sel + block - 1) / block;
  voxel_mean_kernel<<<grid, block, 0, stream>>>(
      num_sel, (const half*)points.data_ptr(), (const int*)sel.data_ptr(),
      (const int*)bucket_id.data_ptr(), num_buckets, num_feature,
      (float*)features.data_ptr(), (int*)counts.data_ptr());
  return features;
}

// ------------------------------------------------------------------ binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("compute_depth", &compute_depth, "lidar points -> per-camera depth map (half)");
  m.def("compute_geometry", &compute_geometry, "frustum -> geometry indices + ranks");
  m.def("bevpool", &bevpool, "depth weights x camera feature interval sum");
  m.def("voxel_offset", &voxel_offset, "points -> voxel linear offsets + valid mask");
  m.def("voxel_mean", &voxel_mean, "bucket mean of point features");
}

}  // namespace bfops
