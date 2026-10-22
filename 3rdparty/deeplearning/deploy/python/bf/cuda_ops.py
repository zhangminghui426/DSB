"""bf 自定义 CUDA 算子 Python 封装（CUDA-BEVFusion 参考实现 Python 版）。

编译产物为 bevfusion_ops.*.so（见 setup.py），本模块负责加载并补充
Python 侧的离散索引逻辑（排序 / 分桶 / interval 计算），与参考 kernel 语义一致：

  compute_depth    参考 camera-depth.cu：点云投影生成 (6,H,W) 深度图
  compute_geometry 参考 camera-geometry.cu：frustum 反投影 -> geometry + ranks（排序/分组在 Python）
  bevpool          参考 camera-bevpool.cu：depth_weights x camera_feature 区间求和
  voxelize         参考 lidar-voxelization.cu：体素化（offset 计算 + 桶内均值累加）
"""
import os

import torch

try:
    import bevfusion_ops as _ops
except ImportError as e:  # 未编译时给出明确指引
    raise ImportError(
        "bf 自定义 CUDA 算子未编译：cd 3rdparty/deeplearning/deploy/python/bf && "
        "python setup.py build_ext --inplace（需 nvcc + CUDA）"
    ) from e


# ---------------------------------------------------------------- compute_depth
def compute_depth(points: torch.Tensor, lidar2image: torch.Tensor, img_aug_matrix: torch.Tensor,
                  image_width: int = 704, image_height: int = 256) -> torch.Tensor:
    """点云 -> 6 相机深度图。

    points (N,5) half；lidar2image / img_aug_matrix (6,4,4) float32；
    返回 depth (6, image_height, image_width) half。
    """
    return _ops.compute_depth(points.contiguous().half(),
                              lidar2image.contiguous().float(),
                              img_aug_matrix.contiguous().float(),
                              image_width, image_height)


# ---------------------------------------------------------------- geometry
def make_frustum(num_camera: int, feat_w: int = 88, feat_h: int = 32, D: int = 118,
                 image_width: int = 704, image_height: int = 256,
                 dbound: tuple = (1.0, 60.0, 0.5), device="cuda") -> torch.Tensor:
    """frustum 预生成（同参考 create_frustum_kernel）：(numel,3) float32。

    frustum[offset] = (ix*w_interval, iy*h_interval, dbound.x + id*dbound.z)
    offset = (id*feat_h + iy)*feat_w + ix；numel = D*feat_h*feat_w
    w_interval/h_interval 取 (size-1)/(feat-1)（同参考 init）
    """
    w_interval = (image_width - 1.0) / (feat_w - 1.0)
    h_interval = (image_height - 1.0) / (feat_h - 1.0)
    ix = torch.arange(feat_w, device=device)
    iy = torch.arange(feat_h, device=device)
    id_ = torch.arange(D, device=device)
    xs = ix * w_interval
    ys = iy * h_interval
    zs = dbound[0] + id_ * dbound[2]
    frustum = torch.stack(torch.meshgrid(zs, ys, xs, indexing="ij"), -1).reshape(-1, 3).float()
    return frustum  # (D*feat_h*feat_w, 3)


def compute_geometry(frustum: torch.Tensor, camera2lidar: torch.Tensor, camera_intrinsics_inv: torch.Tensor,
                     img_aug_matrix_inv: torch.Tensor, bx, dx, nx, geometry_dim, num_camera: int):
    """frustum 反投影 + 排序 + interval 计算（对齐参考 forward）。

    camera2lidar / camera_intrinsics_inv / img_aug_matrix_inv: (6,4,4) float32
    bx/dx: (3,) float32（bev 网格起点/间隔）；nx: (3,) int64（bev 网格尺寸）
    geometry_dim: (3,) int64（coords 线性化维度，(x=360,y=360,z=80)）

    返回 (intervals (M,3) int32, indices (K,) int32, geometry_kept (K,) int32,
          keep_count int32)
    """
    geometry, ranks, keep_count = _ops.compute_geometry(
        frustum.contiguous().float(), camera2lidar.contiguous().float(),
        camera_intrinsics_inv.contiguous().float(), img_aug_matrix_inv.contiguous().float(),
        torch.as_tensor(bx, dtype=torch.float32, device=frustum.device),
        torch.as_tensor(dx, dtype=torch.float32, device=frustum.device),
        torch.as_tensor(nx, dtype=torch.int64, device=frustum.device),
        torch.as_tensor(geometry_dim, dtype=torch.int64, device=frustum.device),
        int(num_camera))

    numel = geometry.numel()
    order = torch.argsort(ranks, stable=True)          # stable sort，同 thrust::stable_sort_by_key
    ranks_sorted = ranks[order]
    kept = ranks_sorted > 0                            # rank==0 为越界点（排在最前）
    idx_kept = order[kept]
    geom_kept = geometry[idx_kept]
    ranks_kept = ranks_sorted[kept]

    if ranks_kept.numel() == 0:
        return (torch.zeros((0, 3), dtype=torch.int32, device=frustum.device),
                torch.zeros((0,), dtype=torch.int32, device=frustum.device),
                torch.zeros((0,), dtype=torch.int32, device=frustum.device),
                0)

    # interval 边界：ranks[i] != ranks[i-1]（同参考 interval_starts / collect_starts）
    grp = ranks_kept[1:] != ranks_kept[:-1]
    start = torch.cat([torch.zeros((1,), dtype=torch.int64, device=frustum.device),
                       grp.nonzero(as_tuple=False).squeeze(1) + 1])
    end = torch.cat([start[1:],
                     torch.tensor([ranks_kept.numel()], dtype=torch.int64, device=frustum.device)])
    intervals = torch.stack([start, end, geom_kept[start]], dim=1).to(torch.int32)
    return intervals, idx_kept, geom_kept, int(keep_count.item())


# ---------------------------------------------------------------- bevpool
def bevpool(camera_feature: torch.Tensor, depth_weights: torch.Tensor, intervals: torch.Tensor,
            indices: torch.Tensor, out_h: int = 360, out_w: int = 360, ndepth: int = 118,
            farea: int = 32 * 88, nchannel: int = 80) -> torch.Tensor:
    """depth_weights x camera_feature 区间求和 -> bev (1, C, out_h, out_w) half。

    camera_feature (BN, farea, C) half（HWC）；depth_weights (BN*D*farea,) half
    intervals (M,3) int32；indices (K,) int32（compute_geometry 输出）
    """
    return _ops.bevpool(camera_feature.contiguous().half(), depth_weights.contiguous().half(),
                        intervals.contiguous().to(torch.int32), indices.contiguous().to(torch.int32),
                        int(out_h), int(out_w), int(ndepth), int(farea), int(nchannel))


# ---------------------------------------------------------------- voxelize
def voxelize(points: torch.Tensor, min_range, voxel_size, grid_size,
             max_voxels: int = 60000, max_points_per_voxel: int = 10) -> tuple:
    """点云体素化（对齐参考 lidar-voxelization：均值填充 + 前 max_points 截断）。

    points (N,5) half；min_range/voxel_size (3,) float；grid_size (3,) int
    返回 (voxels (M,5) float32, coors (M,4) int32 [b,z,y,x], counts (M,) int32)
    """
    offsets, valid = _ops.voxel_offset(points.contiguous().half(),
                                       torch.as_tensor(min_range, dtype=torch.float32, device=points.device),
                                       torch.as_tensor(voxel_size, dtype=torch.float32, device=points.device),
                                       torch.as_tensor(grid_size, dtype=torch.int64, device=points.device))
    idx = valid.nonzero(as_tuple=False).squeeze(1)
    if idx.numel() == 0:
        return (torch.zeros((0, points.size(1)), dtype=torch.float32, device=points.device),
                torch.zeros((0, 4), dtype=torch.int32, device=points.device),
                torch.zeros((0,), dtype=torch.int32, device=points.device))

    off = offsets[idx]
    order = torch.argsort(off, stable=True)            # 同参考 thrust 排序
    off_s = off[order]
    grp = off_s[1:] != off_s[:-1]
    starts = torch.cat([torch.zeros((1,), dtype=torch.int64, device=points.device),
                        grp.nonzero(as_tuple=False).squeeze(1) + 1])
    ends = torch.cat([starts[1:],
                      torch.tensor([off_s.numel()], dtype=torch.int64, device=points.device)])
    counts_full = ends - starts
    kept_b = (counts_full > 0) & (starts < max_voxels)
    starts, ends, counts_full = starts[kept_b], ends[kept_b], counts_full[kept_b]
    if starts.numel() == 0:
        return (torch.zeros((0, points.size(1)), dtype=torch.float32, device=points.device),
                torch.zeros((0, 4), dtype=torch.int32, device=points.device),
                torch.zeros((0,), dtype=torch.int32, device=points.device))

    clamped = counts_full.clamp(max=max_points_per_voxel)
    num_buckets = starts.numel()
    # 桶内点展开：bucket_id + 桶内位置（截断后）
    bucket_id = torch.repeat_interleave(torch.arange(num_buckets, device=points.device), clamped)
    base = torch.repeat_interleave(starts, clamped)
    local = torch.arange(bucket_id.numel(), device=points.device) - torch.repeat_interleave(
        torch.cat([torch.zeros((1,), dtype=torch.int64, device=points.device), clamped.cumsum(0)[:-1]]), clamped)
    sel = order[base + local]                          # 原始点索引（截断后）
    sel_points = idx[sel]

    features = _ops.voxel_mean(points.contiguous().half(), sel_points.contiguous().to(torch.int32),
                               bucket_id.contiguous().to(torch.int32), num_buckets)
    counts = torch.zeros((num_buckets,), dtype=torch.int32, device=points.device)
    counts.scatter_add_(0, bucket_id, torch.ones_like(bucket_id))
    features = features / counts.clamp(min=1).float().unsqueeze(1)

    # coors [b, z, y, x]
    gx, gy, gz = int(grid_size[0]), int(grid_size[1]), int(grid_size[2])
    off_b = off_s[starts]
    x = off_b % gx
    y = (off_b // gx) % gy
    z = off_b // (gx * gy)
    coors = torch.stack([torch.zeros_like(x), z, y, x], dim=1).to(torch.int32)
    return features, coors, counts
