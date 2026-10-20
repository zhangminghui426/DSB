"""体素化 (Voxelization) 工具。

将点云划分为固定尺寸的体素，每个体素内保留最多 max_points 个点。
输出:
  voxel_features: (M, T, C) 每个体素内 T 个点的特征 [x,y,z,intensity]
  voxel_coords:   (M, 3) 体素在网格中的整数坐标 (ix, iy, iz)
  num_points:     (M,) 每个体素内实际点数
"""

from __future__ import annotations

import numpy as np


def points_to_voxels(
    points: np.ndarray,
    voxel_size: tuple[float, float, float],
    point_cloud_range: tuple[float, float, float, float, float, float],
    max_points_per_voxel: int = 10,
    max_num_voxels: int = 12000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """将点云转换为体素。

    Args:
        points: (N, C) 点云, C>=3, 前三维为 xyz, 第四维可选为强度/反射率。
        voxel_size: (vx, vy, vz) 体素尺寸(米)。
        point_cloud_range: (xmin, ymin, zmin, xmax, ymax, zmax)。
        max_points_per_voxel: 每个体素最多保留的点数。
        max_num_voxels: 最多生成的体素数(超出截断)。

    Returns:
        voxel_features: (M, T, C) 体素内点特征, 不足 T 的补 0。
        voxel_coords:   (M, 3) 体素网格坐标 (ix, iy, iz)。
        num_points:     (M,) 每个体素内实际点数。
    """
    x_min, y_min, z_min, x_max, y_max, z_max = point_cloud_range
    vx, vy, vz = voxel_size

    # 1. 范围过滤
    mask = (
        (points[:, 0] >= x_min)
        & (points[:, 0] < x_max)
        & (points[:, 1] >= y_min)
        & (points[:, 1] < y_max)
        & (points[:, 2] >= z_min)
        & (points[:, 2] < z_max)
    )
    pts = points[mask]
    if pts.shape[0] == 0:
        M = 0
        C = points.shape[1]
        return (
            np.zeros((M, max_points_per_voxel, C), dtype=np.float32),
            np.zeros((M, 3), dtype=np.int32),
            np.zeros((M,), dtype=np.int32),
        )

    # 2. 计算每个点的体素索引
    ix = np.floor((pts[:, 0] - x_min) / vx).astype(np.int32)
    iy = np.floor((pts[:, 1] - y_min) / vy).astype(np.int32)
    iz = np.floor((pts[:, 2] - z_min) / vz).astype(np.int32)

    grid_dx = int(round((x_max - x_min) / vx))
    grid_dy = int(round((y_max - y_min) / vy))

    # 用线性编码将 (ix,iy,iz) 映射为唯一 key
    voxel_keys = ix * grid_dy * grid_dx + iy * grid_dx + iz

    # 3. 按体素 key 分组
    order = np.argsort(voxel_keys)
    sorted_keys = voxel_keys[order]
    sorted_pts = pts[order]

    # 找到每个新体素的起始位置
    diff = np.diff(sorted_keys)
    split_idx = np.where(diff != 0)[0] + 1
    starts = np.concatenate([[0], split_idx])
    ends = np.concatenate([split_idx, [len(sorted_keys)]])

    num_voxels = len(starts)
    C = pts.shape[1]

    # 4. 构建输出
    voxel_features = np.zeros((num_voxels, max_points_per_voxel, C), dtype=np.float32)
    voxel_coords = np.zeros((num_voxels, 3), dtype=np.int32)
    num_points = np.zeros((num_voxels,), dtype=np.int32)

    for i in range(num_voxels):
        s, e = starts[i], ends[i]
        n = min(e - s, max_points_per_voxel)
        voxel_features[i, :n] = sorted_pts[s : s + n]
        num_points[i] = n
        # 从第一个点恢复体素坐标
        p0 = sorted_pts[s]
        voxel_coords[i, 0] = int(np.floor((p0[0] - x_min) / vx))
        voxel_coords[i, 1] = int(np.floor((p0[1] - y_min) / vy))
        voxel_coords[i, 2] = int(np.floor((p0[2] - z_min) / vz))

    # 5. 截断到 max_num_voxels
    if num_voxels > max_num_voxels:
        voxel_features = voxel_features[:max_num_voxels]
        voxel_coords = voxel_coords[:max_num_voxels]
        num_points = num_points[:max_num_voxels]

    return voxel_features, voxel_coords, num_points
