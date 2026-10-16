"""Voxelization ops — point cloud voxelization.

Pure PyTorch fallback for CPU; CUDA kernel version optional.
"""

import torch
import numpy as np


def points_to_voxel(points, voxel_size, coors_range, max_points=35, max_voxels=20000):
    """Convert points to voxels (pure PyTorch/numpy version).

    Args:
        points: (N, 4) — x, y, z, intensity
        voxel_size: (3,) — dx, dy, dz
        coors_range: (6,) — x_min, y_min, z_min, x_max, y_max, z_max
        max_points: max points per voxel
        max_voxels: max number of voxels

    Returns:
        voxels: (M, max_points, 4)
        coordinates: (M, 3) — (z, y, x) voxel indices
        num_points_per_voxel: (M,)
    """
    if isinstance(points, torch.Tensor):
        points = points.numpy()

    voxel_size = np.array(voxel_size)
    coors_range = np.array(coors_range)

    # Filter points outside range
    mask = (
        (points[:, 0] >= coors_range[0])
        & (points[:, 0] < coors_range[3])
        & (points[:, 1] >= coors_range[1])
        & (points[:, 1] < coors_range[4])
        & (points[:, 2] >= coors_range[2])
        & (points[:, 2] < coors_range[5])
    )
    points = points[mask]

    # Compute voxel indices
    voxel_idx = np.floor(
        (points[:, :3] - coors_range[:3]) / voxel_size
    ).astype(np.int32)

    # Unique voxels
    voxel_keys = voxel_idx[:, 0] * 100000 + voxel_idx[:, 1] * 1000 + voxel_idx[:, 2]
    unique_keys, inverse = np.unique(voxel_keys, return_inverse=True)

    num_voxels = min(len(unique_keys), max_voxels)

    # Allocate arrays
    voxels = np.zeros((num_voxels, max_points, points.shape[1]), dtype=np.float32)
    coors = np.zeros((num_voxels, 3), dtype=np.int32)
    num_points = np.zeros(num_voxels, dtype=np.int32)

    # Fill voxels
    for i in range(len(points)):
        v = inverse[i]
        if v >= num_voxels:
            continue
        if num_points[v] >= max_points:
            continue
        voxels[v, num_points[v]] = points[i]
        num_points[v] += 1

    # Fill coordinates
    for v in range(num_voxels):
        idx = np.where(inverse == v)[0][0]
        coors[v] = voxel_idx[idx][[2, 1, 0]]  # (z, y, x)

    return (
        torch.from_numpy(voxels),
        torch.from_numpy(coors),
        torch.from_numpy(num_points),
    )
