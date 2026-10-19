"""Deploy-time preprocessing (mirrors BaseDepthTransform depth projection).

project_depth renders the sparse lidar-depth map per camera that feeds the
DepthLSSTransform dtransform at runtime. Pure torch, ONNX-free — the depth
projection stays on the torch/CUDA side of the deployment split.
"""

import torch

__all__ = ["project_depth"]


def project_depth(points, lidar2image, img_aug_matrix, lidar_aug_matrix,
                  image_size=(256, 704)):
    """Project lidar points onto each camera plane -> (Ncam, 1, H, W) depth.

    Same math as models/vtransforms/base.py BaseDepthTransform.forward
    (single frame, batch=1): inverse lidar aug -> lidar2image -> perspective
    divide -> img aug -> grid-sample coordinates.

    Args:
        points: (N, >=3) lidar points, first 3 columns = xyz.
        lidar2image: (Ncam, 4, 4) lidar->camera-image matrices.
        img_aug_matrix: (Ncam, 4, 4) image augmentation matrices.
        lidar_aug_matrix: (4, 4) lidar augmentation matrix.
        image_size: (H, W) output depth resolution (256, 704).

    Returns:
        (Ncam, 1, H, W) float32 sparse depth map (0 = no lidar hit).
    """
    ncam = lidar2image.shape[0]
    H, W = image_size
    depth = torch.zeros(ncam, 1, H, W, dtype=torch.float32, device=points.device)

    cur_coords = points[:, :3].float().clone()
    cur_img_aug_matrix = img_aug_matrix.float()
    cur_lidar_aug_matrix = lidar_aug_matrix.float()
    cur_lidar2image = lidar2image.float()

    # inverse lidar aug
    cur_coords = cur_coords - cur_lidar_aug_matrix[:3, 3]
    cur_coords = torch.inverse(cur_lidar_aug_matrix[:3, :3]).matmul(
        cur_coords.transpose(1, 0)
    )  # (3, N)

    # lidar2image per camera: (Ncam, 3, 3) @ (3, N) -> (Ncam, 3, N)
    cur_coords = cur_lidar2image[:, :3, :3].matmul(
        cur_coords.unsqueeze(0).expand(ncam, -1, -1)
    )
    cur_coords = cur_coords + cur_lidar2image[:, :3, 3].reshape(-1, 3, 1)

    dist = cur_coords[:, 2, :]  # (Ncam, N)
    cur_coords = cur_coords.clone()
    cur_coords[:, 2, :] = torch.clamp(cur_coords[:, 2, :], 1e-5, 1e5)
    cur_coords[:, :2, :] = cur_coords[:, :2, :] / cur_coords[:, 2:3, :]

    # image aug
    cur_coords = cur_img_aug_matrix[:, :3, :3].matmul(cur_coords)
    cur_coords = cur_coords + cur_img_aug_matrix[:, :3, 3].reshape(-1, 3, 1)
    cur_coords = cur_coords[:, :2, :].transpose(1, 2)  # (Ncam, N, 2)
    cur_coords = cur_coords[..., [1, 0]]  # grid-sample (y, x) order

    on_img = (
        (cur_coords[..., 0] < H)
        & (cur_coords[..., 0] >= 0)
        & (cur_coords[..., 1] < W)
        & (cur_coords[..., 1] >= 0)
    )
    for c in range(ncam):
        masked_coords = cur_coords[c, on_img[c]].long()
        masked_dist = dist[c, on_img[c]]
        depth[c, 0, masked_coords[:, 0], masked_coords[:, 1]] = masked_dist
    return depth
