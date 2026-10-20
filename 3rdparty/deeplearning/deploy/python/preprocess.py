"""数据预处理层（Deploy-time Preprocessing）。

把原始输入（图像 / 点云 / 内外参矩阵）转换成模型输入张量：
lidar 稀疏深度投影、点云硬体素化。与 C++ 侧 ``deploy/cpp/preprocess.*``
逐层对应（Preprocessor 类 + project_depth / voxelize）。
"""

import torch

__all__ = ["Preprocessor", "project_depth"]


def project_depth(points, lidar2image, img_aug_matrix, lidar_aug_matrix,
                  image_size=(256, 704)):
    """把 lidar 点投影到每相机平面 -> (Ncam, 1, H, W) 稀疏深度图。

    与 ``models/vtransforms/base.py BaseDepthTransform.forward`` 同数学
    （单帧, batch=1）：逆 lidar 增强 -> lidar2image -> 透视除深度 ->
    img 增强 -> grid-sample 坐标。

    Args:
        points: (N, >=3) lidar 点，前三列 = xyz。
        lidar2image: (Ncam, 4, 4) lidar -> 相机图像 矩阵。
        img_aug_matrix: (Ncam, 4, 4) 图像增强矩阵。
        lidar_aug_matrix: (4, 4) lidar 增强矩阵。
        image_size: (H, W) 输出深度分辨率 (256, 704)。

    Returns:
        (Ncam, 1, H, W) float32 稀疏深度图（0 = 无 lidar 命中）。
    """
    ncam = lidar2image.shape[0]
    H, W = image_size
    depth = torch.zeros(ncam, 1, H, W, dtype=torch.float32, device=points.device)

    cur_coords = points[:, :3].float().clone()
    cur_img_aug_matrix = img_aug_matrix.float()
    cur_lidar_aug_matrix = lidar_aug_matrix.float()
    cur_lidar2image = lidar2image.float()

    # 逆 lidar 增强
    cur_coords = cur_coords - cur_lidar_aug_matrix[:3, 3]
    cur_coords = torch.inverse(cur_lidar_aug_matrix[:3, :3]).matmul(
        cur_coords.transpose(1, 0)
    )  # (3, N)

    # 每相机 lidar2image：(Ncam, 3, 3) @ (3, N) -> (Ncam, 3, N)
    cur_coords = cur_lidar2image[:, :3, :3].matmul(
        cur_coords.unsqueeze(0).expand(ncam, -1, -1)
    )
    cur_coords = cur_coords + cur_lidar2image[:, :3, 3].reshape(-1, 3, 1)

    dist = cur_coords[:, 2, :]  # (Ncam, N)
    cur_coords = cur_coords.clone()
    cur_coords[:, 2, :] = torch.clamp(cur_coords[:, 2, :], 1e-5, 1e5)
    cur_coords[:, :2, :] = cur_coords[:, :2, :] / cur_coords[:, 2:3, :]

    # 图像增强
    cur_coords = cur_img_aug_matrix[:, :3, :3].matmul(cur_coords)
    cur_coords = cur_coords + cur_img_aug_matrix[:, :3, 3].reshape(-1, 3, 1)
    cur_coords = cur_coords[:, :2, :].transpose(1, 2)  # (Ncam, N, 2)
    cur_coords = cur_coords[..., [1, 0]]  # grid-sample (y, x) 顺序

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


class Preprocessor:
    """数据预处理层：collated batch -> 模型输入张量。

    职责边界：只负责把原始样本变成模型输入（深度投影 / 体素化），
    不含模型前向与检测解码。
    """

    def __init__(self, model, image_size=(256, 704)):
        self.voxelize_fn = model.encoders["lidar"]["voxelize"]
        self.image_size = image_size

    def voxelize(self, points):
        """点云硬体素化 -> (feats, coords, sizes)。

        输出留在 points 所在设备；coords 为 (M, 3)（x, y, z），
        batch 维由推理层补齐。sizes 为 None 表示体素化器不返回计数。
        """
        ret = self.voxelize_fn(points.float())
        if len(ret) == 3:
            feats, coords, sizes = ret
        else:
            feats, coords = ret
            sizes = None
        return feats, coords, sizes

    def __call__(self, batch):
        """预处理一个 collated batch（B=1），产出模型输入 dict。

        Returns:
            dict:
                img:    (1,6,3,256,704) 归一化图像。
                depth:  (1,6,1,256,704) lidar 稀疏深度图。
                vfeats: (M,5) 体素内点特征。
                vcoords: (M,3) 体素坐标（x,y,z）。
                vsizes: (M,) 每体素点数（或 None）。
        """
        img = batch["img"]
        depth = project_depth(
            batch["points"][0].cpu(),
            batch["lidar2image"][0].cpu(),
            batch["img_aug_matrix"][0].cpu().float(),
            batch["lidar_aug_matrix"][0].cpu().float(),
            self.image_size,
        ).unsqueeze(0).to(img.device)
        vfeats, vcoords, vsizes = self.voxelize(batch["points"][0])
        return {
            "img": img,
            "depth": depth,
            "vfeats": vfeats,
            "vcoords": vcoords,
            "vsizes": vsizes,
        }
