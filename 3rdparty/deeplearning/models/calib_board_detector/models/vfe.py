"""Voxel Feature Encoding (VFE)。

SECOND 的体素特征编码层:
  1. 逐点 MLP 编码原始特征 (x, y, z, intensity)
  2. 拼接体素内相对坐标
  3. 体素内 max pooling 得到局部聚合特征
  4. 再次 MLP 编码, 输出每个体素的特征向量
"""

from __future__ import annotations

import torch
import torch.nn as nn


class VFELayer(nn.Module):
    """单层 VFE: 逐点编码 + max pooling + 特征拼接。"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.norm = nn.BatchNorm1d(out_channels)

    def forward(self, voxel_features: torch.Tensor, point_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            voxel_features: (B, M, T, C) 体素内点特征。
            point_mask:     (B, M, T) 有效点掩码, 1=有效, 0=padding。

        Returns:
            point_features: (B, M, T, out_channels) 逐点编码后的特征。
        """
        B, M, T, C = voxel_features.shape
        # 逐点 MLP
        x = voxel_features.reshape(B * M * T, C)
        x = self.linear(x)
        x = self.norm(x)
        x = torch.relu(x)
        x = x.reshape(B, M, T, -1)

        # 无效点置 -inf 后做 max pooling
        mask = point_mask.unsqueeze(-1)  # (B, M, T, 1)
        x = x * mask + (~mask.bool()) * (-1e6)
        pooled = x.max(dim=2, keepdim=True)[0]  # (B, M, 1, out_ch)
        pooled = pooled.expand(-1, -1, T, -1)  # (B, M, T, out_ch)

        # 拼接逐点特征和聚合特征
        out = torch.cat([x, pooled], dim=-1)  # (B, M, T, out_ch*2)
        return out


class VoxelFeatureEncoder(nn.Module):
    """VFE 堆叠 + 最终 max pooling。"""

    def __init__(self, in_channels: int = 4, num_filters: tuple[int, ...] = (32, 128)):
        super().__init__()
        # 第一层输入: 原始特征 + 3个相对坐标通道 = in_channels + 3
        layers = []
        prev_ch = in_channels + 3
        for out_ch in num_filters:
            layers.append(VFELayer(prev_ch, out_ch))
            prev_ch = out_ch * 2  # VFELayer 输出 cat[x, pooled]
        self.layers = nn.ModuleList(layers)
        # 最后一层 VFELayer 输出 cat[point_feat, pooled_feat] = out_ch*2
        # 最终 max pooling 后维度为 out_ch*2
        self.out_channels = prev_ch

    def forward(
        self,
        voxel_features: torch.Tensor,
        voxel_coords: torch.Tensor,
        num_points: torch.Tensor,
        grid_size: tuple[int, int, int],
        voxel_size: tuple[float, float, float],
        point_cloud_range: tuple[float, float, float, float, float, float],
    ) -> torch.Tensor:
        """
        Args:
            voxel_features: (B, M, T, C) 原始点特征 [x, y, z, intensity]。
            voxel_coords:   (B, M, 3) 体素网格整数坐标 (ix, iy, iz)。
            num_points:     (B, M) 每个体素内有效点数。
            grid_size:      (Dx, Dy, Dz)。
            voxel_size:     (vx, vy, vz)。
            point_cloud_range: (xmin, ymin, zmin, xmax, ymax, zmax)。

        Returns:
            voxel_embedding: (B, M, out_channels) 每个体素的特征。
        """
        B, M, T, C = voxel_features.shape

        # 构建有效点掩码
        point_mask = (torch.arange(T, device=voxel_features.device)[None, None, :]
                      < num_points[:, :, None]).float()  # (B, M, T)

        # 计算每个点相对体素中心的偏移
        xmin, ymin, zmin = point_cloud_range[:3]
        vx, vy, vz = voxel_size
        # 体素中心坐标
        coords_f = voxel_coords.float()  # (B, M, 3)
        voxel_center_x = (coords_f[..., 0] + 0.5) * vx + xmin
        voxel_center_y = (coords_f[..., 1] + 0.5) * vy + ymin
        voxel_center_z = (coords_f[..., 2] + 0.5) * vz + zmin
        voxel_center = torch.stack([voxel_center_x, voxel_center_y, voxel_center_z], dim=-1)  # (B, M, 3)
        voxel_center = voxel_center.unsqueeze(2).expand(-1, -1, T, -1)  # (B, M, T, 3)

        # 点相对体素中心偏移
        pts_xyz = voxel_features[..., :3]
        relative_xyz = pts_xyz - voxel_center  # (B, M, T, 3)

        # 拼接到原始特征
        x = torch.cat([voxel_features, relative_xyz], dim=-1)  # (B, M, T, C+3)

        # 逐层 VFE
        for layer in self.layers:
            x = layer(x, point_mask)

        # 最终 max pooling
        mask = point_mask.unsqueeze(-1)
        x = x * mask + (~mask.bool()) * (-1e6)
        voxel_embedding = x.max(dim=2)[0]  # (B, M, out_ch)
        return voxel_embedding
