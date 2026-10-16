"""稀疏卷积 Backbone (SpMiddleResNetFPN)。

使用 spconv 进行 3D 稀疏卷积编码, 再通过反卷积多尺度上采样,
最终在 BEV 视角拼接出致密特征图。

若 spconv 不可用, 自动 fallback 到致密 3D 卷积 (scatter to dense),
保证训练流程可跑通。
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    import spconv.pytorch as spconv
    HAS_SPCONV = True
except ImportError:
    HAS_SPCONV = False


class SparseBackend(nn.Module):
    """稀疏卷积后端, 自动选择 spconv 或致密 fallback。"""

    def __init__(self, in_channels: int, grid_size: tuple[int, int, int]):
        super().__init__()
        self.grid_size = grid_size  # (Dx, Dy, Dz)
        self.in_channels = in_channels
        self.use_spconv = HAS_SPCONV

    def features_to_dense(
        self,
        voxel_features: torch.Tensor,
        voxel_coords: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """将稀疏体素特征 scatter 到致密 3D 网格。

        Args:
            voxel_features: (B, M, C) 体素特征。
            voxel_coords:   (B, M, 3) 体素坐标 [ix, iy, iz]。
            batch_size:      batch 大小。

        Returns:
            dense: (B, C, Dz, Dy, Dx) 致密张量。
        """
        Dx, Dy, Dz = self.grid_size
        B, M, C = voxel_features.shape
        dense = torch.zeros(B, C, Dz, Dy, Dx, device=voxel_features.device, dtype=voxel_features.dtype)
        for b in range(batch_size):
            ix = voxel_coords[b, :, 0].long()
            iy = voxel_coords[b, :, 1].long()
            iz = voxel_coords[b, :, 2].long()
            feat = voxel_features[b]  # (M, C)
            dense[b, :, iz, iy, ix] = feat.t()
        return dense

    def sparse_conv_forward(
        self,
        x: torch.Tensor,
        module: nn.Module,
        indices_key: str = "indices",
    ) -> torch.Tensor:
        """统一前向接口 (当前由子类实现具体逻辑)。"""
        return module(x)


class SpMiddleFPN(nn.Module):
    """SECOND 稀疏中间层 + FPN。

    结构:
      - 3 个稀疏卷积下采样 (stride=2)
      - 3 个稀疏反卷积上采样 (stride=2)
      - 在 BEV 维度 (z) 压缩, 拼接多尺度特征
    """

    def __init__(
        self,
        in_channels: int = 128,
        sparse_filters: tuple[int, ...] = (64, 128, 256),
        upsample_filters: tuple[int, ...] = (128, 128, 128),
        grid_size: tuple[int, int, int] = (200, 200, 25),
    ):
        super().__init__()
        self.grid_size = grid_size  # (Dx, Dy, Dz)
        self.in_channels = in_channels
        # spconv 反向传播在纯 CPU 环境下不可用, 仅在 CUDA 可用时使用稀疏卷积
        self.use_spconv = HAS_SPCONV and torch.cuda.is_available()

        # 下采样层
        if self.use_spconv:
            self.down_convs = nn.ModuleList()
            prev_ch = in_channels
            for out_ch in sparse_filters:
                self.down_convs.append(
                    spconv.SparseSequential(
                        spconv.SparseConv3d(prev_ch, out_ch, kernel_size=3,
                                            stride=2, padding=1, indice_key=f"sp{out_ch}"),
                        nn.BatchNorm1d(out_ch),
                        nn.ReLU(),
                    )
                )
                prev_ch = out_ch

            # 上采样层 (反卷积), stride 由 indice_key 关联的前向卷积决定
            self.up_convs = nn.ModuleList()
            for i, out_ch in enumerate(upsample_filters):
                in_ch_deconv = sparse_filters[-(i + 1)]
                self.up_convs.append(
                    spconv.SparseSequential(
                        spconv.SparseInverseConv3d(in_ch_deconv, out_ch, kernel_size=3,
                                                   indice_key=f"sp{sparse_filters[-(i+1)]}"),
                        nn.BatchNorm1d(out_ch),
                        nn.ReLU(),
                    )
                )
        else:
            # Fallback: 致密 3D 卷积
            self.down_convs = nn.ModuleList()
            prev_ch = in_channels
            for out_ch in sparse_filters:
                self.down_convs.append(nn.Sequential(
                    nn.Conv3d(prev_ch, out_ch, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm3d(out_ch),
                    nn.ReLU(),
                ))
                prev_ch = out_ch

            self.up_convs = nn.ModuleList()
            for i, out_ch in enumerate(upsample_filters):
                in_ch_deconv = sparse_filters[-(i + 1)]
                self.up_convs.append(nn.Sequential(
                    nn.ConvTranspose3d(in_ch_deconv, out_ch, kernel_size=3,
                                        stride=2, padding=1, output_padding=1),
                    nn.BatchNorm3d(out_ch),
                    nn.ReLU(),
                ))

        self.out_channels = sum(upsample_filters)  # 384

    def forward(
        self,
        voxel_embedding: torch.Tensor,
        voxel_coords: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """
        Args:
            voxel_embedding: (B, M, C) 体素特征。
            voxel_coords:   (B, M, 3) [ix, iy, iz]。
            batch_size:      batch。

        Returns:
            bev_feature: (B, C_bev, Dy, Dx) BEV 特征图。
        """
        Dx, Dy, Dz = self.grid_size

        if self.use_spconv:
            return self._forward_spconv(voxel_embedding, voxel_coords, batch_size, Dx, Dy, Dz)
        else:
            return self._forward_dense(voxel_embedding, voxel_coords, batch_size, Dx, Dy, Dz)

    def _forward_spconv(self, voxel_embedding, voxel_coords, batch_size, Dx, Dy, Dz):
        B, M, C = voxel_embedding.shape
        # 构建 spconv indices: (N, 4) = [batch, iz, iy, ix]
        all_indices = []
        all_features = []
        for b in range(B):
            n = voxel_coords[b].shape[0]
            batch_idx = torch.full((n, 1), b, device=voxel_embedding.device, dtype=torch.int32)
            coords = voxel_coords[b].int()  # (n, 3) [ix, iy, iz]
            # 转置为 [iz, iy, ix]
            sp_coords = torch.stack([coords[:, 2], coords[:, 1], coords[:, 0]], dim=1)
            indices = torch.cat([batch_idx, sp_coords], dim=1)
            all_indices.append(indices)
            all_features.append(voxel_embedding[b])
        indices = torch.cat(all_indices, dim=0).int()
        features = torch.cat(all_features, dim=0)

        # spconv spatial_shape 顺序: (Dz, Dy, Dx)
        x = spconv.SparseConvTensor(features, indices, (Dz, Dy, Dx), batch_size)

        # 下采样, 保存各层输出
        downs = []
        for conv in self.down_convs:
            x = conv(x)
            downs.append(x)

        # 上采样, 从最深层开始
        # 第一层反卷积用最深的 down 输出
        x = self.up_convs[0](downs[-1])
        ups = [x.dense()]
        for i in range(1, len(self.up_convs)):
            # 残差连接: 上采样结果 + 对应下采样层 (简化: 直接上采样)
            x = self.up_convs[i](downs[-(i + 1)])
            ups.append(x.dense())

        # ups[0] 对应 1/4 分辨率, ups[1] 对应 1/2, ups[2] 对应 1/1
        # 统一到原始分辨率 (Dz, Dy, Dx) 后在 z 维度压缩
        bev_list = []
        for up in ups:
            # 插值到原始尺寸
            up = nn.functional.interpolate(
                up, size=(Dz, Dy, Dx), mode="trilinear", align_corners=False
            )
            # z 维度压缩为 BEV
            bev = up.sum(dim=2)  # (B, C, Dy, Dx)
            bev_list.append(bev)
        return torch.cat(bev_list, dim=1)

    def _forward_dense(self, voxel_embedding, voxel_coords, batch_size, Dx, Dy, Dz):
        """致密 fallback, 用 index_put_ 替代 Python 循环 scatter。"""
        B, M, C = voxel_embedding.shape
        device = voxel_embedding.device

        # 坐标
        ix = voxel_coords[..., 0].long()  # (B, M)
        iy = voxel_coords[..., 1].long()
        iz = voxel_coords[..., 2].long()

        # 构建 5D 索引: (b, c, iz, iy, ix), 形状 (B, C, M)
        # 用广播而不是显式 expand, 节省内存
        b_idx = torch.arange(B, device=device).view(B, 1, 1)
        c_idx = torch.arange(C, device=device).view(1, C, 1)
        iz_exp = iz.view(B, 1, M)
        iy_exp = iy.view(B, 1, M)
        ix_exp = ix.view(B, 1, M)

        # 体素特征转成 (B, C, M) 以匹配索引
        values = voxel_embedding.permute(0, 2, 1)  # (B, C, M)

        # scatter 到致密网格
        dense = torch.zeros(B, C, Dz, Dy, Dx, device=device, dtype=voxel_embedding.dtype)
        dense.index_put_((b_idx, c_idx, iz_exp, iy_exp, ix_exp), values, accumulate=False)

        x = dense
        downs = []
        for conv in self.down_convs:
            x = conv(x)
            downs.append(x)

        ups = []
        x = self.up_convs[0](downs[-1])
        ups.append(x)
        for i in range(1, len(self.up_convs)):
            x = self.up_convs[i](downs[-(i + 1)])
            ups.append(x)

        bev_list = []
        for up in ups:
            up = nn.functional.interpolate(
                up, size=(Dz, Dy, Dx), mode="trilinear", align_corners=False
            )
            bev = up.sum(dim=2)
            bev_list.append(bev)
        return torch.cat(bev_list, dim=1)
