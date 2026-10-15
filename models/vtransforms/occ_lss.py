"""LSSViewTransformer — Lift-Splat-Shoot view transform with BEVPoolv2.

Ported 1:1 (numerics) from the FlashOCC plugin; mmengine BaseModule/registry
removed. frustum/grid_* are plain tensor attributes (NOT buffers) — they never
appear in the state_dict and are moved with .to(device) at use.

Key numeric details preserved:
- frustum (D, fH, fW, 3) order (u, v, d); d = arange(*depth_cfg), optional SID
- get_ego_coor: (frustum - post_trans) -> inv(post_rots) -> *(u*d, v*d, d)
  -> sensor2ego[:3,:3] @ inv(cam2imgs) -> + trans -> bda LAST
- ranks use inclusive-end grid semantics of the original torch.range calls
- bev_pool_v2 kernel is fp32-only: explicit .float() casts keep AMP safe
- collapse_z: cat(unbind(dim=2), 1) — z-major channel folding
"""

import torch
from torch import nn

from ops import bev_pool_v2

__all__ = ["LSSViewTransformer"]


class LSSViewTransformer(nn.Module):
    def __init__(self, grid_config, input_size, downsample=16, in_channels=512,
                 out_channels=64, sid=False, collapse_z=True):
        super().__init__()
        self.grid_config = grid_config
        self.downsample = downsample
        self.create_grid_infos(**grid_config)
        self.sid = sid
        self.frustum = self.create_frustum(grid_config["depth"], input_size, downsample)
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.depth_net = nn.Conv2d(in_channels, self.D + out_channels, kernel_size=1, padding=0)
        self.collapse_z = collapse_z

    def create_grid_infos(self, x, y, z, **kwargs):
        """grid_lower_bound / grid_interval / grid_size as plain tensors."""
        self.grid_lower_bound = torch.Tensor([cfg[0] for cfg in [x, y, z]])
        self.grid_interval = torch.Tensor([cfg[2] for cfg in [x, y, z]])
        self.grid_size = torch.Tensor([(cfg[1] - cfg[0]) / cfg[2] for cfg in [x, y, z]])

    def create_frustum(self, depth_cfg, input_size, downsample):
        """(D, fH, fW, 3) frustum template, channels ordered (u, v, d)."""
        H_in, W_in = input_size
        H_feat, W_feat = H_in // downsample, W_in // downsample
        d = torch.arange(*depth_cfg, dtype=torch.float).view(-1, 1, 1).expand(-1, H_feat, W_feat)
        self.D = d.shape[0]
        if self.sid:
            # Spacing-Increasing Discretization (STS)
            d_sid = torch.arange(self.D).float()
            depth_cfg_t = torch.tensor(depth_cfg).float()
            d_sid = torch.exp(
                torch.log(depth_cfg_t[0])
                + d_sid / (self.D - 1) * torch.log((depth_cfg_t[1] - 1) / depth_cfg_t[0])
            )
            d = d_sid.view(-1, 1, 1).expand(-1, H_feat, W_feat)

        x = torch.linspace(0, W_in - 1, W_feat, dtype=torch.float).view(1, 1, W_feat).expand(self.D, H_feat, W_feat)
        y = torch.linspace(0, H_in - 1, H_feat, dtype=torch.float).view(1, H_feat, 1).expand(self.D, H_feat, W_feat)
        return torch.stack((x, y, d), -1)

    def get_ego_coor(self, sensor2ego, ego2global, cam2imgs, post_rots, post_trans, bda):
        """Frustum points -> ego frame: (B, N, D, fH, fW, 3)."""
        B, N, _, _ = sensor2ego.shape

        points = self.frustum.to(sensor2ego) - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))

        points = torch.cat((points[..., :2, :] * points[..., 2:3, :], points[..., 2:3, :]), 5)
        combine = sensor2ego[:, :, :3, :3].matmul(torch.inverse(cam2imgs))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += sensor2ego[:, :, :3, 3].view(B, N, 1, 1, 1, 3)
        # bda applied last
        points = bda.view(B, 1, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1)).squeeze(-1)
        return points

    def voxel_pooling_prepare_v2(self, coor):
        """coor (B,N,D,fH,fW,3) -> sorted ranks + pillar intervals."""
        B, N, D, H, W, _ = coor.shape
        num_points = B * N * D * H * W
        # arange reproduces the original torch.range(0, n-1) inclusive values
        ranks_depth = torch.arange(0, num_points, dtype=torch.int, device=coor.device)
        ranks_feat = torch.arange(0, num_points // D, dtype=torch.int, device=coor.device)
        ranks_feat = ranks_feat.reshape(B, N, 1, H, W).expand(B, N, D, H, W).flatten()

        coor = (coor - self.grid_lower_bound.to(coor)) / self.grid_interval.to(coor)
        coor = coor.long().view(num_points, 3)
        batch_idx = (
            torch.arange(0, B, device=coor.device)
            .reshape(B, 1).expand(B, num_points // B).reshape(num_points, 1).to(coor)
        )
        coor = torch.cat((coor, batch_idx), 1)

        kept = (
            (coor[:, 0] >= 0) & (coor[:, 0] < self.grid_size[0])
            & (coor[:, 1] >= 0) & (coor[:, 1] < self.grid_size[1])
            & (coor[:, 2] >= 0) & (coor[:, 2] < self.grid_size[2])
        )
        if len(kept) == 0:
            return None, None, None, None, None

        coor, ranks_depth, ranks_feat = coor[kept], ranks_depth[kept], ranks_feat[kept]

        ranks_bev = coor[:, 3] * (self.grid_size[2] * self.grid_size[1] * self.grid_size[0])
        ranks_bev += coor[:, 2] * (self.grid_size[1] * self.grid_size[0])
        ranks_bev += coor[:, 1] * self.grid_size[0] + coor[:, 0]
        order = ranks_bev.argsort()
        ranks_bev, ranks_depth, ranks_feat = ranks_bev[order], ranks_depth[order], ranks_feat[order]

        kept = torch.ones(ranks_bev.shape[0], device=ranks_bev.device, dtype=torch.bool)
        kept[1:] = ranks_bev[1:] != ranks_bev[:-1]
        interval_starts = torch.where(kept)[0].int()
        if len(interval_starts) == 0:
            return None, None, None, None, None
        interval_lengths = torch.zeros_like(interval_starts)
        interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
        interval_lengths[-1] = ranks_bev.shape[0] - interval_starts[-1]
        return (
            ranks_bev.int().contiguous(),
            ranks_depth.int().contiguous(),
            ranks_feat.int().contiguous(),
            interval_starts.int().contiguous(),
            interval_lengths.int().contiguous(),
        )

    def voxel_pooling_v2(self, coor, depth, feat):
        """-> (B, C*Dz, Dy, Dx) BEV feature."""
        ranks_bev, ranks_depth, ranks_feat, interval_starts, interval_lengths = (
            self.voxel_pooling_prepare_v2(coor)
        )
        if ranks_feat is None:
            print("warning ---> no points within the predefined bev receptive field")
            dummy = torch.zeros(
                size=[feat.shape[0], feat.shape[2],
                      int(self.grid_size[2]), int(self.grid_size[1]), int(self.grid_size[0])]
            ).to(feat)
            return torch.cat(dummy.unbind(dim=2), 1)

        feat = feat.permute(0, 1, 3, 4, 2)  # (B, N, fH, fW, C)
        depth = depth.float()  # bev_pool_v2 kernel is fp32-only (AMP-safe)
        feat = feat.float()
        bev_feat_shape = (
            depth.shape[0], int(self.grid_size[2]), int(self.grid_size[1]),
            int(self.grid_size[0]), feat.shape[-1],
        )
        bev_feat = bev_pool_v2(
            depth, feat, ranks_depth, ranks_feat, ranks_bev,
            bev_feat_shape, interval_starts, interval_lengths,
        )  # (B, C, Dz, Dy, Dx)
        if self.collapse_z:
            bev_feat = torch.cat(bev_feat.unbind(dim=2), 1)  # z-major folding
        return bev_feat

    def view_transform(self, input, depth, tran_feat):
        """depth (B*N, D, fH, fW) + tran_feat (B*N, C, fH, fW) ->
        (bev_feat (B, C*Dz, Dy, Dx), depth)."""
        B, N, _, H, W = input[0].shape
        coor = self.get_ego_coor(*input[1:7])
        bev_feat = self.voxel_pooling_v2(
            coor, depth.view(B, N, self.D, H, W),
            tran_feat.view(B, N, self.out_channels, H, W),
        )
        return bev_feat, depth

    def forward(self, input):
        """input: [imgs, sensor2egos, ego2globals, intrins, post_rots,
        post_trans, bda] -> (bev_feat (B, C*Dz, Dy, Dx), depth (B*N, D, fH, fW)).
        """
        x = input[0]
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)

        x = self.depth_net(x)
        depth_digit = x[:, :self.D, ...]
        tran_feat = x[:, self.D:self.D + self.out_channels, ...]
        depth = depth_digit.softmax(dim=1)
        return self.view_transform(input, depth, tran_feat)
