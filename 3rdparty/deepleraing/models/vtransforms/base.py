"""LSS view transforms (base geometry + BEV pooling).

Ported from mmdet3d/models/vtransforms/base.py. Changes vs the fork:
- mmcv force_fp32/auto_fp16 decorators replaced by explicit
  `torch.autocast(enabled=False)` + float casts in the geometry paths
  (frustum projection / bev_pool are fp32-sensitive and the CUDA kernel is
  fp32-only).
- The fork's radar-aware knobs (use_points / depth_input / height_expand /
  add_depth_features / mats_dict threading) are dropped: they were
  inconsistent with LSSTransform/DepthLSSTransform signatures and with the
  official checkpoints (dtransform expects a single-channel scalar depth
  map). Upstream BEVFusion semantics are restored.
- Parameter/buffer names kept identical to the original (dx, bx, nx,
  frustum) for checkpoint compatibility.
"""

from typing import Tuple

import torch
from torch import nn

from ops import bev_pool_v2_sum

__all__ = ["BaseTransform", "BaseDepthTransform"]


def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor(
        [(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]]
    )
    return dx, bx, nx


class BaseTransform(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        zbound: Tuple[float, float, float],
        dbound: Tuple[float, float, float],
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.image_size = image_size
        self.feature_size = feature_size
        self.xbound = xbound
        self.ybound = ybound
        self.zbound = zbound
        self.dbound = dbound

        dx, bx, nx = gen_dx_bx(self.xbound, self.ybound, self.zbound)
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.C = out_channels
        self.frustum = self.create_frustum()
        self.D = self.frustum.shape[0]
        self.fp16_enabled = False

    def create_frustum(self):
        iH, iW = self.image_size
        fH, fW = self.feature_size

        ds = (
            torch.arange(*self.dbound, dtype=torch.float)
            .view(-1, 1, 1)
            .expand(-1, fH, fW)
        )
        D, _, _ = ds.shape

        xs = (
            torch.linspace(0, iW - 1, fW, dtype=torch.float)
            .view(1, 1, fW)
            .expand(D, fH, fW)
        )
        ys = (
            torch.linspace(0, iH - 1, fH, dtype=torch.float)
            .view(1, fH, 1)
            .expand(D, fH, fW)
        )

        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(
        self,
        camera2lidar_rots,
        camera2lidar_trans,
        intrins,
        post_rots,
        post_trans,
        **kwargs,
    ):
        # fp32 geometry (inverse/matmul precision matters here)
        device = camera2lidar_trans.device
        # ONNX has no matrix-inverse op: callers may pass precomputed
        # post_rots_inv / intrins_inv (deploy/export path)
        post_rots_inv = kwargs.get("post_rots_inv", None)
        intrins_inv = kwargs.get("intrins_inv", None)
        with torch.autocast(device_type=device.type, enabled=False):
            camera2lidar_rots = camera2lidar_rots.float()
            camera2lidar_trans = camera2lidar_trans.float()
            intrins = intrins.float()
            post_rots = post_rots.float()
            post_trans = post_trans.float()

            B, N, _ = camera2lidar_trans.shape

            # undo post-transformation
            # B x N x D x H x W x 3
            points = self.frustum.float() - post_trans.view(B, N, 1, 1, 1, 3)
            if post_rots_inv is None:
                post_rots_inv = torch.inverse(post_rots)
            points = (
                post_rots_inv.float()
                .view(B, N, 1, 1, 1, 3, 3)
                .matmul(points.unsqueeze(-1))
            )
            # cam_to_lidar
            points = torch.cat(
                (
                    points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                    points[:, :, :, :, :, 2:3],
                ),
                5,
            )
            if intrins_inv is None:
                intrins_inv = torch.inverse(intrins)
            combine = camera2lidar_rots.matmul(intrins_inv.float())
            points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
            points += camera2lidar_trans.view(B, N, 1, 1, 1, 3)

            if "extra_rots" in kwargs:
                extra_rots = kwargs["extra_rots"]
                points = (
                    extra_rots.view(B, 1, 1, 1, 1, 3, 3)
                    .repeat(1, N, 1, 1, 1, 1, 1)
                    .matmul(points.unsqueeze(-1))
                    .squeeze(-1)
                )
            if "extra_trans" in kwargs:
                extra_trans = kwargs["extra_trans"]
                points += extra_trans.view(B, 1, 1, 1, 1, 3).repeat(1, N, 1, 1, 1, 1)

            return points

    def get_cam_feats(self, x):
        raise NotImplementedError

    def bev_pool(self, geom_feats, x):
        if getattr(self, "onnx_export", False):
            return self.bev_pool_onnx(geom_feats, x)
        device = x.device
        with torch.autocast(device_type=device.type, enabled=False):
            B, N, D, H, W, C = x.shape
            Nprime = B * N * D * H * W

            x = x.float().reshape(Nprime, C)

            # flatten indices
            geom_feats = ((geom_feats.float() - (self.bx - self.dx / 2.0)) / self.dx).long()
            geom_feats = geom_feats.view(Nprime, 3)
            batch_ix = torch.cat(
                [
                    torch.full(
                        [Nprime // B, 1], ix, device=x.device, dtype=torch.long
                    )
                    for ix in range(B)
                ]
            )
            geom_feats = torch.cat((geom_feats, batch_ix), 1)

            # filter out points that are outside box
            kept = (
                (geom_feats[:, 0] >= 0)
                & (geom_feats[:, 0] < self.nx[0])
                & (geom_feats[:, 1] >= 0)
                & (geom_feats[:, 1] < self.nx[1])
                & (geom_feats[:, 2] >= 0)
                & (geom_feats[:, 2] < self.nx[2])
            )
            x = x[kept]
            geom_feats = geom_feats[kept]

            x = bev_pool_v2_sum(x, geom_feats, B, self.nx[2], self.nx[0], self.nx[1])

            # collapse Z
            final = torch.cat(x.unbind(dim=2), 1)

            return final

    def bev_pool_onnx(self, geom_feats, x):
        """ONNX-export-friendly bev_pool (same semantics as the CUDA op).

        sort by rank -> cumsum -> per-segment difference -> scatter into
        (B, D, H, W, C). Uses only ONNX-native ops (sort/cumsum/nonzero/gather/
        index_add).
        """
        B, N, Df, Hf, Wf, C = x.shape
        Nprime = B * N * Df * Hf * Wf
        # output BEV grid dims (nx = [X, Y, Z]); D/H/W below follow the
        # CUDA wrapper call bev_pool(feats, coords, B, nx[2], nx[0], nx[1])
        D, H, W = int(self.nx[2]), int(self.nx[0]), int(self.nx[1])

        x = x.float().reshape(Nprime, C)

        geom_feats = ((geom_feats.float() - (self.bx - self.dx / 2.0)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat(
            [torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long) for ix in range(B)]
        )
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        kept = (
            (geom_feats[:, 0] >= 0)
            & (geom_feats[:, 0] < H)
            & (geom_feats[:, 1] >= 0)
            & (geom_feats[:, 1] < W)
            & (geom_feats[:, 2] >= 0)
            & (geom_feats[:, 2] < D)
        )
        x = x[kept]
        geom_feats = geom_feats[kept]
        n = x.shape[0]

        # scatter-add into the (B, D, H, W, C) grid; addition is commutative so
        # ONNX export path: emit the bf.plugins::BevPool custom op (TensorRT
        # plugin); eager forward runs the exact scatter-add reference
        from ops.bev_pool_v2 import bev_pool_plugin_fn

        out = bev_pool_plugin_fn(x, geom_feats, B, D, H, W)
        # CUDA bev_pool returns (B, C, D, H, W)
        out = out.permute(0, 4, 1, 2, 3)

        final = torch.cat(out.unbind(dim=2), 1)
        return final

    def forward(
        self,
        img,
        points,
        radar,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas=None,
        **kwargs,
    ):
        # NOTE: the fork's BaseTransform.forward lacked the `metas` parameter
        # that extract_camera_features passes positionally (BaseDepthTransform
        # has it) — a latent breakage of the LSSTransform/seg path; restored
        # here per upstream BEVFusion call convention.
        rots = camera2ego[..., :3, :3]
        trans = camera2ego[..., :3, 3]
        intrins = camera_intrinsics[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        lidar2ego_rots = lidar2ego[..., :3, :3]
        lidar2ego_trans = lidar2ego[..., :3, 3]
        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]

        extra_rots = lidar_aug_matrix[..., :3, :3]
        extra_trans = lidar_aug_matrix[..., :3, 3]

        geom = self.get_geometry(
            camera2lidar_rots,
            camera2lidar_trans,
            intrins,
            post_rots,
            post_trans,
            extra_rots=extra_rots,
            extra_trans=extra_trans,
        )
        x = self.get_cam_feats(img)

        use_depth = False
        if type(x) == tuple:
            x, depth = x
            use_depth = True

        x = self.bev_pool(geom, x)

        if use_depth:
            return x, depth
        else:
            return x


class BaseDepthTransform(BaseTransform):
    """Adds a sparse lidar-depth map (1 channel per camera) as auxiliary
    input to the depth network (upstream BEVFusion DepthLSSTransform
    semantics; radar-aware fork extensions removed)."""

    def forward(
        self,
        img,
        points,
        radar,
        sensor2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        cam_intrinsic,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas=None,
        **kwargs,
    ):
        rots = sensor2ego[..., :3, :3]
        trans = sensor2ego[..., :3, 3]
        intrins = cam_intrinsic[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        lidar2ego_rots = lidar2ego[..., :3, :3]
        lidar2ego_trans = lidar2ego[..., :3, 3]
        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]

        batch_size = len(points)
        # scalar lidar-depth projection, fp32 geometry
        with torch.autocast(device_type=img.device.type, enabled=False):
            depth = torch.zeros(
                batch_size,
                img.shape[1],
                1,
                *self.image_size,
                device=points[0].device,
            )

            for b in range(batch_size):
                cur_coords = points[b][:, :3].float().clone()
                cur_img_aug_matrix = img_aug_matrix[b].float()
                cur_lidar_aug_matrix = lidar_aug_matrix[b].float()
                cur_lidar2image = lidar2image[b].float()

                # inverse aug
                cur_coords -= cur_lidar_aug_matrix[:3, 3]
                cur_coords = torch.inverse(cur_lidar_aug_matrix[:3, :3]).matmul(
                    cur_coords.transpose(1, 0)
                )
                # lidar2image
                cur_coords = cur_lidar2image[:, :3, :3].matmul(cur_coords)
                cur_coords += cur_lidar2image[:, :3, 3].reshape(-1, 3, 1)
                # get 2d coords
                dist = cur_coords[:, 2, :]
                cur_coords[:, 2, :] = torch.clamp(cur_coords[:, 2, :], 1e-5, 1e5)
                cur_coords[:, :2, :] /= cur_coords[:, 2:3, :]

                # imgaug
                cur_coords = cur_img_aug_matrix[:, :3, :3].matmul(cur_coords)
                cur_coords += cur_img_aug_matrix[:, :3, 3].reshape(-1, 3, 1)
                cur_coords = cur_coords[:, :2, :].transpose(1, 2)

                # normalize coords for grid sample
                cur_coords = cur_coords[..., [1, 0]]

                on_img = (
                    (cur_coords[..., 0] < self.image_size[0])
                    & (cur_coords[..., 0] >= 0)
                    & (cur_coords[..., 1] < self.image_size[1])
                    & (cur_coords[..., 1] >= 0)
                )
                for c in range(on_img.shape[0]):
                    masked_coords = cur_coords[c, on_img[c]].long()
                    masked_dist = dist[c, on_img[c]]
                    depth[b, c, 0, masked_coords[:, 0], masked_coords[:, 1]] = (
                        masked_dist
                    )

        extra_rots = lidar_aug_matrix[..., :3, :3]
        extra_trans = lidar_aug_matrix[..., :3, 3]
        geom = self.get_geometry(
            camera2lidar_rots,
            camera2lidar_trans,
            intrins,
            post_rots,
            post_trans,
            extra_rots=extra_rots,
            extra_trans=extra_trans,
        )
        x = self.get_cam_feats(img, depth)

        use_depth = False
        if type(x) == tuple:
            x, depth = x
            use_depth = True

        x = self.bev_pool(geom, x)

        if use_depth:
            return x, depth
        else:
            return x
