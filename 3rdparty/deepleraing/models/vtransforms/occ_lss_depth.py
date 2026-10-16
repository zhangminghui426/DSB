"""Depth-supervised LSS view transforms (BEVDepth / BEVStereo).

LSSViewTransformerBEVDepth: replaces the plain depth_net with the camera-aware
DepthNet; adds the 27-dim mlp_input, GT depth discretization (UD or SID) and
the BCE depth loss (fp32, matching the old @force_fp32 semantics).
LSSViewTransformerBEVStereo: adds the downsample-4 cost-volume frustum.
"""

import torch
import torch.nn.functional as F
from torch.amp import autocast

from models.vtransforms.depthnet import DepthNet
from models.vtransforms.occ_lss import LSSViewTransformer

__all__ = ["LSSViewTransformerBEVDepth", "LSSViewTransformerBEVStereo"]


class LSSViewTransformerBEVDepth(LSSViewTransformer):
    def __init__(self, loss_depth_weight=3.0, depthnet_cfg=dict(), **kwargs):
        super().__init__(**kwargs)
        self.loss_depth_weight = loss_depth_weight
        self.depth_net = DepthNet(
            in_channels=self.in_channels,
            mid_channels=self.in_channels,
            context_channels=self.out_channels,
            depth_channels=self.D,
            **dict(depthnet_cfg),
        )

    def get_mlp_input(self, sensor2ego, ego2global, intrin, post_rot, post_tran, bda):
        """Camera-aware 27-dim input: intrinsics + aug + bda + sensor2ego.

        sensor2ego: (B, N, 4, 4); intrin: (B, N, 3, 3); post_rot: (B, N, 3, 3);
        post_tran: (B, N, 3); bda: (B, 3, 3) -> (B, N, 27)
        """
        B, N, _, _ = sensor2ego.shape
        bda = bda.view(B, 1, 3, 3).repeat(1, N, 1, 1)
        mlp_input = torch.stack([
            intrin[:, :, 0, 0],
            intrin[:, :, 1, 1],
            intrin[:, :, 0, 2],
            intrin[:, :, 1, 2],
            post_rot[:, :, 0, 0],
            post_rot[:, :, 0, 1],
            post_tran[:, :, 0],
            post_rot[:, :, 1, 0],
            post_rot[:, :, 1, 1],
            post_tran[:, :, 1],
            bda[:, :, 0, 0],
            bda[:, :, 0, 1],
            bda[:, :, 1, 0],
            bda[:, :, 1, 1],
            bda[:, :, 2, 2],
        ], dim=-1)
        sensor2ego = sensor2ego[:, :, :3, :].reshape(B, N, -1)
        return torch.cat([mlp_input, sensor2ego], dim=-1)

    def forward(self, input, stereo_metas=None):
        """input: [imgs, sensor2egos, ego2globals, intrins, post_rots,
        post_trans, bda, mlp_input] -> (bev_feat, depth)."""
        (x, rots, trans, intrins, post_rots, post_trans, bda, mlp_input) = input[:8]

        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        x = self.depth_net(x, mlp_input, stereo_metas)
        depth_digit = x[:, :self.D, ...]
        tran_feat = x[:, self.D:self.D + self.out_channels, ...]
        depth = depth_digit.softmax(dim=1)
        return self.view_transform(input, depth, tran_feat)

    def get_downsampled_gt_depth(self, gt_depths):
        """(B, N, img_h, img_w) metric depth -> (B*N*fH*fW, D) one-hot bins."""
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(
            B * N, H // self.downsample, self.downsample,
            W // self.downsample, self.downsample, 1,
        )
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(-1, self.downsample * self.downsample)
        gt_depths_tmp = torch.where(
            gt_depths == 0.0, 1e5 * torch.ones_like(gt_depths), gt_depths
        )
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // self.downsample, W // self.downsample)

        if not self.sid:
            # (d - (min_dist - interval)) / interval == (d - min)/interval + 1
            gt_depths = (
                gt_depths - (self.grid_config["depth"][0] - self.grid_config["depth"][2])
            ) / self.grid_config["depth"][2]
        else:
            gt_depths = torch.log(gt_depths) - torch.log(
                torch.tensor(self.grid_config["depth"][0]).float()
            )
            gt_depths = gt_depths * (self.D - 1) / torch.log(
                torch.tensor(self.grid_config["depth"][1] - 1.0).float()
                / self.grid_config["depth"][0]
            )
            gt_depths = gt_depths + 1.0

        gt_depths = torch.where(
            (gt_depths < self.D + 1) & (gt_depths >= 0.0),
            gt_depths, torch.zeros_like(gt_depths),
        )
        gt_depths = F.one_hot(gt_depths.long(), num_classes=self.D + 1).view(
            -1, self.D + 1
        )[:, 1:]
        return gt_depths.float()

    def get_depth_loss(self, depth_labels, depth_preds):
        """BCE on softmaxed depth over foreground pixels (fp32; the original
        ran under @force_fp32), normalized by max(1, #fg)."""
        depth_labels = self.get_downsampled_gt_depth(depth_labels)
        depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(-1, self.D)
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0
        depth_labels = depth_labels[fg_mask]
        depth_preds = depth_preds[fg_mask]
        with autocast("cuda", enabled=False):
            depth_loss = F.binary_cross_entropy(
                depth_preds.float(), depth_labels.float(), reduction="none"
            ).sum() / max(1.0, fg_mask.sum())
        return self.loss_depth_weight * depth_loss


class LSSViewTransformerBEVStereo(LSSViewTransformerBEVDepth):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # (D, fH_stereo, fW_stereo, 3) frustum at downsample 4 for the cost volume
        self.cv_frustum = self.create_frustum(
            kwargs["grid_config"]["depth"], kwargs["input_size"], downsample=4
        )
