"""DepthNet — camera-aware depth & context prediction for BEVDepth/Stereo.

Parameter naming matches the original (reduce_conv/context_conv/bn/depth_mlp/
context_mlp/depth_se/context_se/cost_volumn_net/depth_conv with mmdet-style
BasicBlock keys). ASPP does constructor-time kaiming init (live behavior in
the original). use_dcn=True is NOT ported (the r50-4d-stereo config sets
use_dcn=False).
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch.amp import autocast
from torch.utils.checkpoint import checkpoint

from models.bev_encoder import BasicBlock

__all__ = ["DepthNet"]


class ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation):
        super().__init__()
        self.atrous_conv = nn.Conv2d(
            inplanes, planes, kernel_size=kernel_size, stride=1,
            padding=padding, dilation=dilation, bias=False,
        )
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        return self.relu(self.bn(self.atrous_conv(x)))


class ASPP(nn.Module):
    def __init__(self, inplanes, mid_channels=256):
        super().__init__()
        dilations = [1, 6, 12, 18]
        self.aspp1 = ASPPModule(inplanes, mid_channels, 1, padding=0, dilation=dilations[0])
        self.aspp2 = ASPPModule(inplanes, mid_channels, 3, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = ASPPModule(inplanes, mid_channels, 3, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = ASPPModule(inplanes, mid_channels, 3, padding=dilations[3], dilation=dilations[3])
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, mid_channels, 1, stride=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
        )
        self.conv1 = nn.Conv2d(int(mid_channels * 5), inplanes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode="bilinear", align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return self.dropout(x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.ReLU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class SELayer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_reduce = nn.Conv2d(channels, channels, 1, bias=True)
        self.act1 = nn.ReLU()
        self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = nn.Sigmoid()

    def forward(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


class DepthNet(nn.Module):
    """SE-gated depth/context branches over a 27-dim camera-aware MLP input;
    stereo adds a cost-volume branch (group-wise grid_sample warping)."""

    def __init__(self, in_channels, mid_channels, context_channels, depth_channels,
                 use_dcn=False, use_aspp=True, with_cp=False, stereo=False,
                 bias=0.0, aspp_mid_channels=-1):
        super().__init__()
        if use_dcn:
            raise NotImplementedError("use_dcn=True is not ported (configs use use_dcn=False)")
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.context_conv = nn.Conv2d(mid_channels, context_channels, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm1d(27)
        self.depth_mlp = Mlp(in_features=27, hidden_features=mid_channels, out_features=mid_channels)
        self.depth_se = SELayer(channels=mid_channels)
        self.context_mlp = Mlp(in_features=27, hidden_features=mid_channels, out_features=mid_channels)
        self.context_se = SELayer(channels=mid_channels)

        depth_conv_input_channels = mid_channels
        downsample = None
        if stereo:
            depth_conv_input_channels += depth_channels
            downsample = nn.Conv2d(depth_conv_input_channels, mid_channels, 1, 1, 0)
            cost_volumn_net = []
            for _ in range(2):
                cost_volumn_net.extend([
                    nn.Conv2d(depth_channels, depth_channels, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(depth_channels),
                ])
            self.cost_volumn_net = nn.Sequential(*cost_volumn_net)
            self.bias = bias

        depth_conv_list = [
            BasicBlock(depth_conv_input_channels, mid_channels, downsample=downsample),
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
        ]
        if use_aspp:
            if aspp_mid_channels < 0:
                aspp_mid_channels = mid_channels
            depth_conv_list.append(ASPP(mid_channels, aspp_mid_channels))
        depth_conv_list.append(
            nn.Conv2d(mid_channels, depth_channels, kernel_size=1, stride=1, padding=0)
        )
        self.depth_conv = nn.Sequential(*depth_conv_list)
        self.with_cp = with_cp
        self.depth_channels = depth_channels

    def gen_grid(self, metas, B, N, D, H, W, hi, wi):
        """Warp grid: current pixels -> prev-camera pixels (normalized)."""
        frustum = metas["frustum"]
        # undo image augmentation
        points = frustum - metas["post_trans"].view(B, N, 1, 1, 1, 3)
        points = torch.inverse(metas["post_rots"]).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))
        # (u, v, d) -> (du, dv, d)
        points = torch.cat((points[..., :2, :] * points[..., 2:3, :], points[..., 2:3, :]), 5)
        # cur_pixel -> curr_camera -> prev_camera
        rots = metas["k2s_sensor"][:, :, :3, :3].contiguous()
        trans = metas["k2s_sensor"][:, :, :3, 3].contiguous()
        combine = rots.matmul(torch.inverse(metas["intrins"]))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points)
        points += trans.view(B, N, 1, 1, 1, 3, 1)

        neg_mask = points[..., 2, 0] < 1e-3
        # prev_camera -> prev_pixel
        points = metas["intrins"].view(B, N, 1, 1, 1, 3, 3).matmul(points)
        points = points[..., :2, :] / points[..., 2:3, :]
        # redo image augmentation
        points = metas["post_rots"][..., :2, :2].view(B, N, 1, 1, 1, 2, 2).matmul(points).squeeze(-1)
        points += metas["post_trans"][..., :2].view(B, N, 1, 1, 1, 2)

        px = points[..., 0] / (wi - 1.0) * 2.0 - 1.0
        py = points[..., 1] / (hi - 1.0) * 2.0 - 1.0
        px[neg_mask] = -2
        py[neg_mask] = -2
        grid = torch.stack([px, py], dim=-1)
        return grid.view(B * N, D * H, W, 2)

    def calculate_cost_volumn(self, metas):
        """Group-wise |curr - warp(prev)| matching cost -> softmax over depth.

        Forced to fp32 (autocast off): the warp-grid coordinate chain
        (intrinsics @ metric frustum coords) reaches ~1e5 for near-zero-depth
        points and overflows fp16 to inf, NaN-ing the grid_sample. The
        original plugin had this latent AMP bug on the stereo path.
        """
        with autocast("cuda", enabled=False):
            prev, curr = metas["cv_feat_list"]
            prev = prev.float()
            curr = curr.float()
            group_size = 4
            _, c, hf, wf = curr.shape
            hi, wi = hf * 4, wf * 4
            B, N, _ = metas["post_trans"].shape
            D, H, W, _ = metas["frustum"].shape
            grid = self.gen_grid(metas, B, N, D, H, W, hi, wi).float()

            prev = prev.view(B * N, -1, H, W)
            curr = curr.view(B * N, -1, H, W)
            cost_volumn = 0
            for fid in range(curr.shape[1] // group_size):
                prev_curr = prev[:, fid * group_size:(fid + 1) * group_size, ...]
                wrap_prev = F.grid_sample(prev_curr, grid, align_corners=True, padding_mode="zeros")
                curr_tmp = curr[:, fid * group_size:(fid + 1) * group_size, ...]
                cost_volumn_tmp = curr_tmp.unsqueeze(2) - wrap_prev.view(B * N, -1, D, H, W)
                cost_volumn += cost_volumn_tmp.abs().sum(dim=1)
            if not self.bias == 0:
                invalid = wrap_prev[:, 0, ...].view(B * N, D, H, W) == 0
                cost_volumn[invalid] = cost_volumn[invalid] + self.bias
            return (-cost_volumn).softmax(dim=1)

    def forward(self, x, mlp_input, stereo_metas=None):
        """x: (B*N_views, C, fH, fW), mlp_input: (B, N_views, 27)
        -> (B*N_views, D+C_context, fH, fW)"""
        mlp_input = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1]))
        x = self.reduce_conv(x)

        context_se = self.context_mlp(mlp_input)[..., None, None]
        context = self.context_se(x, context_se)
        context = self.context_conv(context)

        depth_se = self.depth_mlp(mlp_input)[..., None, None]
        depth = self.depth_se(x, depth_se)

        if stereo_metas is not None:
            if stereo_metas["cv_feat_list"][0] is None:
                BN, _, H, W = x.shape
                scale_factor = float(stereo_metas["downsample"]) / stereo_metas["cv_downsample"]
                cost_volumn = torch.zeros(
                    (BN, self.depth_channels, int(H * scale_factor), int(W * scale_factor))
                ).to(x)
            else:
                with torch.no_grad():
                    cost_volumn = self.calculate_cost_volumn(stereo_metas)
            cost_volumn = self.cost_volumn_net(cost_volumn)
            depth = torch.cat([depth, cost_volumn], dim=1)

        if self.with_cp:
            depth = checkpoint(self.depth_conv, depth, use_reentrant=False)
        else:
            depth = self.depth_conv(depth)
        return torch.cat([depth, context], dim=1)
