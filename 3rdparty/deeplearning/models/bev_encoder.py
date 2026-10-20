"""BEV encoder: CustomResNet backbone + FPN_LSS neck.

Key naming is load-bearing for the official checkpoints:
- CustomResNet: layers.{stage}.{block}.{conv1,bn1,conv2,bn2}, and the stage-0
  downsample is a PLAIN 3x3 Conv2d WITH bias and NO BN (keys
  layers.{s}.0.downsample.{weight,bias}) — unlike torchvision's .0/.1 pair.
- FPN_LSS: conv.{0,1,3,4} and up2.{1,2,4} — Sequential slot indices ARE the
  state_dict names.
"""

import torch
import torch.utils.checkpoint as checkpoint
from torch import nn

__all__ = ["BasicBlock", "CustomResNet", "FPN_LSS"]


class BasicBlock(nn.Module):
    """mmdet-key-compatible residual block (conv1/bn1/conv2/bn2)."""

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = self.downsample(x) if self.downsample is not None else x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class CustomResNet(nn.Module):
    """3-stage strided BEV backbone: (B, C, Dy, Dx) -> [(2C, Dy/2), (4C, Dy/4), (8C, Dy/8)]."""

    def __init__(self, numC_input, num_layer=(2, 2, 2), num_channels=None,
                 stride=(2, 2, 2), backbone_output_ids=None, with_cp=False):
        super().__init__()
        assert len(num_layer) == len(stride)
        num_channels = (
            [numC_input * 2 ** (i + 1) for i in range(len(num_layer))]
            if num_channels is None else list(num_channels)
        )
        self.backbone_output_ids = (
            range(len(num_layer)) if backbone_output_ids is None else backbone_output_ids
        )

        layers = []
        curr_numC = numC_input
        for i in range(len(num_layer)):
            # first block of each stage downsamples via plain biased Conv3x3 (no BN)
            layer = [
                BasicBlock(
                    curr_numC, num_channels[i], stride=stride[i],
                    downsample=nn.Conv2d(curr_numC, num_channels[i], 3, stride[i], 1),
                )
            ]
            curr_numC = num_channels[i]
            layer.extend(
                [BasicBlock(curr_numC, num_channels[i], stride=1, downsample=None)
                 for _ in range(num_layer[i] - 1)]
            )
            layers.append(nn.Sequential(*layer))
        self.layers = nn.Sequential(*layers)
        self.with_cp = with_cp

    def forward(self, x):
        feats = []
        for lid, layer in enumerate(self.layers):
            if self.with_cp and self.training:
                x = checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
            if lid in self.backbone_output_ids:
                feats.append(x)
        return feats


class FPN_LSS(nn.Module):
    """Fuse feats[input_feature_index] (low-level) with 4x-upsampled high-level,
    then extra 2x upsample back to the BEV grid: -> (B, out_channels, 2*H, 2*W)."""

    def __init__(self, in_channels, out_channels, scale_factor=4,
                 input_feature_index=(0, 2), extra_upsample=2, lateral=None):
        super().__init__()
        self.input_feature_index = input_feature_index
        self.extra_upsample = extra_upsample is not None
        self.out_channels = out_channels
        self.up = nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=True)

        channels_factor = 2 if self.extra_upsample else 1
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * channels_factor, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * channels_factor),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels * channels_factor, out_channels * channels_factor, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * channels_factor),
            nn.ReLU(inplace=True),
        )
        if self.extra_upsample:
            self.up2 = nn.Sequential(
                nn.Upsample(scale_factor=extra_upsample, mode="bilinear", align_corners=True),
                nn.Conv2d(out_channels * channels_factor, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 1, padding=0),
            )
        self.lateral = lateral is not None
        if self.lateral:
            self.lateral_conv = nn.Sequential(
                nn.Conv2d(lateral, lateral, 1, padding=0, bias=False),
                nn.BatchNorm2d(lateral),
                nn.ReLU(inplace=True),
            )

    def forward(self, feats):
        x2, x1 = feats[self.input_feature_index[0]], feats[self.input_feature_index[1]]
        if self.lateral:
            x2 = self.lateral_conv(x2)
        x1 = self.up(x1)
        x = self.conv(torch.cat([x2, x1], dim=1))
        if self.extra_upsample:
            x = self.up2(x)
        return x
