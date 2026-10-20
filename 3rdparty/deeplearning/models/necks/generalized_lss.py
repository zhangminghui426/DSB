"""GeneralizedLSSFPN — top-down FPN concatenating upsampled higher-level
features with the current level (LSS-style).

Submodule names (lateral_convs/fpn_convs of ConvBNReLU: .conv/.bn) are
state_dict keys — keep them.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers import ConvBNReLU

__all__ = ["GeneralizedLSSFPN"]


class GeneralizedLSSFPN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_outs,
        start_level=0,
        end_level=-1,
        upsample_cfg=dict(mode="bilinear", align_corners=True),
    ):
        super().__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.upsample_cfg = dict(upsample_cfg)

        if end_level == -1:
            self.backbone_end_level = self.num_ins - 1
        else:
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            self.lateral_convs.append(
                ConvBNReLU(
                    in_channels[i]
                    + (
                        in_channels[i + 1]
                        if i == self.backbone_end_level - 1
                        else out_channels
                    ),
                    out_channels,
                    1,
                )
            )
            self.fpn_convs.append(
                ConvBNReLU(out_channels, out_channels, 3, padding=1)
            )

    def forward(self, inputs):
        """upsample -> cat -> conv1x1 -> conv3x3, top-down."""
        assert len(inputs) == len(self.in_channels)
        laterals = [inputs[i + self.start_level] for i in range(len(inputs))]

        used_backbone_levels = len(laterals) - 1
        for i in range(used_backbone_levels - 1, -1, -1):
            x = F.interpolate(
                laterals[i + 1],
                size=laterals[i].shape[2:],
                **self.upsample_cfg,
            )
            laterals[i] = torch.cat([laterals[i], x], dim=1)
            laterals[i] = self.lateral_convs[i](laterals[i])
            laterals[i] = self.fpn_convs[i](laterals[i])

        return tuple(laterals[i] for i in range(used_backbone_levels))
