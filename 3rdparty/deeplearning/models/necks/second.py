"""SECONDFPN — per-level (upsample, BN, ReLU) deblocks concatenated.

deblocks.N = Sequential(conv, bn, relu): positional state_dict keys
(deblocks.N.0.weight / .1.*) — keep the flat layout.
"""

import torch
from torch import nn

__all__ = ["SECONDFPN"]


class SECONDFPN(nn.Module):
    def __init__(
        self,
        in_channels=[128, 128, 256],
        out_channels=[256, 256, 256],
        upsample_strides=[1, 2, 4],
        bn_eps=1e-3,
        bn_momentum=0.01,
        upsample_bias=False,
        use_conv_for_no_stride=False,
    ):
        super().__init__()
        assert len(out_channels) == len(upsample_strides) == len(in_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels

        deblocks = []
        for i, out_channel in enumerate(out_channels):
            stride = upsample_strides[i]
            if stride > 1 or not use_conv_for_no_stride:
                upsample_layer = nn.ConvTranspose2d(
                    in_channels[i], out_channel,
                    kernel_size=stride, stride=stride, bias=upsample_bias,
                )
            else:
                upsample_layer = nn.Conv2d(
                    in_channels[i], out_channel, kernel_size=1, stride=1, bias=False
                )
            deblocks.append(
                nn.Sequential(
                    upsample_layer,
                    nn.BatchNorm2d(out_channel, eps=bn_eps, momentum=bn_momentum),
                    nn.ReLU(inplace=True),
                )
            )
        self.deblocks = nn.ModuleList(deblocks)

    def forward(self, x):
        """x: list of (N, C_i, H, W) -> [concatenated (N, sum(C), H, W)]."""
        assert len(x) == len(self.in_channels)
        ups = [deblock(x[i]) for i, deblock in enumerate(self.deblocks)]
        out = torch.cat(ups, dim=1) if len(ups) > 1 else ups[0]
        return [out]
