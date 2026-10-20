"""Minimal conv blocks with named submodules (det/seg + occ heads).

Merged from the BEVFusion and FlashOCC forks;

Attribute names are state_dict keys (`<path>.conv.weight`, `<path>.bn.*`) —
kept identical to the official checkpoints, so these must NOT be replaced by
positional nn.Sequential. Init replicates the mmcv ConvModule defaults:
kaiming_normal(a=1, fan_in, relu) for conv, constant 1/0 for BN.
"""

from torch import nn

__all__ = ["ConvBNReLU", "ConvBN1d", "Conv2dBlock"]


class ConvBNReLU(nn.Module):
    """Conv2d(bias=False) + BatchNorm2d + ReLU(inplace)."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding,
            dilation, groups, bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.conv.weight, a=1, nonlinearity="relu")
        nn.init.constant_(self.bn.weight, 1.0)
        nn.init.constant_(self.bn.bias, 0.0)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ConvBN1d(nn.Module):
    """Conv1d(bias=False) + BatchNorm1d + ReLU(inplace)."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride, padding,
            dilation, groups, bias=False,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.conv.weight, a=1, nonlinearity="relu")
        nn.init.constant_(self.bn.weight, 1.0)
        nn.init.constant_(self.bn.bias, 0.0)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Conv2dBlock(nn.Module):
    """Conv2d (+ optional inplace ReLU), no norm. bias=True by default,
    matching the ConvModule(bias=True/'auto'-without-norm) call sites."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, act=False):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding,
            dilation, groups, bias=bias,
        )
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.conv(x))
