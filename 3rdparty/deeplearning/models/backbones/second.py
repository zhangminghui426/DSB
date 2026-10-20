"""SECOND backbone — strided-conv stages of (conv, BN, ReLU) triples.

blocks.N is a FLAT Sequential [conv,bn,relu, conv,bn,relu, ...]: conv weights
sit at indices 0,3,6... and BNs at 1,4,7... in the state_dict — keep the
flat layout (regrouping would renumber checkpoint keys).
"""

from torch import nn

__all__ = ["SECOND"]


class SECOND(nn.Module):
    def __init__(
        self,
        in_channels=128,
        out_channels=[128, 128, 256],
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        bn_eps=1e-3,
        bn_momentum=0.01,
    ):
        super().__init__()
        assert len(layer_strides) == len(layer_nums)
        assert len(out_channels) == len(layer_nums)

        in_filters = [in_channels, *out_channels[:-1]]
        # note: with stride > 1, conv2d same-padding != pad-conv2d; use padding=1
        blocks = []
        for i, layer_num in enumerate(layer_nums):
            block = [
                nn.Conv2d(
                    in_filters[i], out_channels[i], 3,
                    stride=layer_strides[i], padding=1, bias=False,
                ),
                nn.BatchNorm2d(out_channels[i], eps=bn_eps, momentum=bn_momentum),
                nn.ReLU(inplace=True),
            ]
            for _ in range(layer_num):
                block.append(
                    nn.Conv2d(out_channels[i], out_channels[i], 3, padding=1, bias=False)
                )
                block.append(
                    nn.BatchNorm2d(out_channels[i], eps=bn_eps, momentum=bn_momentum)
                )
                block.append(nn.ReLU(inplace=True))
            blocks.append(nn.Sequential(*block))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        """(N, C, H, W) -> tuple of per-stage features."""
        outs = []
        for block in self.blocks:
            x = block(x)
            outs.append(x)
        return tuple(outs)
