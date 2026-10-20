"""CustomFPN — 1x1 lateral convs, top-down nearest-upsample-add, 3x3 outputs.

FlashOCC r50 usage: in_channels [1024, 2048], out_channels 256, num_outs 1,
out_ids [0] -> state_dict keys lateral_convs.{0,1}.conv.{weight,bias} +
fpn_convs.0.conv.{weight,bias} (conv with bias, no norm, no activation).

Init replicates the live mmengine BaseModule init_cfg: Xavier-uniform on
Conv2d weights, zeros on biases.
"""

import torch.nn.functional as F
from torch import nn

from models.layers import Conv2dBlock

__all__ = ["CustomFPN"]


class CustomFPN(nn.Module):
    def __init__(self, in_channels, out_channels, num_outs, start_level=0,
                 end_level=-1, out_ids=(0,), upsample_cfg=dict(mode="nearest")):
        super().__init__()
        assert isinstance(in_channels, (list, tuple))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.upsample_cfg = dict(upsample_cfg)
        self.out_ids = list(out_ids)
        self.backbone_end_level = self.num_ins if end_level == -1 else end_level
        self.start_level = start_level

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            self.lateral_convs.append(
                Conv2dBlock(in_channels[i], out_channels, 1, bias=True)
            )
            if i in self.out_ids:
                self.fpn_convs.append(
                    Conv2dBlock(out_channels, out_channels, 3, padding=1, bias=True)
                )
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        for i in range(len(laterals) - 1, 0, -1):
            if "scale_factor" in self.upsample_cfg:
                laterals[i - 1] = laterals[i - 1] + F.interpolate(laterals[i], **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg
                )
        return [self.fpn_convs[k](laterals[i]) for k, i in enumerate(self.out_ids)]
