"""Region Proposal Network (RPN) 检测头。

在 BEV 特征图上输出分类和回归预测:
  - cls: (B, num_anchor_per_loc * num_class, Dy, Dx)
  - loc: (B, num_anchor_per_loc * 7, Dy, Dx)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RPN(nn.Module):
    def __init__(
        self,
        in_channels: int = 384,
        num_anchor_per_loc: int = 2,
        num_class: int = 1,
    ):
        super().__init__()
        self.num_anchor_per_loc = num_anchor_per_loc
        self.num_class = num_class

        # 共享卷积层
        self.shared = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        # 分类分支
        self.cls_head = nn.Conv2d(128, num_anchor_per_loc * num_class, kernel_size=1)
        # 回归分支 (7: x, y, z, w, l, h, yaw)
        self.loc_head = nn.Conv2d(128, num_anchor_per_loc * 7, kernel_size=1)

    def forward(self, bev_feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            bev_feature: (B, C, Dy, Dx)。

        Returns:
            cls_pred: (B, num_anchor_per_loc * num_class, Dy, Dx)。
            loc_pred: (B, num_anchor_per_loc * 7, Dy, Dx)。
        """
        x = self.shared(bev_feature)
        cls_pred = self.cls_head(x)
        loc_pred = self.loc_head(x)
        return cls_pred, loc_pred
