"""BEVOCCHead2D — FlashOCC channel-to-height 2D occupancy head.

Keys are load-bearing: final_conv.conv.{weight,bias} (Conv2dBlock keeps the
mmcv ConvModule attribute name) and predicter.{0,2} (Linear-Softplus-Linear).

forward folds channels to z: (B,C,Dy,Dx) -> final_conv -> permute(0,3,2,1)
-> predicter -> view(B, Dx, Dy, Dz, num_classes) with channel index z*18+cls.
"""

import numpy as np
import torch
from torch import nn

from models.layers import Conv2dBlock
from models.losses import create_loss

__all__ = ["BEVOCCHead2D"]

# nuScenes occupancy class frequencies (for the optional class-balanced CE)
nusc_class_frequencies = np.array([
    944004, 1897170, 152386, 2391677, 16957802, 724139, 189027, 2074468,
    413451, 2384460, 5916653, 175883646, 4275424, 51393615, 61411620,
    105975596, 116424404, 1892500630,
])


class BEVOCCHead2D(nn.Module):
    def __init__(self, in_dim=256, out_dim=256, Dz=16, use_mask=True,
                 num_classes=18, use_predicter=True, class_balance=False,
                 loss_occ=None):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.Dz = Dz
        out_channels = out_dim if use_predicter else num_classes * Dz
        self.final_conv = Conv2dBlock(in_dim, out_channels, 3, stride=1,
                                      padding=1, bias=True, act=True)
        self.use_predicter = use_predicter
        if use_predicter:
            self.predicter = nn.Sequential(
                nn.Linear(out_dim, out_dim * 2),
                nn.Softplus(),
                nn.Linear(out_dim * 2, num_classes * Dz),
            )
        self.use_mask = use_mask
        self.num_classes = num_classes

        self.class_balance = class_balance
        if class_balance:
            class_weights = torch.from_numpy(
                1 / np.log(nusc_class_frequencies[:num_classes] + 0.001)
            )
            self.cls_weights = class_weights
            loss_occ = dict(loss_occ)
            loss_occ["class_weight"] = class_weights.tolist()
        self.loss_occ = create_loss(loss_occ)

    def forward(self, img_feats):
        """(B, C, Dy, Dx) -> (B, Dx, Dy, Dz, num_classes)"""
        occ_pred = self.final_conv(img_feats).permute(0, 3, 2, 1)
        bs, Dx, Dy = occ_pred.shape[:3]
        if self.use_predicter:
            occ_pred = self.predicter(occ_pred)
            occ_pred = occ_pred.view(bs, Dx, Dy, self.Dz, self.num_classes)
        return occ_pred

    def loss(self, occ_pred, voxel_semantics, mask_camera):
        """Camera-mask-weighted CE; avg_factor = mask_camera.sum()."""
        loss = dict()
        voxel_semantics = voxel_semantics.long()
        if self.use_mask:
            mask_camera = mask_camera.to(torch.int32)
            voxel_semantics = voxel_semantics.reshape(-1)
            preds = occ_pred.reshape(-1, self.num_classes)
            mask_camera = mask_camera.reshape(-1)

            if self.class_balance:
                valid_voxels = voxel_semantics[mask_camera.bool()]
                num_total_samples = 0
                for i in range(self.num_classes):
                    num_total_samples += (valid_voxels == i).sum() * self.cls_weights[i]
            else:
                num_total_samples = mask_camera.sum()

            loss["loss_occ"] = self.loss_occ(
                preds, voxel_semantics, mask_camera, avg_factor=num_total_samples
            )
        else:
            voxel_semantics = voxel_semantics.reshape(-1)
            preds = occ_pred.reshape(-1, self.num_classes)
            if self.class_balance:
                num_total_samples = 0
                for i in range(self.num_classes):
                    num_total_samples += (voxel_semantics == i).sum() * self.cls_weights[i]
            else:
                num_total_samples = len(voxel_semantics)
            loss["loss_occ"] = self.loss_occ(preds, voxel_semantics, avg_factor=num_total_samples)
        return loss

    def get_occ(self, occ_pred, img_metas=None):
        """(B, Dx, Dy, Dz, C) -> list of (Dx, Dy, Dz) uint8 argmax grids."""
        occ_score = occ_pred.softmax(-1)
        occ_res = occ_score.argmax(-1)
        return list(occ_res.cpu().numpy().astype(np.uint8))
