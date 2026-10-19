"""Lidar branch at deploy time: voxelization + sparse encoder (torch/spconv).

Mirrors BevPerception.extract_features (hard voxelization -> mean reduce ->
SparseEncoder dense BEV). Sparse ops are not part of the exported ONNX
graphs; they run on the torch/spconv runtime side of the deployment split.
"""

import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["LidarBranchEncoder"]


class LidarBranchEncoder(nn.Module):
    """Voxelize + SparseEncoder -> (B, C*D, H, W) lidar BEV feature."""

    def __init__(self, model):
        super().__init__()
        self.voxelize = model.encoders["lidar"]["voxelize"]
        self.backbone = model.encoders["lidar"]["backbone"]
        self.voxelize_reduce = getattr(model, "voxelize_reduce", True)

    def forward(self, points_list, use_fp16=False):
        """points_list: list of (N_i, >=3) point tensors -> (B, C*D, H, W).

        Output lands on the same device as the input points.
        """
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points_list):
            ret = self.voxelize(res.float())
            if len(ret) == 3:
                f, c, n = ret  # hard voxelize
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode="constant", value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(-1, 1)
                feats = feats.contiguous()
        batch_size = coords[-1, 0] + 1
        return self.backbone(feats, coords, batch_size, sizes=sizes)
