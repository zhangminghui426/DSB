"""Tests for view transforms (LSS, DepthLSS)."""

import torch

from models.vtransforms.lss import LSSTransform
from models.vtransforms.depth_lss import DepthLSSTransform


def test_lss_forward():
    model = LSSTransform(
        in_channels=256,
        grid_config={
            "x": [-54.0, 54.0, 0.3],
            "y": [-54.0, 54.0, 0.3],
            "z": [-10.0, 10.0, 20.0],
            "depth": [1.0, 60.0, 0.5],
        },
        dbound=[1.0, 60.0, 0.5],
    )

    x = torch.randn(1, 6, 256, 32, 88)
    rots = torch.eye(4).repeat(1, 6, 1, 1)
    trans = torch.zeros(1, 6, 4)
    intrins = torch.eye(3).repeat(1, 6, 1, 1)
    post_rots = torch.eye(3).repeat(1, 6, 1, 1)
    post_trans = torch.zeros(1, 6, 3)

    out = model(x, rots, trans, intrins, post_rots, post_trans)
    assert out.shape[0] == 1
