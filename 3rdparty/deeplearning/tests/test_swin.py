"""Tests for Swin Transformer backbone."""

import torch

from models.backbones.swin import SwinTransformer


def test_swin_forward():
    model = SwinTransformer(
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        out_indices=(1, 2, 3),
    )

    x = torch.randn(1, 3, 256, 704)
    outs = model(x)

    assert len(outs) == 3
    assert outs[0].shape[0] == 1
    assert outs[1].shape[0] == 1
    assert outs[2].shape[0] == 1
