"""Tests for Sparse Encoder."""

import torch

from models.backbones.sparse_encoder import SparseEncoder


def test_sparse_encoder_forward():
    encoder = SparseEncoder(
        in_channels=4,
        sparse_shape=[1440, 1440, 41],
        output_channels=256,
    )

    # Create dummy sparse input
    coors = torch.tensor([[0, 100, 200, 10], [0, 300, 400, 20]], dtype=torch.int32)
    features = torch.randn(2, 4)

    # Forward pass
    out = encoder(features, coors)

    assert out.shape[0] == 1
    assert out.shape[1] == 256
