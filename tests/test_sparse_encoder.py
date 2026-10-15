"""SparseEncoder (spconv 2.x) verification: checkpoint key alignment +
weight-layout remap + forward shape. Run from repo root."""

import sys

import os.path as osp; sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), ".."))

import torch

from models.backbones.sparse_encoder import SparseEncoder

DET_CFG = dict(
    in_channels=5,
    sparse_shape=[1440, 1440, 41],
    output_channels=128,
    encoder_channels=[[16, 16, 32], [32, 32, 64], [64, 64, 128], [128, 128]],
    encoder_paddings=[[0, 0, 1], [0, 0, 1], [0, 0, [1, 1, 0]], [0, 0]],
    block_type="basicblock",
)


def remap_lidar_weights(state, prefix="encoders.lidar.backbone."):
    """spconv 1.x (K,D,H,Cin,Cout) -> spconv 2.x (Cout,K,D,H,Cin)."""
    out = {}
    for k, v in state.items():
        if not k.startswith(prefix):
            continue
        k2 = k[len(prefix) :]
        if v.ndim == 5:
            v = v.permute(4, 0, 1, 2, 3).contiguous()
        out[k2] = v
    return out


def main():
    ckpt = torch.load("ckpts/bevfusion-det.pth", map_location="cpu")["state_dict"]
    ref = remap_lidar_weights(ckpt)
    print("ckpt lidar keys:", len(ref))

    m = SparseEncoder(**DET_CFG)
    mine = dict(m.state_dict())
    print("model keys:", len(mine))

    missing = sorted(set(ref) - set(mine))
    unexpected = sorted(set(mine) - set(ref))
    mismatch = {k for k in set(ref) & set(mine) if tuple(ref[k].shape) != tuple(mine[k].shape)}
    assert not missing, f"missing: {missing[:10]}"
    assert not unexpected, f"unexpected: {unexpected[:10]}"
    assert not mismatch, f"shape mismatch: {sorted(mismatch)[:10]}"
    msg = m.load_state_dict(ref, strict=True)
    assert not msg.missing_keys and not msg.unexpected_keys
    print("[OK] checkpoint keys: 0 missing / 0 unexpected / 0 shape-mismatch (after 5-dim permute)")

    # forward smoke on GPU with synthetic voxels
    dev = "cuda:0"
    m = m.to(dev).eval()
    N = 30000
    feats = torch.randn(N, 5, device=dev)
    coords = torch.stack(
        [
            torch.zeros(N, dtype=torch.int, device=dev),  # batch
            torch.randint(0, 1440, (N,), device=dev),  # x
            torch.randint(0, 1440, (N,), device=dev),  # y
            torch.randint(0, 40, (N,), device=dev),  # z
        ],
        dim=1,
    ).int()
    coords = torch.unique(coords, dim=0)
    feats = feats[: coords.shape[0]]

    with torch.no_grad():
        out = m(feats, coords, 1)
    print("forward out:", tuple(out.shape))
    assert out.shape == (1, 256, 180, 180), out.shape
    assert torch.isfinite(out).all()

    # fp16 autocast forward (training uses AMP)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        out16 = m(feats, coords, 1)
    assert torch.isfinite(out16.float()).all()
    print("[OK] fp16 autocast forward finite")

    # backward
    m.train()
    out = m(feats, coords, 1)
    out.sum().backward()
    g = m.conv_input[0].weight.grad
    assert g is not None and torch.isfinite(g).all()
    print("[OK] backward: conv_input grad finite")
    print("SPARSE ENCODER TESTS PASSED")


if __name__ == "__main__":
    main()
