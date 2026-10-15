"""LSSViewTransformer numeric tests: frustum (UD/SID), ego-coordinate
projection, voxel-pooling ranks vs a brute-force reference, and GT depth
discretization + depth loss.

Run from repo root:
    python tests/test_vtransform.py
"""

import sys

import os.path as osp; sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), ".."))

import torch

from models.vtransforms.occ_lss import LSSViewTransformer
from models.vtransforms.occ_lss_depth import LSSViewTransformerBEVDepth

GRID = {"x": [-40, 40, 0.4], "y": [-40, 40, 0.4], "z": [-1, 5.4, 6.4],
        "depth": [1.0, 45.0, 0.5]}


def make_vt(sid=False):
    return LSSViewTransformer(
        grid_config=GRID, input_size=(256, 704), downsample=16,
        in_channels=8, out_channels=4, sid=sid, collapse_z=True,
    )


def test_frustum_ud():
    vt = make_vt()
    f = vt.frustum
    assert vt.D == 88
    assert tuple(f.shape) == (88, 16, 44, 3)
    d = f[..., 2]
    assert torch.allclose(d[0], torch.full((16, 44), 1.0))
    assert torch.allclose(d[87], torch.full((16, 44), 44.5))
    assert torch.allclose(f[..., 0][0, 0], torch.linspace(0, 703, 44))
    assert torch.allclose(f[..., 1][0, :, 0], torch.linspace(0, 255, 16))
    # frustum must NOT be a state_dict entry (plain attribute)
    assert "frustum" not in dict(vt.state_dict())
    print("[OK] frustum UD: D=88, (u,v,d) order, non-buffer")


def test_frustum_sid():
    vt = make_vt(sid=True)
    assert vt.D == 88
    i = torch.arange(88).float()
    expect = torch.exp(i / 87 * torch.log(torch.tensor(44.0)))  # d0=1 -> 44^(i/87)
    assert torch.allclose(vt.frustum[..., 2][:, 0, 0], expect, atol=1e-4)
    print("[OK] frustum SID: 44^(i/87) spacing")


def test_ego_coor_identity():
    vt = make_vt()
    B, N = 1, 2
    identity4 = torch.eye(4).expand(B, N, 4, 4)
    identity3 = torch.eye(3).expand(B, N, 3, 3)
    zeros3 = torch.zeros(B, N, 3)
    bda = torch.eye(3).expand(B, 3, 3)
    pts = vt.get_ego_coor(identity4, identity4, identity3, identity3, zeros3, bda)
    f = vt.frustum
    u, v, d = f[..., 0], f[..., 1], f[..., 2]
    expect = torch.stack([u * d, v * d, d], -1).view(1, 1, *f.shape[:3], 3).expand(B, N, *f.shape[:3], 3)
    assert torch.allclose(pts, expect, atol=1e-4)
    print("[OK] get_ego_coor with identity poses == (u*d, v*d, d)")


def test_ranks_reference():
    """voxel_pooling_prepare_v2 vs a brute-force python reference."""
    vt = make_vt()
    B, N, D, H, W = 1, 2, 2, 2, 2
    # deterministic multi-point-per-voxel layout: 8 pts in voxel (0,0,0),
    # 7 pts in voxel (1,0,0), 1 pt far outside -> intervals of length 8 and 7
    coor = torch.zeros(B, N, D, H, W, 3)
    coor[..., :] = torch.tensor([-39.9, -39.9, -0.9])   # voxel x=0
    coor.view(-1, 3)[8:15] = torch.tensor([-39.5, -39.9, -0.9])  # voxel x=1
    coor.view(-1, 3)[15] = torch.tensor([1000.0, 0.0, 0.0])      # outside

    ranks_bev, ranks_depth, ranks_feat, starts, lengths = vt.voxel_pooling_prepare_v2(coor)

    # brute-force reference
    lb, iv, gs = vt.grid_lower_bound, vt.grid_interval, vt.grid_size
    num = B * N * D * H * W
    cells = ((coor - lb) / iv).long().view(num, 3)
    rd = torch.arange(num, dtype=torch.long)
    rf = torch.arange(num // D, dtype=torch.long).reshape(B, N, 1, H, W).expand(B, N, D, H, W).flatten()
    kept = (
        (cells[:, 0] >= 0) & (cells[:, 0] < gs[0])
        & (cells[:, 1] >= 0) & (cells[:, 1] < gs[1])
        & (cells[:, 2] >= 0) & (cells[:, 2] < gs[2])
    )
    cells, rd, rf = cells[kept], rd[kept], rf[kept]
    b_idx = torch.arange(num, dtype=torch.long)[kept] // (num // B)
    bev = b_idx * (gs[2] * gs[1] * gs[0]) + cells[:, 2] * (gs[1] * gs[0]) + cells[:, 1] * gs[0] + cells[:, 0]

    triples_model = sorted(zip(ranks_bev.tolist(), ranks_depth.tolist(), ranks_feat.tolist()))
    triples_ref = sorted(zip(bev.tolist(), rd.tolist(), rf.tolist()))
    assert triples_model == triples_ref, "ranks mismatch vs reference"

    # intervals must partition the sorted ranks by unique bev id
    assert starts[0] == 0
    assert int(lengths.sum()) == len(triples_model)
    bev_sorted = ranks_bev.tolist()
    pos = 0
    for s, l in zip(starts.tolist(), lengths.tolist()):
        assert s == pos
        assert len(set(bev_sorted[s:s + l])) == 1
        pos = s + l
    print(f"[OK] ranks vs brute-force reference ({len(triples_model)} points kept)")


def test_gt_depth_discretization():
    vt = LSSViewTransformerBEVDepth(
        grid_config=GRID, input_size=(32, 32), downsample=4,
        in_channels=8, out_channels=4, sid=False, collapse_z=True,
    )
    B, N, Himg, Wimg = 1, 1, 32, 32
    gt = torch.zeros(B, N, Himg, Wimg)
    gt[0, 0, 0, 0] = 1.0    # -> bin index 1 -> one-hot col 0
    gt[0, 0, 0, 4] = 2.25   # (2.25-0.5)/0.5 = 3.5 -> long 3 -> col 2
    gt[0, 0, 8, 0] = 0.4    # below near bound -> zero row
    gt[0, 0, 12, 0] = 500.0  # beyond far bound -> zero row
    out = vt.get_downsampled_gt_depth(gt)
    assert out.shape == (B * N * 8 * 8, 88)
    px = out[0]  # pixel (0,0) window contains depths {1.0 (min), 0...} -> min nonzero = 1.0
    assert px[0] == 1.0 and px.sum() == 1.0
    px2 = out[1]  # window (row0, col1) contains 2.25
    assert px2[2] == 1.0 and px2.sum() == 1.0
    assert out[2 * 8 + 0].sum() == 0.0  # 0.4 window -> zeros
    assert out[3 * 8 + 0].sum() == 0.0  # 500 window -> zeros

    vt_sid = LSSViewTransformerBEVDepth(
        grid_config=GRID, input_size=(32, 32), downsample=4,
        in_channels=8, out_channels=4, sid=True, collapse_z=True,
    )
    gt2 = torch.zeros(B, N, Himg, Wimg)
    gt2[0, 0, 0, 0] = 44.0  # SID far bin: (log44-log1)*87/log44 + 1 = 88 -> col 87
    out2 = vt_sid.get_downsampled_gt_depth(gt2)
    assert out2[0][87] == 1.0 and out2[0].sum() == 1.0
    print("[OK] GT depth discretization UD + SID")


def test_depth_loss_finite():
    vt = LSSViewTransformerBEVDepth(
        grid_config=GRID, input_size=(32, 32), downsample=4,
        in_channels=8, out_channels=4, sid=False, collapse_z=True,
        loss_depth_weight=0.05,
    )
    B, N = 1, 6
    gt = torch.zeros(B, N, 32, 32)
    gt[:, :, ::3, ::5] = torch.rand(B, N, 11, 7) * 40 + 1  # sparse lidar-like
    preds = torch.rand(B * N, 88, 8, 8).softmax(dim=1)
    loss = vt.get_depth_loss(gt, preds)
    assert torch.isfinite(loss) and loss.item() > 0
    print(f"[OK] depth loss finite: {loss.item():.4f}")


def main():
    test_frustum_ud()
    test_frustum_sid()
    test_ego_coor_identity()
    test_ranks_reference()
    test_gt_depth_discretization()
    test_depth_loss_finite()
    print("ALL VTRANSFORM TESTS PASSED")


if __name__ == "__main__":
    main()
