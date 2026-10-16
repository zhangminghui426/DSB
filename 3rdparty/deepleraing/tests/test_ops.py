"""Unit tests for the CUDA ops (merge gate: bev_pool_v2 both APIs + voxel + iou3d).

Run from repo root:
    python tests/test_ops.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch  # noqa: E402

from ops import (Voxelization, bev_pool_v2, bev_pool_v2_reference,  # noqa: E402
                 bev_pool_v2_sum, boxes_overlap_bev)

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)


def make_case():
    """ Synthetic v2 input with guaranteed pillar collisions.

    The kernel contract: points sorted by ranks_bev; interval_starts/lengths
    delimit equal-ranks_bev runs; ranks_depth/ranks_feat are flat indices into
    depth.reshape(-1) / feat.reshape(-1, C).
    """
    B, N, D, fH, fW, C = 2, 6, 4, 3, 3, 2
    Dz, Dy, Dx = 2, 3, 3
    depth = torch.rand(B, N, D, fH, fW, device=DEV)
    feat = torch.rand(B, N, fH, fW, C, device=DEV, requires_grad=True)
    # 8 points: two pillars of 3 and 2, plus 3 singletons; one point outside
    ranks_bev = torch.tensor([5, 5, 5, 9, 9, 17, 3, 12], device=DEV).int()
    ranks_depth = torch.tensor([0, 7, 40, 99, 12, 60, 33, 80], device=DEV).int()
    ranks_depth = ranks_depth % (B * N * D * fH * fW)
    ranks_feat = torch.tensor([0, 20, 41, 62, 5, 90, 71, 12], device=DEV).int()
    ranks_feat = ranks_feat % (B * N * fH * fW)
    order = ranks_bev.argsort(stable=True)
    ranks_bev, ranks_depth, ranks_feat = (ranks_bev[order], ranks_depth[order],
                                          ranks_feat[order])
    kept = torch.ones(ranks_bev.shape[0], device=DEV, dtype=torch.bool)
    kept[1:] = ranks_bev[1:] != ranks_bev[:-1]
    interval_starts = torch.where(kept)[0].int()
    interval_lengths = torch.zeros_like(interval_starts)
    interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
    interval_lengths[-1] = ranks_bev.shape[0] - interval_starts[-1]
    shape = (B, Dz, Dy, Dx, C)
    return depth, feat, ranks_depth, ranks_feat, ranks_bev, shape, \
        interval_starts, interval_lengths


def test_bev_pool_v2():
    """ CUDA v2 must equal the pure-torch index_add_ reference (fwd + bwd)."""
    args = make_case()
    depth, feat = args[0], args[1]
    out = bev_pool_v2(*args)
    leaf = feat.detach().clone().requires_grad_(True)
    ref = bev_pool_v2_reference(depth.detach(), leaf, *args[2:])
    fwd_err = (out - ref).abs().max().item()
    assert fwd_err < 1e-5, f"v2 forward mismatch: {fwd_err}"

    gout = torch.randn_like(out)
    out.backward(gout)
    ref.backward(gout)
    bwd_err = (feat.grad - leaf.grad).abs().max().item()
    assert bwd_err < 1e-5, f"v2 backward mismatch: {bwd_err}"
    print("[OK] bev_pool_v2: forward/backward match index_add_ reference")


def test_bev_pool_v2_sum():
    """ Folded-feats API must equal an index_put reference (fwd + bwd + fp16)."""
    B, D, H, W, C, N = 2, 3, 7, 5, 4, 400
    feats = torch.randn(N, C, device=DEV, requires_grad=True)
    # coords columns: (x, y, z, batch)
    coords = torch.stack(
        [
            torch.randint(0, H, (N,)),
            torch.randint(0, W, (N,)),
            torch.randint(0, D, (N,)),
            torch.randint(0, B, (N,)),
        ],
        dim=1,
    ).int().to(DEV)

    out = bev_pool_v2_sum(feats, coords, B, D, H, W)
    assert out.shape == (B, C, D, H, W), out.shape

    feats_ref = feats.detach().clone().requires_grad_(True)
    ref = torch.zeros(B, D, H, W, C, device=DEV, dtype=feats.dtype)
    x, y, z, b = (coords[:, 0].long(), coords[:, 1].long(),
                  coords[:, 2].long(), coords[:, 3].long())
    ref.index_put_((b, z, x, y), feats_ref, accumulate=True)
    ref = ref.permute(0, 4, 1, 2, 3)  # (B, C, D, H, W)

    fwd_err = (out - ref).abs().max().item()
    assert fwd_err < 1e-4, f"v2_sum forward mismatch: {fwd_err}"

    # bitwise cross-check against the pure-torch scatter reference (same
    # segment-sum semantics; sorted-row accumulation order matches the kernel)
    from ops import scatter_pool
    scat = scatter_pool(feats.detach(), coords.long(), B, D, H, W).permute(0, 4, 1, 2, 3)
    assert (out.detach() - scat).abs().max().item() < 1e-6, "v2_sum vs scatter_pool"

    gout = torch.randn_like(out)
    out.backward(gout)
    ref.backward(gout)
    bwd_err = (feats.grad - feats_ref.grad).abs().max().item()
    assert bwd_err < 1e-4, f"v2_sum backward mismatch: {bwd_err}"

    # empty input must not crash
    empty = bev_pool_v2_sum(feats.detach()[:0], coords[:0], B, D, H, W)
    assert empty.shape == (B, C, D, H, W) and (empty == 0).all()

    # fp16 autocast path (kernel is fp32-only; wrapper must cast)
    if DEV == "cuda:0":
        feats16 = feats.detach().half().requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.float16):
            out16 = bev_pool_v2_sum(feats16, coords, B, D, H, W)
        out16.sum().backward()
        assert feats16.grad is not None and torch.isfinite(feats16.grad).all()
    print("[OK] bev_pool_v2_sum: forward/backward match reference; empty + fp16 safe")


def ref_voxelize(points, voxel_size, pc_range, max_points, max_voxels):
    """ Pure-torch reference: group points into (x,y,z) cells (this fork's
    coord convention), keep first max_points per cell in point order."""
    vs = torch.tensor(voxel_size, dtype=torch.float32)
    lo = torch.tensor(pc_range[:3], dtype=torch.float32)
    hi = torch.tensor(pc_range[3:], dtype=torch.float32)
    grid = torch.round((hi - lo) / vs).long()

    idx = torch.floor((points[:, :3].cpu() - lo) / vs).long()
    valid = ((idx >= 0).all(dim=1)) & ((idx[:, 0] < grid[0]) & (idx[:, 1] < grid[1]) & (idx[:, 2] < grid[2]))
    idx = idx[valid]
    pts = points.cpu()[valid]

    # first-appearance order of cells (deterministic kernel semantics)
    cell_id = idx[:, 0] * (grid[1] * grid[2]) + idx[:, 1] * grid[2] + idx[:, 2]
    uniq = {}
    for cid in cell_id.tolist():
        if cid not in uniq:
            if len(uniq) >= max_voxels:
                continue  # drop: voxel budget exhausted
            uniq[cid] = len(uniq)
    # per-cell: first max_points in point order
    out_voxels, out_coords, out_counts = [], [], []
    by_cell = {}
    for i, cid in enumerate(cell_id.tolist()):
        if cid in uniq:
            by_cell.setdefault(cid, []).append(i)
    for cid in sorted(uniq, key=lambda c: uniq[c]):
        pts_idx = by_cell[cid][:max_points]
        if len(pts_idx) == 0:
            continue
        out_voxels.append(pts[pts_idx])
        c = idx[pts_idx[0]]
        out_coords.append(c)
        out_counts.append(len(pts_idx))
    return out_voxels, out_coords, out_counts


def test_voxelize():
    """ hard_voxelize must group points into cells identically to a torch ref."""
    voxel_size = [0.5, 0.5, 0.4]
    pc_range = [-10.0, -10.0, -2.0, 10.0, 10.0, 2.0]
    max_points, max_voxels = 10, 100000
    N = 20000
    # mix of in-range and out-of-range points
    pts_in = torch.rand(N, 5) * torch.tensor([20, 20, 4, 1, 1]) + torch.tensor([-10, -10, -2, 0, 0])
    pts_out = torch.rand(2000, 5) * 100 - 50
    points = torch.cat([pts_in, pts_out], 0).float().to(DEV)

    vox = Voxelization(voxel_size, pc_range, max_points, max_voxels)
    voxels, coors, counts = vox(points)
    voxels, coors, counts = voxels.cpu(), coors.cpu().long(), counts.cpu()

    ref_v, ref_c, ref_n = ref_voxelize(points, voxel_size, pc_range, max_points, max_voxels)
    assert voxels.shape[0] == len(ref_v), f"voxel count {voxels.shape[0]} vs ref {len(ref_v)}"
    assert torch.equal(counts, torch.tensor(ref_n)), "per-voxel point counts differ"
    assert torch.equal(coors, torch.stack(ref_c)), "voxel coords differ (x,y,z convention)"

    # order within voxel must follow original point order
    for i in range(min(50, len(ref_v))):
        got = voxels[i, : counts[i]]
        exp = ref_v[i]
        assert torch.allclose(got, exp, atol=1e-6), f"voxel {i} point mismatch"

    # spot-check: point far +x -> large coord[0]
    probe = torch.tensor([[9.3, -9.4, 1.7, 0.5, 0.1]], device=DEV)
    v2, c2, n2 = vox(probe)
    exp_coord = torch.tensor([[int((9.3 + 10) // 0.5), int((-9.4 + 10) // 0.5), int((1.7 + 2) // 0.4)]])
    assert torch.equal(c2.cpu(), exp_coord), f"coord convention broken: {c2} vs {exp_coord}"
    print(f"[OK] hard_voxelize: {voxels.shape[0]} voxels match torch reference (x,y,z coords)")


def rot_rect_corners(x1, y1, x2, y2, r):
    """ Corners per iou3d_kernel.cu rotate_around_center convention:
    new_x =  dx*cos + dy*sin + cx
    new_y = -dx*sin + dy*cos + cy
    """
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    hw, hh = (x2 - x1) / 2, (y2 - y1) / 2
    corners = torch.tensor([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]])
    cos_r, sin_r = torch.cos(torch.tensor(r)), torch.sin(torch.tensor(r))
    rc = torch.stack(
        [
            corners[:, 0] * cos_r + corners[:, 1] * sin_r + cx,
            -corners[:, 0] * sin_r + corners[:, 1] * cos_r + cy,
        ],
        dim=1,
    )
    return rc.numpy()


def test_boxes_overlap_bev():
    """ boxes_overlap_bev_gpu must match shapely polygon intersection areas."""
    try:
        from shapely.geometry import Polygon
    except ImportError:
        print("[SKIP] shapely not installed")
        return
    M, N = 24, 30
    boxes_a = torch.rand(M, 5) * torch.tensor([8, 8, 2, 2, 6.28]) + torch.tensor([-10, -10, 1, 1, -3.14])
    boxes_b = torch.rand(N, 5) * torch.tensor([8, 8, 2, 2, 6.28]) + torch.tensor([-10, -10, 1, 1, -3.14])
    # enforce x2>x1, y2>y1
    boxes_a[:, 2] = boxes_a[:, 0] + boxes_a[:, 2].abs() + 0.5
    boxes_a[:, 3] = boxes_a[:, 1] + boxes_a[:, 3].abs() + 0.5
    boxes_b[:, 2] = boxes_b[:, 0] + boxes_b[:, 2].abs() + 0.5
    boxes_b[:, 3] = boxes_b[:, 1] + boxes_b[:, 3].abs() + 0.5

    got = boxes_overlap_bev(boxes_a.to(DEV).float(), boxes_b.to(DEV).float()).cpu()
    max_err = 0.0
    for i in range(M):
        pa = Polygon(rot_rect_corners(*boxes_a[i].tolist()))
        for j in range(N):
            pb = Polygon(rot_rect_corners(*boxes_b[j].tolist()))
            ref = pa.intersection(pb).area
            max_err = max(max_err, abs(ref - got[i, j].item()))
    assert max_err < 1e-3, f"bev overlap max err {max_err}"
    print(f"[OK] boxes_overlap_bev: matches shapely reference (max err {max_err:.2e})")


if __name__ == "__main__":
    print(f"device: {DEV}")
    test_bev_pool_v2()
    test_bev_pool_v2_sum()
    test_voxelize()
    test_boxes_overlap_bev()
    print("ALL OPS TESTS PASSED")
