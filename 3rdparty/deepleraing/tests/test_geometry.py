""" Unit tests for datasets.geometry: LiDARPoints, LiDARInstance3DBoxes, TransFusionBBoxCoder.

Run from repo root:
    python tests/test_geometry.py
"""

import sys
import math
import numpy as np
import torch

import os.path as osp; sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), ".."))

from datasets.geometry.points import LiDARPoints
from datasets.geometry.boxes import (
    LiDARInstance3DBoxes,
    limit_period,
    xywhr2xyxyr,
    rotation_3d_in_axis,
)
from models.core.bbox_coder import TransFusionBBoxCoder

HAS_CUDA = torch.cuda.is_available()

passed = 0
failed = 0


def check(name, condition, msg=""):
    global passed, failed
    if condition:
        print(f"  [OK] {name}")
        passed += 1
    else:
        print(f"  [FAIL] {name}: {msg}")
        failed += 1


# ===========================================================================
# LiDARPoints tests
# ===========================================================================
print("=== LiDARPoints ===")

# 1) rotate(yaw) then rotate(-yaw) ≈ identity
torch.manual_seed(42)
pts_data = torch.randn(100, 5)  # (x, y, z, intensity, ring)
pts = LiDARPoints(pts_data.clone(), points_dim=5)
original = pts.tensor.clone()

pts.rotate(0.3)
pts.rotate(-0.3)
diff = (pts.tensor[:, :3] - original[:, :3]).abs().max().item()
check(
    "rotate roundtrip (0.3 then -0.3)",
    diff < 1e-5,
    f"max diff={diff}",
)

# 2) translate + scale roundtrip
pts2 = LiDARPoints(pts_data.clone(), points_dim=5)
pts2.translate(torch.tensor([1.0, 2.0, 3.0]))
pts2.translate(torch.tensor([-1.0, -2.0, -3.0]))
diff_t = (pts2.tensor[:, :3] - original[:, :3]).abs().max().item()
check("translate roundtrip", diff_t < 1e-5, f"max diff={diff_t}")

pts3 = LiDARPoints(pts_data.clone(), points_dim=5)
pts3.scale(2.0)
pts3.scale(0.5)
diff_s = (pts3.tensor[:, :3] - original[:, :3]).abs().max().item()
check("scale roundtrip", diff_s < 1e-5, f"max diff={diff_s}")

# 3) in_range_3d on hand-made points
hand_pts = LiDARPoints(
    torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 0.0],
            [5.0, 5.0, 5.0, 1.0, 0.0],
            [-1.0, -1.0, -1.0, 1.0, 0.0],
        ]
    ),
    points_dim=5,
)
mask = hand_pts.in_range_3d([0.0, 0.0, 0.0, 2.0, 2.0, 2.0])
expected = torch.tensor([False, True, False, False])
check(
    "in_range_3d mask",
    torch.equal(mask, expected),
    f"got {mask}, expected {expected}",
)

# 4) in_range_bev
mask_bev = hand_pts.in_range_bev([0.0, 0.0, 2.0, 2.0])
expected_bev = torch.tensor([False, True, False, False])
check(
    "in_range_bev mask",
    torch.equal(mask_bev, expected_bev),
    f"got {mask_bev}, expected {expected_bev}",
)

# 5) __getitem__ and cat
sub = hand_pts[1]
check("__getitem__ int returns 1 point", len(sub) == 1 and sub.tensor[0, 0] == 1.0)

cat_pts = LiDARPoints.cat([hand_pts[:2], hand_pts[2:]])
check("cat roundtrip", len(cat_pts) == 4 and torch.allclose(cat_pts.tensor, hand_pts.tensor))

# 6) to/clone
cloned = hand_pts.clone()
check("clone", torch.equal(cloned.tensor, hand_pts.tensor) and cloned is not hand_pts)

print()

# ===========================================================================
# LiDARInstance3DBoxes tests
# ===========================================================================
print("=== LiDARInstance3DBoxes ===")

# Box layout: (x, y, z, x_size, y_size, z_size, yaw, [vx, vy])
# origin=(0.5, 0.5, 0) => z is the bottom of the box

# 1) Basic construction and properties
box_data = torch.tensor(
    [
        [1.0, 2.0, 0.0, 3.0, 4.0, 5.0, 0.0, 0.1, 0.2],  # box0
        [10.0, 20.0, 1.0, 2.0, 3.0, 4.0, 0.5, 0.0, 0.0],  # box1
    ]
)
boxes = LiDARInstance3DBoxes(box_data, box_dim=9, with_yaw=True, origin=(0.5, 0.5, 0))
check("box_dim=9", boxes.box_dim == 9)
check("len", len(boxes) == 2)
check("volume", torch.allclose(boxes.volume, torch.tensor([60.0, 24.0])))

# 2) .bev gives (x, y, x_size, y_size, yaw)
bev = boxes.bev
check(
    ".bev layout",
    torch.allclose(bev[0], torch.tensor([1.0, 2.0, 3.0, 4.0, 0.0])),
    f"got {bev[0]}",
)

# 3) gravity_center: z_gravity = z_bottom + z_size * 0.5
gc = boxes.gravity_center
check(
    "gravity_center z",
    torch.allclose(gc[:, 2], torch.tensor([2.5, 3.0])),
    f"got {gc[:, 2]}",
)

# 4) origin=(0.5, 0.5, 0) conversion: if we pass origin=(0.5, 0.5, 0.5),
# z should shift by -z_size * 0.5
box_center = torch.tensor([[1.0, 2.0, 2.5, 3.0, 4.0, 5.0, 0.0]])
boxes_origin_center = LiDARInstance3DBoxes(
    box_center, box_dim=7, with_yaw=True, origin=(0.5, 0.5, 0.5)
)
# Should convert to bottom-center: z becomes 2.5 + 5.0 * (0 - 0.5) = 0.0
check(
    "origin conversion (0.5,0.5,0.5) -> (0.5,0.5,0)",
    abs(boxes_origin_center.tensor[0, 2].item() - 0.0) < 1e-6,
    f"got z={boxes_origin_center.tensor[0, 2].item()}",
)

# 5) rotate(pi/2) maps box at (1,0) yaw=0 to (0,1) yaw=pi/2
box_rot = LiDARInstance3DBoxes(
    torch.tensor([[1.0, 0.0, 0.0, 2.0, 3.0, 1.0, 0.0]]),
    box_dim=7,
    with_yaw=True,
)
box_rot.rotate(math.pi / 2)
center = box_rot.tensor[0, :3]
yaw = box_rot.tensor[0, 6].item()
# In LiDAR coords, rotate(pi/2) maps (1,0) → (0,-1) (clockwise from +z)
check(
    "rotate(pi/2) center",
    torch.allclose(center, torch.tensor([0.0, -1.0, 0.0]), atol=1e-6),
    f"got {center}",
)
check(
    "rotate(pi/2) yaw",
    abs(yaw - math.pi / 2) < 1e-6,
    f"got yaw={yaw}",
)

# 6) Corners consistency: after rotate, corners should also rotate
box_for_corners = LiDARInstance3DBoxes(
    torch.tensor([[0.0, 0.0, 0.0, 2.0, 4.0, 1.0, 0.0]]),
    box_dim=7,
    with_yaw=True,
)
corners_before = box_for_corners.corners.clone()

box_for_corners2 = LiDARInstance3DBoxes(
    torch.tensor([[0.0, 0.0, 0.0, 2.0, 4.0, 1.0, 0.0]]),
    box_dim=7,
    with_yaw=True,
)
box_for_corners2.rotate(math.pi / 2)
corners_after = box_for_corners2.corners

# Rotate corners_before by pi/2 around z and compare
rot_mat = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
expected_corners = corners_before[0] @ rot_mat
check(
    "corners consistent with rotate",
    torch.allclose(corners_after[0], expected_corners, atol=1e-5),
    f"max diff={(corners_after[0] - expected_corners).abs().max().item()}",
)

# 7) volume = x_size * y_size * z_size
box_vol = LiDARInstance3DBoxes(
    torch.tensor([[0.0, 0.0, 0.0, 2.0, 3.0, 4.0, 0.5]]),
    box_dim=7,
    with_yaw=True,
)
check("volume = w*l*h", abs(box_vol.volume.item() - 24.0) < 1e-6)

# 8) overlaps: identical boxes IoU ≈ 1, disjoint ≈ 0 (needs CUDA)
if HAS_CUDA:
    box_a = LiDARInstance3DBoxes(
        torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0]]),
        box_dim=7,
        with_yaw=True,
    )
    iou_same = LiDARInstance3DBoxes.overlaps(box_a, box_a.clone())
    check(
        "overlaps identical IoU≈1",
        abs(iou_same[0, 0].item() - 1.0) < 0.01,
        f"got {iou_same[0, 0].item()}",
    )

    box_far = LiDARInstance3DBoxes(
        torch.tensor([[100.0, 100.0, 0.0, 2.0, 2.0, 2.0, 0.0]]),
        box_dim=7,
        with_yaw=True,
    )
    iou_far = LiDARInstance3DBoxes.overlaps(box_a, box_far)
    check(
        "overlaps disjoint IoU≈0",
        iou_far[0, 0].item() < 0.01,
        f"got {iou_far[0, 0].item()}",
    )

    # 9) Compare overlaps vs shapely for 3 random rotated pairs
    try:
        from shapely.geometry import Polygon

        torch.manual_seed(123)
        for trial in range(3):
            # Random box A
            xa, ya = torch.randn(2) * 5
            wa, la = torch.rand(2) * 3 + 1
            yaw_a = torch.rand(1).item() * math.pi * 2

            # Random box B
            xb, yb = torch.randn(2) * 5
            wb, lb = torch.rand(2) * 3 + 1
            yaw_b = torch.rand(1).item() * math.pi * 2

            h = 2.0
            boxA = LiDARInstance3DBoxes(
                torch.tensor([[xa.item(), ya.item(), 0.0, wa.item(), la.item(), h, yaw_a]]),
                box_dim=7,
                with_yaw=True,
            )
            boxB = LiDARInstance3DBoxes(
                torch.tensor([[xb.item(), yb.item(), 0.0, wb.item(), lb.item(), h, yaw_b]]),
                box_dim=7,
                with_yaw=True,
            )

            iou_3d = LiDARInstance3DBoxes.overlaps(boxA, boxB)[0, 0].item()

            # Compute with shapely using corners
            cA = boxA.corners[0].cpu().numpy()
            cB = boxB.corners[0].cpu().numpy()
            # Bottom face corners (z=0 level): indices 0,1,2,3
            polyA = Polygon(cA[:4, :2])
            polyB = Polygon(cB[:4, :2])
            bev_intersection = polyA.intersection(polyB).area
            bev_union = polyA.union(polyB).area
            bev_iou = bev_intersection / bev_union if bev_union > 0 else 0.0

            # Height overlap is full (both at z=0, height=2)
            # So 3D IoU = BEV IoU * h_overlap / (volA + volB - BEV_IoU_area * h_overlap)
            vol_a = wa.item() * la.item() * h
            vol_b = wb.item() * lb.item() * h
            intersection_3d = bev_intersection * h
            union_3d = vol_a + vol_b - intersection_3d
            expected_iou = intersection_3d / union_3d if union_3d > 0 else 0.0

            check(
                f"overlaps vs shapely trial {trial}",
                abs(iou_3d - expected_iou) < 0.02,
                f"got {iou_3d:.4f}, expected {expected_iou:.4f}",
            )
    except ImportError:
        print("  [SKIP] shapely not installed")
else:
    print("  [SKIP] overlaps tests (no CUDA)")

# 10) tensor property roundtrip
box_rt = LiDARInstance3DBoxes(
    torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.1, 0.2, 0.3]]),
    box_dim=9,
    with_yaw=True,
)
check(
    "tensor roundtrip",
    torch.allclose(box_rt.tensor[0], torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.1, 0.2, 0.3])),
)

# 11) cat of two sets
b1 = LiDARInstance3DBoxes(
    torch.tensor([[1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]]),
    box_dim=7,
    with_yaw=True,
)
b2 = LiDARInstance3DBoxes(
    torch.tensor([[2.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0]]),
    box_dim=7,
    with_yaw=True,
)
b_cat = LiDARInstance3DBoxes.cat([b1, b2])
check("cat two box sets", len(b_cat) == 2 and b_cat.tensor[0, 0] == 1.0 and b_cat.tensor[1, 0] == 2.0)

# 12) __getitem__
single = b_cat[0]
check("__getitem__", len(single) == 1 and single.tensor[0, 0] == 1.0)

# 13) limit_yaw
box_ly = LiDARInstance3DBoxes(
    torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 4.0]]),
    box_dim=7,
    with_yaw=True,
)
box_ly.limit_yaw()
check(
    "limit_yaw",
    -math.pi / 2 - 1e-6 <= box_ly.tensor[0, 6].item() <= math.pi / 2 + 1e-6,
    f"got yaw={box_ly.tensor[0, 6].item()}",
)

# 14) velocity rotation
box_vel = LiDARInstance3DBoxes(
    torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0]]),
    box_dim=9,
    with_yaw=True,
)
box_vel.rotate(math.pi / 2)
# vx,vy = (1,0) rotated by pi/2 should become (0,1)
# velocity (1,0) rotated pi/2 in LiDAR convention → (0,-1)
check(
    "velocity rotates with box",
    torch.allclose(box_vel.tensor[0, 7:9], torch.tensor([0.0, -1.0]), atol=1e-6),
    f"got vel={box_vel.tensor[0, 7:9]}",
)

print()

# ===========================================================================
# TransFusionBBoxCoder tests
# ===========================================================================
print("=== TransFusionBBoxCoder ===")

coder = TransFusionBBoxCoder(
    pc_range=[-54.0, -54.0],
    voxel_size=[0.075, 0.075],
    out_size_factor=8,
    post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
    score_threshold=0.0,
    code_size=10,
)

# Build synthetic inputs
# Feature map: 180x180, flattened to 32400 proposals
num_cls = 10
num_proposals = 180 * 180  # 32400
B = 1

# Place a peak at class 3, at grid cell (row=90, col=60)
target_row, target_col = 90, 60
target_idx = target_row * 180 + target_col

heatmap = torch.full((B, num_cls, num_proposals), -10.0)
heatmap[0, 3, target_idx] = 10.0  # high logit for class 3

# center: feature map coordinates for each proposal
# For the peak cell, center_x = col, center_y = row
center = torch.zeros(B, 2, num_proposals)
# Set all proposals to their grid coordinates
rows = torch.arange(180).repeat(180)  # 0,0,...,0,1,1,...,1,...
cols = torch.arange(180).repeat_interleave(180)  # wait, this is wrong
# Actually: for idx = row*180 + col, row = idx // 180, col = idx % 180
all_rows = torch.arange(num_proposals) // 180
all_cols = torch.arange(num_proposals) % 180
center[0, 0, :] = all_cols.float()  # x = col
center[0, 1, :] = all_rows.float()  # y = row

# dim: log-scale dimensions (w, l, h)
log_w, log_l, log_h = math.log(2.0), math.log(3.0), math.log(1.5)
dim = torch.zeros(B, 3, num_proposals)
dim[0, 0, :] = log_w
dim[0, 1, :] = log_l
dim[0, 2, :] = log_h

# height: gravity center z
target_z_gravity = 2.0
height = torch.zeros(B, 1, num_proposals)
height[0, 0, :] = target_z_gravity

# rot: sin/cos of yaw
target_yaw = 0.3
rot = torch.zeros(B, 2, num_proposals)
rot[0, 0, :] = math.sin(target_yaw)
rot[0, 1, :] = math.cos(target_yaw)

# velocity
vel = torch.zeros(B, 2, num_proposals)
vel[0, 0, :] = 1.0
vel[0, 1, :] = -0.5

# Decode without filter
results = coder.decode(
    heatmap.clone(), rot.clone(), dim.clone(), center.clone(), height.clone(), vel.clone(),
    filter=False,
)
pred = results[0]
check("decode returns 1 batch", len(results) == 1)
check(
    "decode num proposals",
    pred["bboxes"].shape[0] == num_proposals,
    f"got {pred['bboxes'].shape[0]}",
)

# Check the peak box
peak_box = pred["bboxes"][target_idx]
peak_score = pred["scores"][target_idx]
peak_label = pred["labels"][target_idx]

check("decode peak label = 3", peak_label.item() == 3, f"got {peak_label.item()}")
check(
    "decode peak score ≈ 1.0 (sigmoid of 10)",
    peak_score.item() > 0.99,
    f"got {peak_score.item()}",
)

# Expected center: col * out_size_factor * voxel_size + pc_range[0]
expected_x = target_col * 8 * 0.075 + (-54.0)
expected_y = target_row * 8 * 0.075 + (-54.0)
check(
    "decode x",
    abs(peak_box[0].item() - expected_x) < 1e-4,
    f"got {peak_box[0].item()}, expected {expected_x}",
)
check(
    "decode y",
    abs(peak_box[1].item() - expected_y) < 1e-4,
    f"got {peak_box[1].item()}, expected {expected_y}",
)

# Expected z: height - dim_h * 0.5 (gravity center to bottom center)
expected_z = target_z_gravity - 1.5 * 0.5
check(
    "decode z (bottom center)",
    abs(peak_box[2].item() - expected_z) < 1e-4,
    f"got {peak_box[2].item()}, expected {expected_z}",
)

# Expected dims: exp(log_w), exp(log_l), exp(log_h)
check("decode w", abs(peak_box[3].item() - 2.0) < 1e-4, f"got {peak_box[3].item()}")
check("decode l", abs(peak_box[4].item() - 3.0) < 1e-4, f"got {peak_box[4].item()}")
check("decode h", abs(peak_box[5].item() - 1.5) < 1e-4, f"got {peak_box[5].item()}")

# Expected yaw: atan2(sin, cos)
expected_yaw = math.atan2(math.sin(target_yaw), math.cos(target_yaw))
check(
    "decode yaw",
    abs(peak_box[6].item() - expected_yaw) < 1e-4,
    f"got {peak_box[6].item()}, expected {expected_yaw}",
)

# Expected velocity
check(
    "decode vx",
    abs(peak_box[7].item() - 1.0) < 1e-4,
    f"got {peak_box[7].item()}",
)
check(
    "decode vy",
    abs(peak_box[8].item() - (-0.5)) < 1e-4,
    f"got {peak_box[8].item()}",
)

# Test encode roundtrip
gt_boxes = torch.tensor(
    [[expected_x, expected_y, expected_z, 2.0, 3.0, 1.5, expected_yaw, 1.0, -0.5]]
)
encoded = coder.encode(gt_boxes)
check(
    "encode center_x",
    abs(encoded[0, 0].item() - target_col) < 1e-3,
    f"got {encoded[0, 0].item()}, expected {target_col}",
)
check(
    "encode center_y",
    abs(encoded[0, 1].item() - target_row) < 1e-3,
    f"got {encoded[0, 1].item()}, expected {target_row}",
)
check(
    "encode gravity_z",
    abs(encoded[0, 2].item() - (expected_z + 1.5 * 0.5)) < 1e-4,
    f"got {encoded[0, 2].item()}",
)

# Test filter with score threshold
coder_filtered = TransFusionBBoxCoder(
    pc_range=[-54.0, -54.0],
    voxel_size=[0.075, 0.075],
    out_size_factor=8,
    post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
    score_threshold=0.5,
    code_size=10,
)
results_f = coder_filtered.decode(
    heatmap.clone(), rot.clone(), dim.clone(), center.clone(), height.clone(), vel.clone(),
    filter=True,
)
pred_f = results_f[0]
# With score_threshold=0.5 and only one peak with sigmoid(10)≈1.0, should get exactly 1 box
check(
    "decode filter keeps 1 high-score box",
    pred_f["bboxes"].shape[0] == 1,
    f"got {pred_f['bboxes'].shape[0]}",
)
check(
    "filtered box is at peak",
    pred_f["labels"][0].item() == 3,
    f"got label {pred_f['labels'][0].item()}",
)

print()

# ===========================================================================
# Summary
# ===========================================================================
print(f"\n{'='*60}")
print(f"Results: {passed} passed, {failed} failed")
print(f"{'='*60}")

if failed > 0:
    sys.exit(1)
else:
    print("All tests passed!")
    sys.exit(0)
