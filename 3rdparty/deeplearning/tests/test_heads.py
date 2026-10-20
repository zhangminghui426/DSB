""" Verification tests for ported heads: TransFusionHead + BEVSegmentationHead.

Run from repo root:
    python tests/test_heads.py
"""

import sys
import os

# Ensure repo root is on sys.path for project imports
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import copy
import torch
import numpy as np


def count_keys(model, ckpt_sd, prefix):
    """ Compare model state_dict against checkpoint keys for a given prefix.

    Returns (num_matched, missing, unexpected, shape_mismatch).
    """
    model_sd = model.state_dict()
    ckpt_keys = set()
    for k in ckpt_sd:
        if k.startswith(prefix):
            ckpt_keys.add(k[len(prefix):])
    model_keys = set(model_sd.keys())

    matched = 0
    shape_mismatch = []
    for k in ckpt_keys & model_keys:
        if model_sd[k].shape == ckpt_sd[prefix + k].shape:
            matched += 1
        else:
            shape_mismatch.append(k)

    missing = model_keys - ckpt_keys
    unexpected = ckpt_keys - model_keys
    return matched, missing, unexpected, shape_mismatch


def test_transfusion_ckpt_gate():
    """ Test (b): checkpoint key matching for TransFusionHead."""
    from models.heads.transfusion import TransFusionHead

    head_cfg = dict(
        in_channels=512,
        num_proposals=200,
        auxiliary=True,
        hidden_channel=128,
        num_classes=10,
        num_decoder_layers=1,
        num_heads=8,
        nms_kernel_size=3,
        ffn_channel=256,
        dropout=0.1,
        bn_momentum=0.1,
        activation="relu",
        common_heads={
            "center": [2, 2],
            "height": [1, 2],
            "dim": [3, 2],
            "rot": [2, 2],
            "vel": [2, 2],
        },
        bbox_coder=dict(
            type="TransFusionBBoxCoder",
            pc_range=[-54, -54],
            post_center_range=[-61.2, -61.2, -10, 61.2, 61.2, 10],
            score_threshold=0.0,
            out_size_factor=8,
            voxel_size=[0.075, 0.075],
            code_size=10,
        ),
        train_cfg=dict(
            dataset="nuScenes",
            point_cloud_range=[-54, -54, -5, 54, 54, 3],
            grid_size=[1440, 1440, 41],
            voxel_size=[0.075, 0.075, 0.2],
            out_size_factor=8,
            gaussian_overlap=0.1,
            min_radius=2,
            pos_weight=-1,
            code_weights=[1, 1, 1, 1, 1, 1, 1, 1, 0.2, 0.2],
            assigner=dict(
                type="HungarianAssigner3D",
                iou_calculator=dict(type="BboxOverlaps3D", coordinate="lidar"),
                cls_cost=dict(type="FocalLossCost", gamma=2.0, alpha=0.25, weight=0.15),
                reg_cost=dict(type="BBoxBEVL1Cost", weight=0.25),
                iou_cost=dict(type="IoU3DCost", weight=0.25),
            ),
        ),
        test_cfg=dict(
            dataset="nuScenes",
            grid_size=[1440, 1440, 1],
            out_size_factor=8,
            voxel_size=[0.075, 0.075],
            pc_range=[-54, -54],
            nms_type=None,
        ),
        loss_cls=dict(
            type="FocalLoss",
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            reduction="mean",
            loss_weight=1.0,
        ),
        loss_heatmap=dict(
            type="GaussianFocalLoss",
            reduction="mean",
            loss_weight=1.0,
        ),
        loss_bbox=dict(
            type="L1Loss",
            reduction="mean",
            loss_weight=0.25,
        ),
    )

    head = TransFusionHead(**head_cfg)

    ckpt_path = os.path.join(REPO, "ckpts", "bevfusion-det.pth")
    ckpt_sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]

    matched, missing, unexpected, shape_mismatch = count_keys(
        head, ckpt_sd, "heads.object."
    )
    print(f"  TransFusionHead ckpt gate: matched={matched}, "
          f"missing={len(missing)}, unexpected={len(unexpected)}, "
          f"shape_mismatch={len(shape_mismatch)}")
    if missing:
        print(f"    MISSING: {sorted(missing)[:10]}...")
    if unexpected:
        print(f"    UNEXPECTED: {sorted(unexpected)[:10]}...")
    if shape_mismatch:
        print(f"    SHAPE_MISMATCH: {shape_mismatch}")
    assert len(missing) == 0, f"missing keys: {missing}"
    assert len(unexpected) == 0, f"unexpected keys: {unexpected}"
    assert len(shape_mismatch) == 0, f"shape mismatch: {shape_mismatch}"
    return head, head_cfg


def test_transfusion_forward(head):
    """ Test (c): forward on GPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    head = head.to(device)
    head.eval()
    with torch.no_grad():
        x = torch.randn(1, 512, 180, 180, device=device)
        result = head(x)
    assert isinstance(result, tuple)
    assert len(result) == 1
    pred = result[0]
    assert isinstance(pred, list)
    assert len(pred) == 1
    d = pred[0]
    expected_keys = {"center", "height", "dim", "rot", "vel", "heatmap",
                     "query_heatmap_score", "dense_heatmap"}
    assert set(d.keys()) >= expected_keys, f"missing keys: {expected_keys - set(d.keys())}"
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            assert torch.isfinite(v).all(), f"non-finite in {k}"
    print(f"  TransFusionHead forward OK, keys={sorted(d.keys())}")
    return result


def test_transfusion_loss(head, head_cfg):
    """ Test (d): loss computation with synthetic GT."""
    from datasets.geometry.boxes import LiDARInstance3DBoxes

    device = next(head.parameters()).device
    head.train()

    # Re-run forward in training mode to get gradients
    x = torch.randn(1, 512, 180, 180, device=device)
    pred_result = head(x)

    # Create synthetic GT boxes (x,y,z,x_size,y_size,z_size,yaw,vx,vy)
    gt_bboxes_list = [
        LiDARInstance3DBoxes(
            torch.tensor([
                [5.0, 3.0, -1.0, 4.0, 2.0, 1.5, 0.3, 1.0, 0.5],
                [10.0, -5.0, -0.5, 3.0, 1.5, 1.0, -0.5, 0.0, 0.0],
                [-8.0, 2.0, -1.5, 5.0, 2.5, 2.0, 1.0, -1.0, 0.3],
                [15.0, 8.0, -0.8, 2.0, 1.0, 0.8, 0.0, 0.5, 0.2],
                [-3.0, -10.0, -1.2, 3.5, 1.8, 1.2, -1.0, 0.3, -0.1],
            ], device=device),
            box_dim=9,
        )
    ]
    gt_labels_list = [
        torch.tensor([0, 1, 2, 3, 4], dtype=torch.long, device=device)
    ]

    # Zero grads before loss
    for p in head.parameters():
        if p.grad is not None:
            p.grad.zero_()

    # Compute loss
    loss_dict = head.loss(gt_bboxes_list, gt_labels_list, pred_result)

    assert isinstance(loss_dict, dict)
    for k, v in loss_dict.items():
        if isinstance(v, torch.Tensor):
            assert torch.isfinite(v).all(), f"non-finite loss {k}: {v}"

    # Test backward
    grad_losses = [v for v in loss_dict.values() if isinstance(v, torch.Tensor) and v.requires_grad]
    if grad_losses:
        total_loss = sum(grad_losses)
        total_loss.backward()
        # Check that some gradient exists
        has_grad = False
        for p in head.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "no gradients computed"
    else:
        print("    WARNING: no grad-requiring losses found")

    print(f"  TransFusionHead loss OK, keys={sorted(loss_dict.keys())}")
    print(f"    loss values: {', '.join(f'{k}={v.item():.4f}' for k,v in loss_dict.items() if isinstance(v, torch.Tensor) and v.dim()==0)}")


def test_transfusion_get_bboxes(head, pred_result):
    """ Test (e): get_bboxes."""
    from datasets.geometry.boxes import LiDARInstance3DBoxes

    result = head.get_bboxes(
        pred_result,
        metas=[{"box_type_3d": LiDARInstance3DBoxes}],
    )
    assert len(result) == 1, f"expected 1 layer, got {len(result)}"
    assert len(result[0]) == 3, f"expected (boxes, scores, labels), got {len(result[0])}"
    boxes, scores, labels = result[0]
    assert isinstance(boxes, LiDARInstance3DBoxes)
    assert scores.shape[0] == boxes.tensor.shape[0]
    assert labels.shape[0] == scores.shape[0]
    print(f"  TransFusionHead get_bboxes OK: "
          f"boxes={boxes.tensor.shape}, scores={scores.shape}, labels={labels.shape}")


def test_segm_ckpt_gate():
    """ Test (f): BEVSegmentationHead checkpoint gate."""
    from models.heads.segm_vanilla import BEVSegmentationHead

    head = BEVSegmentationHead(
        in_channels=512,
        grid_transform=dict(
            input_scope=[[-51.2, 51.2, 0.8], [-51.2, 51.2, 0.8]],
            output_scope=[[-50, 50, 0.5], [-50, 50, 0.5]],
        ),
        classes=["drivable_area", "ped_crossing", "walkway", "stop_line", "carpark_area", "divider"],
        loss="focal",
    )

    ckpt_path = os.path.join(REPO, "ckpts", "bevfusion-seg.pth")
    ckpt_sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]

    matched, missing, unexpected, shape_mismatch = count_keys(
        head, ckpt_sd, "heads.map."
    )
    print(f"  BEVSegmentationHead ckpt gate: matched={matched}, "
          f"missing={len(missing)}, unexpected={len(unexpected)}, "
          f"shape_mismatch={len(shape_mismatch)}")
    if missing:
        print(f"    MISSING: {sorted(missing)[:10]}...")
    if unexpected:
        print(f"    UNEXPECTED: {sorted(unexpected)[:10]}...")
    if shape_mismatch:
        print(f"    SHAPE_MISMATCH: {shape_mismatch}")
    assert len(missing) == 0, f"missing keys: {missing}"
    assert len(unexpected) == 0, f"unexpected keys: {unexpected}"
    assert len(shape_mismatch) == 0, f"shape mismatch: {shape_mismatch}"
    return head


def test_segm_forward_and_loss(head):
    """ Test (f): BEVSegmentationHead forward + loss."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    head = head.to(device)
    head.train()

    # Forward: input (1, 512, 128, 128) — BEV grid for seg config: 102.4/0.8=128
    x = torch.randn(1, 512, 128, 128, device=device)
    # GT masks: (1, 6, 200, 200) — output scope: 100/0.5=200
    gt_masks = torch.randint(0, 2, (1, 6, 200, 200), dtype=torch.long, device=device)

    losses = head(x, target=gt_masks)
    assert isinstance(losses, dict)
    for k, v in losses.items():
        if isinstance(v, torch.Tensor):
            assert torch.isfinite(v).all(), f"non-finite seg loss {k}: {v}"

    total_loss = sum(v for v in losses.values() if isinstance(v, torch.Tensor))
    total_loss.backward()

    print(f"  BEVSegmentationHead forward+loss OK, loss keys={sorted(losses.keys())}")
    print(f"    total_loss={total_loss.item():.4f}")

    # Test eval mode
    head.eval()
    with torch.no_grad():
        out = head(x)
    assert isinstance(out, torch.Tensor)
    assert torch.isfinite(out).all()
    print(f"  BEVSegmentationHead eval OK, output shape={out.shape}")


def main():
    print("=" * 60)
    print("Test heads")
    print("=" * 60)

    # (b) Checkpoint gate for TransFusionHead
    print("\n[1] TransFusionHead ckpt gate...")
    head, head_cfg = test_transfusion_ckpt_gate()
    print("[OK]")

    # (c) Forward on GPU
    print("\n[2] TransFusionHead forward...")
    pred_result = test_transfusion_forward(head)
    print("[OK]")

    # (d) Loss
    print("\n[3] TransFusionHead loss...")
    test_transfusion_loss(head, head_cfg)
    print("[OK]")

    # (e) get_bboxes
    print("\n[4] TransFusionHead get_bboxes...")
    test_transfusion_get_bboxes(head, pred_result)
    print("[OK]")

    # (f) BEVSegmentationHead
    print("\n[5] BEVSegmentationHead ckpt gate...")
    seg_head = test_segm_ckpt_gate()
    print("[OK]")

    print("\n[6] BEVSegmentationHead forward+loss...")
    test_segm_forward_and_loss(seg_head)
    print("[OK]")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
