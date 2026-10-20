""" Unit tests for models.core.sampler, models.core.assigner, and models.losses."""

import os
import sys
import torch
import torch.nn.functional as F

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.losses import (
    FocalLoss,
    GaussianFocalLoss,
    L1Loss,
    SmoothL1Loss,
    py_sigmoid_focal_loss,
)
from models.core.sampler import AssignResult, PseudoSampler, SamplingResult
from models.core.assigner import (
    HungarianAssigner3D,
    FocalLossCost,
    BBoxBEVL1Cost,
    IoU3DCost,
)


def test_focal_loss_basic():
    """ Test FocalLoss with one-hot targets."""
    print("Testing FocalLoss basic...")
    loss_fn = FocalLoss(use_sigmoid=True, gamma=2.0, alpha=0.25)

    # Pred logits (N, C)
    pred = torch.tensor([[2.0, -1.0, 0.5],
                         [-0.5, 1.5, -2.0]])
    # Target as integer labels (N,)
    target = torch.tensor([0, 1])

    loss = loss_fn(pred, target)
    assert loss.ndim == 0, "Loss should be scalar"
    assert torch.isfinite(loss), "Loss should be finite"
    print(f"  FocalLoss with int labels: {loss.item():.4f}")

    # Target as one-hot (N, C) - TransFusionHead style
    target_onehot = torch.tensor([[1.0, 0.0, 0.0],
                                  [0.0, 1.0, 0.0]])
    loss2 = loss_fn(pred, target_onehot)
    assert loss2.ndim == 0, "Loss should be scalar"
    assert torch.isfinite(loss2), "Loss should be finite"
    print(f"  FocalLoss with one-hot: {loss2.item():.4f}")

    # Manual computation for verification
    pred_sigmoid = pred.sigmoid()
    pt = (1 - pred_sigmoid) * target_onehot + pred_sigmoid * (1 - target_onehot)
    focal_weight = (0.25 * target_onehot + 0.75 * (1 - target_onehot)) * pt.pow(2.0)
    bce = F.binary_cross_entropy_with_logits(pred, target_onehot, reduction='none')
    manual_loss = (bce * focal_weight).mean()
    assert torch.allclose(loss2, manual_loss, rtol=1e-4), \
        f"Loss mismatch: {loss2.item()} vs {manual_loss.item()}"
    print(f"  Manual verification passed: {manual_loss.item():.4f}")
    print("[OK] FocalLoss basic")


def test_focal_loss_avg_factor():
    """ Test FocalLoss with avg_factor."""
    print("Testing FocalLoss avg_factor...")
    loss_fn = FocalLoss(use_sigmoid=True, gamma=2.0, alpha=0.25)

    pred = torch.randn(10, 5)
    target = torch.randint(0, 5, (10,))

    loss1 = loss_fn(pred, target, avg_factor=10.0)
    loss2 = loss_fn(pred, target, avg_factor=20.0)

    # With larger avg_factor, loss should be smaller
    assert loss2 < loss1, "Larger avg_factor should give smaller loss"
    ratio = loss1.item() / loss2.item()
    assert abs(ratio - 2.0) < 0.01, f"Ratio should be ~2.0, got {ratio}"
    print(f"  avg_factor=10: {loss1.item():.4f}, avg_factor=20: {loss2.item():.4f}")
    print("[OK] FocalLoss avg_factor")


def test_focal_loss_weight():
    """ Test FocalLoss with sample weights."""
    print("Testing FocalLoss weight...")
    loss_fn = FocalLoss(use_sigmoid=True, gamma=2.0, alpha=0.25)

    pred = torch.randn(5, 3)
    target = torch.randint(0, 3, (5,))
    weight = torch.tensor([1.0, 1.0, 0.0, 1.0, 1.0])

    loss_weighted = loss_fn(pred, target, weight=weight)
    loss_unweighted = loss_fn(pred, target)

    # Weighted loss should be different
    assert not torch.allclose(loss_weighted, loss_unweighted), \
        "Weighted and unweighted losses should differ"
    print(f"  Weighted: {loss_weighted.item():.4f}, Unweighted: {loss_unweighted.item():.4f}")
    print("[OK] FocalLoss weight")


def test_gaussian_focal_loss():
    """ Test GaussianFocalLoss."""
    print("Testing GaussianFocalLoss...")
    loss_fn = GaussianFocalLoss(alpha=2.0, gamma=4.0)

    pred = torch.tensor([0.9, 0.1, 0.5, 0.8])
    target = torch.tensor([1.0, 0.0, 0.3, 1.0])

    loss = loss_fn(pred, target)
    assert loss.ndim == 0, "Loss should be scalar"
    assert torch.isfinite(loss), "Loss should be finite"

    # Manual computation
    eps = 1e-12
    pos_weights = target.eq(1)
    neg_weights = (1 - target).pow(4.0)
    pos_loss = -(pred + eps).log() * (1 - pred).pow(2.0) * pos_weights
    neg_loss = -(1 - pred + eps).log() * pred.pow(2.0) * neg_weights
    manual_loss = (pos_loss + neg_loss).mean()

    assert torch.allclose(loss, manual_loss, rtol=1e-4), \
        f"Loss mismatch: {loss.item()} vs {manual_loss.item()}"
    print(f"  GaussianFocalLoss: {loss.item():.4f}")
    print("[OK] GaussianFocalLoss")


def test_l1_loss():
    """ Test L1Loss."""
    print("Testing L1Loss...")
    loss_fn = L1Loss()

    pred = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.5, 2.5, 3.5])

    loss = loss_fn(pred, target)
    expected = F.l1_loss(pred, target)

    assert torch.allclose(loss, expected, rtol=1e-5), \
        f"L1Loss mismatch: {loss.item()} vs {expected.item()}"
    print(f"  L1Loss: {loss.item():.4f}")
    print("[OK] L1Loss")


def test_smooth_l1_loss():
    """ Test SmoothL1Loss."""
    print("Testing SmoothL1Loss...")
    loss_fn = SmoothL1Loss(beta=1.0)

    pred = torch.tensor([1.0, 2.0, 3.0, 10.0])
    target = torch.tensor([1.5, 2.5, 5.0, 10.5])

    loss = loss_fn(pred, target)
    expected = F.smooth_l1_loss(pred, target, beta=1.0)

    assert torch.allclose(loss, expected, rtol=1e-5), \
        f"SmoothL1Loss mismatch: {loss.item()} vs {expected.item()}"
    print(f"  SmoothL1Loss: {loss.item():.4f}")
    print("[OK] SmoothL1Loss")


def test_pseudo_sampler():
    """ Test PseudoSampler."""
    print("Testing PseudoSampler...")
    sampler = PseudoSampler()

    # Create assign result
    num_preds = 10
    num_gts = 3
    gt_inds = torch.tensor([1, 2, 0, 3, 0, 0, 0, 0, 0, 0])  # 1-based, 0=neg
    max_overlaps = torch.tensor([0.9, 0.8, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    labels = torch.tensor([0, 1, 0, 2, 0, 0, 0, 0, 0, 0])

    assign_result = AssignResult(num_gts, gt_inds, max_overlaps, labels)

    # Create dummy boxes
    bboxes = torch.randn(num_preds, 4)
    gt_bboxes = torch.randn(num_gts, 4)

    sampling_result = sampler.sample(assign_result, bboxes, gt_bboxes)

    assert isinstance(sampling_result, SamplingResult)
    assert len(sampling_result.pos_inds) == 3, \
        f"Expected 3 positive samples, got {len(sampling_result.pos_inds)}"
    assert len(sampling_result.neg_inds) == 7, \
        f"Expected 7 negative samples, got {len(sampling_result.neg_inds)}"
    print(f"  pos_inds: {sampling_result.pos_inds}")
    print(f"  neg_inds: {sampling_result.neg_inds}")
    print("[OK] PseudoSampler")


def test_assign_result_add_gt():
    """ Test AssignResult.add_gt_."""
    print("Testing AssignResult.add_gt_...")
    num_preds = 5
    num_gts = 3
    gt_inds = torch.tensor([1, 0, 2, 0, 3])
    max_overlaps = torch.tensor([0.9, 0.0, 0.8, 0.0, 0.7])
    labels = torch.tensor([0, 0, 1, 0, 2])

    assign_result = AssignResult(num_gts, gt_inds, max_overlaps, labels)
    assert len(assign_result.gt_inds) == num_preds

    # Add gt as proposals
    gt_labels = torch.tensor([0, 1, 2])
    assign_result.add_gt_(gt_labels)

    assert len(assign_result.gt_inds) == num_preds + num_gts, \
        f"Expected {num_preds + num_gts} after add_gt_, got {len(assign_result.gt_inds)}"
    # First 3 should be 1, 2, 3 (gt inds)
    assert torch.equal(assign_result.gt_inds[:num_gts], torch.tensor([1, 2, 3])), \
        f"First {num_gts} should be [1,2,3]"
    print(f"  After add_gt_: {assign_result.gt_inds}")
    print("[OK] AssignResult.add_gt_")


class MinimalBoxes3D:
    """ Minimal stub for LiDARInstance3DBoxes for testing."""

    def __init__(self, tensor, box_dim=9):
        self.tensor = tensor
        self.box_dim = box_dim

    @property
    def bev(self):
        """ Return BEV boxes in XYWHR format (N, 5)."""
        # Assume tensor is (N, 9): [x, y, z, w, l, h, rot, vx, vy]
        # BEV: [x, y, w, l, rot]
        return self.tensor[:, [0, 1, 3, 4, 6]]

    @classmethod
    def overlaps(cls, boxes1, boxes2, mode='iou'):
        """ Calculate 3D IoU using BEV overlaps and height overlaps."""
        from ops import boxes_overlap_bev, boxes_iou_bev

        # Convert to XYXYR format for CUDA ops
        # boxes are [x, y, w, l, rot]
        def bev_to_xyxyr(bev_boxes):
            """ Convert [x, y, w, l, rot] to [x1, y1, x2, y2, rot]."""
            xyxyr = torch.zeros(bev_boxes.shape[0], 5, device=bev_boxes.device)
            xyxyr[:, 0] = bev_boxes[:, 0] - bev_boxes[:, 2] / 2  # x1
            xyxyr[:, 1] = bev_boxes[:, 1] - bev_boxes[:, 3] / 2  # y1
            xyxyr[:, 2] = bev_boxes[:, 0] + bev_boxes[:, 2] / 2  # x2
            xyxyr[:, 3] = bev_boxes[:, 1] + bev_boxes[:, 3] / 2  # y2
            xyxyr[:, 4] = bev_boxes[:, 4]  # rot
            return xyxyr

        bev1_xyxyr = bev_to_xyxyr(boxes1.bev)
        bev2_xyxyr = bev_to_xyxyr(boxes2.bev)

        if mode == 'iou':
            bev_iou = boxes_iou_bev(bev1_xyxyr, bev2_xyxyr)
        else:
            # For 'iof' mode, use overlap and divide by boxes1 area
            bev_overlap = boxes_overlap_bev(bev1_xyxyr, bev2_xyxyr)
            areas1 = (bev1_xyxyr[:, 2] - bev1_xyxyr[:, 0]) * (bev1_xyxyr[:, 3] - bev1_xyxyr[:, 1])
            bev_iou = bev_overlap / areas1.unsqueeze(1).clamp(min=1e-6)

        # Simplified: just return BEV IoU for now (full 3D would need height)
        return bev_iou


def test_hungarian_assigner_3d():
    """ Test HungarianAssigner3D with synthetic data."""
    print("Testing HungarianAssigner3D...")

    # Create assigner with default costs
    assigner = HungarianAssigner3D(
        cls_cost=dict(type="FocalLossCost", gamma=2.0, alpha=0.25, weight=0.15),
        reg_cost=dict(type="BBoxBEVL1Cost", weight=0.25),
        iou_cost=dict(type="IoU3DCost", weight=0.25),
        iou_calculator=dict(type="BboxOverlaps3D"),
    )

    # Create synthetic data
    num_gt = 3
    num_pred = 20
    num_classes = 5
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # GT boxes (x, y, z, w, l, h, rot, vx, vy)
    gt_bboxes = torch.tensor([
        [0.0, 0.0, 0.0, 2.0, 2.0, 1.5, 0.0, 1.0, 0.0],
        [5.0, 5.0, 0.0, 2.0, 2.0, 1.5, 0.5, 0.0, 1.0],
        [10.0, 10.0, 0.0, 2.0, 2.0, 1.5, 1.0, -1.0, 0.0],
    ], device=device)
    gt_labels = torch.tensor([0, 1, 2], device=device)

    # Pred boxes - make some close to GT
    pred_bboxes = torch.randn(num_pred, 9, device=device)
    # Make first 3 predictions close to GT
    pred_bboxes[0] = gt_bboxes[0] + torch.randn(9, device=device) * 0.1
    pred_bboxes[1] = gt_bboxes[1] + torch.randn(9, device=device) * 0.1
    pred_bboxes[2] = gt_bboxes[2] + torch.randn(9, device=device) * 0.1

    # Classification predictions [num_query, num_class]
    cls_pred = torch.randn(num_pred, num_classes, device=device)

    # Wrap boxes in MinimalBoxes3D
    pred_boxes = MinimalBoxes3D(pred_bboxes)
    gt_boxes = MinimalBoxes3D(gt_bboxes)

    # Override iou_calculator to use our stub
    def mock_iou_calc(bboxes1, bboxes2, mode='iou'):
        return MinimalBoxes3D.overlaps(bboxes1, bboxes2, mode=mode)

    assigner.iou_calculator = mock_iou_calc

    # Run assignment
    assign_result = assigner.assign(
        pred_boxes, gt_boxes, gt_labels, cls_pred, train_cfg=None
    )

    # Verify results
    assert isinstance(assign_result, AssignResult)
    assert len(assign_result.gt_inds) == num_pred, \
        f"Expected {num_pred} assignments, got {len(assign_result.gt_inds)}"

    # Count positive assignments (gt_inds > 0)
    num_pos = (assign_result.gt_inds > 0).sum().item()
    print(f"  num_gt: {num_gt}, num_pred: {num_pred}, num_pos_assigned: {num_pos}")

    # Every GT should be assigned (when num_pred >= num_gt)
    assert num_pos == num_gt, \
        f"Expected {num_gt} positive assignments, got {num_pos}"

    # Check that assigned_gt_inds are valid (1-based indices)
    pos_inds = assign_result.gt_inds[assign_result.gt_inds > 0]
    assert pos_inds.min() >= 1 and pos_inds.max() <= num_gt, \
        f"Assigned gt inds should be in [1, {num_gt}], got {pos_inds}"

    # Check labels are assigned for positive samples
    pos_labels = assign_result.labels[assign_result.gt_inds > 0]
    assert torch.all(pos_labels >= 0), "Positive samples should have valid labels"

    # Check cost matrix was finite (implicitly verified if assignment succeeded)
    print(f"  Assigned gt_inds: {assign_result.gt_inds}")
    print(f"  Assigned labels: {assign_result.labels}")
    print("[OK] HungarianAssigner3D")


def test_focal_loss_cost():
    """ Test FocalLossCost."""
    print("Testing FocalLossCost...")
    cost_fn = FocalLossCost(weight=0.15, alpha=0.25, gamma=2.0)

    cls_pred = torch.randn(10, 5)  # [num_query, num_class]
    gt_labels = torch.tensor([0, 1, 2])  # [num_gt]

    cost = cost_fn(cls_pred, gt_labels)
    assert cost.shape == (10, 3), f"Expected shape (10, 3), got {cost.shape}"
    assert torch.all(torch.isfinite(cost)), "Cost should be finite"
    print(f"  Cost shape: {cost.shape}, mean: {cost.mean().item():.4f}")
    print("[OK] FocalLossCost")


def test_bbox_bev_l1_cost():
    """ Test BBoxBEVL1Cost."""
    print("Testing BBoxBEVL1Cost...")
    cost_fn = BBoxBEVL1Cost(weight=0.25)

    # Boxes with (x, y, ...) format
    pred_bboxes = torch.tensor([[0.0, 0.0, 1.0, 1.0],
                                 [5.0, 5.0, 2.0, 2.0]])
    gt_bboxes = torch.tensor([[0.5, 0.5, 1.0, 1.0],
                               [5.5, 5.5, 2.0, 2.0]])

    cost = cost_fn(pred_bboxes, gt_bboxes)
    assert cost.shape == (2, 2), f"Expected shape (2, 2), got {cost.shape}"
    assert torch.all(torch.isfinite(cost)), "Cost should be finite"

    # Diagonal should be smaller (closer matches)
    assert cost[0, 0] < cost[0, 1], "Diagonal should be smaller"
    assert cost[1, 1] < cost[1, 0], "Diagonal should be smaller"
    print(f"  Cost matrix:\n{cost}")
    print("[OK] BBoxBEVL1Cost")


def test_iou_3d_cost():
    """ Test IoU3DCost."""
    print("Testing IoU3DCost...")
    cost_fn = IoU3DCost(weight=0.25)

    # Fake IoU matrix
    iou = torch.tensor([[0.9, 0.1, 0.2],
                         [0.1, 0.8, 0.15],
                         [0.2, 0.15, 0.85]])

    cost = cost_fn(iou)
    assert cost.shape == (3, 3), f"Expected shape (3, 3), got {cost.shape}"
    # Cost should be negative of IoU
    expected = -iou * 0.25
    assert torch.allclose(cost, expected), "Cost should be -iou * weight"
    print(f"  IoU:\n{iou}")
    print(f"  Cost:\n{cost}")
    print("[OK] IoU3DCost")


def main():
    """ Run all tests."""
    print("=" * 60)
    print("Running models.core and losses unit tests")
    print("=" * 60)

    tests = [
        test_focal_loss_basic,
        test_focal_loss_avg_factor,
        test_focal_loss_weight,
        test_gaussian_focal_loss,
        test_l1_loss,
        test_smooth_l1_loss,
        test_pseudo_sampler,
        test_assign_result_add_gt,
        test_focal_loss_cost,
        test_bbox_bev_l1_cost,
        test_iou_3d_cost,
        test_hungarian_assigner_3d,
    ]

    failed = []
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"[FAILED] {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed.append(test.__name__)

    print("=" * 60)
    if failed:
        print(f"FAILED: {len(failed)} tests failed:")
        for name in failed:
            print(f"  - {name}")
        sys.exit(1)
    else:
        print("SUCCESS: All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
