"""HungarianAssigner3D + match costs (FocalLoss / BBoxBEVL1 / BBox3DL1 / IoU3D)
for the TransFusion head. build_bbox_assigner/build_cost resolve config type strings.
"""

import torch

from .sampler import AssignResult

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None

__all__ = [
    "HungarianAssigner3D",
    "FocalLossCost",
    "BBoxBEVL1Cost",
    "IoU3DCost",
    "BBox3DL1Cost",
    "BboxOverlaps3D",
    "build_bbox_assigner",
    "build_cost",
]


class FocalLossCost:
    """FocalLossCost.

    Args:
        weight (int | float, optional): loss_weight
        alpha (int | float, optional): focal_loss alpha
        gamma (int | float, optional): focal_loss gamma
        eps (float, optional): default 1e-12
    """

    def __init__(self, weight=1., alpha=0.25, gamma=2, eps=1e-12):
        self.weight = weight
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def __call__(self, cls_pred, gt_labels):
        """
        Args:
            cls_pred (Tensor): Predicted classification logits, shape
                [num_query, num_class].
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).

        Returns:
            torch.Tensor: cls_cost value with weight
        """
        cls_pred = cls_pred.sigmoid()
        neg_cost = -(1 - cls_pred + self.eps).log() * (
            1 - self.alpha) * cls_pred.pow(self.gamma)
        pos_cost = -(cls_pred + self.eps).log() * self.alpha * (
            1 - cls_pred).pow(self.gamma)
        cls_cost = pos_cost[:, gt_labels] - neg_cost[:, gt_labels]
        return cls_cost * self.weight


class BBoxBEVL1Cost:
    """BBoxBEVL1Cost for BEV center L1 distance.

    Args:
        weight (int | float, optional): loss_weight
    """

    def __init__(self, weight=1.0):
        self.weight = weight

    def __call__(self, bboxes, gt_bboxes, train_cfg=None):
        """
        Args:
            bboxes (Tensor or box object): Predicted boxes with shape [num_query, D].
            gt_bboxes (Tensor or box object): Ground truth boxes with shape [num_gt, D].
            train_cfg (dict, optional): Training config with point_cloud_range.

        Returns:
            torch.Tensor: bbox_cost value with weight
        """
        # Extract tensor from box objects if needed
        if hasattr(bboxes, 'tensor'):
            bboxes = bboxes.tensor
        if hasattr(gt_bboxes, 'tensor'):
            gt_bboxes = gt_bboxes.tensor

        if train_cfg is not None and 'point_cloud_range' in train_cfg:
            pc_start = bboxes.new_tensor(train_cfg['point_cloud_range'][0:2])
            pc_range = bboxes.new_tensor(
                train_cfg['point_cloud_range'][3:5]) - bboxes.new_tensor(
                    train_cfg['point_cloud_range'][0:2])
            # normalize the box center to [0, 1]
            normalized_bboxes_xy = (bboxes[:, :2] - pc_start) / pc_range
            normalized_gt_bboxes_xy = (gt_bboxes[:, :2] - pc_start) / pc_range
        else:
            # If no train_cfg, use raw coordinates
            normalized_bboxes_xy = bboxes[:, :2]
            normalized_gt_bboxes_xy = gt_bboxes[:, :2]

        reg_cost = torch.cdist(normalized_bboxes_xy, normalized_gt_bboxes_xy, p=1)
        return reg_cost * self.weight


class BBox3DL1Cost:
    """BBox3DL1Cost.

    Args:
        weight (int | float, optional): loss_weight
    """

    def __init__(self, weight=1.0):
        self.weight = weight

    def __call__(self, bbox_pred, gt_bboxes):
        """
        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                [num_query, 4].
            gt_bboxes (Tensor): Ground truth boxes with normalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].

        Returns:
            torch.Tensor: bbox_cost value with weight
        """
        bbox_cost = torch.cdist(bbox_pred, gt_bboxes, p=1)
        return bbox_cost * self.weight


class IoU3DCost:
    """IoU3DCost.

    Args:
        weight (int | float, optional): loss_weight
    """

    def __init__(self, weight=1.0):
        self.weight = weight

    def __call__(self, iou):
        """
        Args:
            iou (Tensor): IoU matrix of shape [num_query, num_gt].

        Returns:
            torch.Tensor: iou_cost value with weight
        """
        iou_cost = -iou
        return iou_cost * self.weight


class BboxOverlaps3D:
    """3D IoU Calculator.

    Args:
        coordinate (str): The coordinate system, valid options are
            'camera', 'lidar', and 'depth'.
    """

    def __init__(self, coordinate='lidar'):
        assert coordinate in ["camera", "lidar", "depth"]
        self.coordinate = coordinate

    def __call__(self, bboxes1, bboxes2, mode="iou"):
        """Calculate 3D IoU.

        Args:
            bboxes1 (torch.Tensor): shape (N, 7+C) [x, y, z, h, w, l, ry].
            bboxes2 (torch.Tensor): shape (M, 7+C) [x, y, z, h, w, l, ry].
            mode (str): "iou" (intersection over union) or
                iof (intersection over foreground).

        Return:
            torch.Tensor: Bbox overlaps results of bboxes1 and bboxes2
                with shape (N, M) (aligned mode is not supported currently).
        """
        # Lazy import: datasets.geometry depends on ops, avoid import-time cost
        try:
            from datasets.geometry.boxes import LiDARInstance3DBoxes
        except (ImportError, AttributeError):
            # Fallback: assume bboxes1/bboxes2 are already box objects with .overlaps
            if hasattr(bboxes1, 'overlaps'):
                return bboxes1.overlaps(bboxes1, bboxes2, mode=mode)
            else:
                raise RuntimeError(
                    "datasets.geometry.boxes not available and bboxes don't have .overlaps method")

        # Convert tensors to box objects and call overlaps
        bboxes1_obj = LiDARInstance3DBoxes(bboxes1, box_dim=bboxes1.shape[-1])
        bboxes2_obj = LiDARInstance3DBoxes(bboxes2, box_dim=bboxes2.shape[-1])
        return bboxes1_obj.overlaps(bboxes1_obj, bboxes2_obj, mode=mode)

    def __repr__(self):
        """str: return a string that describes the module"""
        repr_str = self.__class__.__name__
        repr_str += f"(coordinate={self.coordinate})"
        return repr_str


class HungarianAssigner3D:
    """Computes one-to-one matching between predictions and ground truth.

    This class computes an assignment between the targets and the predictions
    based on the costs. The costs are weighted sum of three components:
    classification cost, regression L1 cost and regression iou cost. The
    targets don't include the no_object, so generally there are more
    predictions than targets. After the one-to-one matching, the un-matched
    are treated as backgrounds. Thus each query prediction will be assigned
    with `0` or a positive integer indicating the ground truth index:
    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        cls_cost (dict): Classification cost config.
        reg_cost (dict): Regression L1 cost config.
        iou_cost (dict): Regression iou cost config.
        iou_calculator (dict): IoU calculator config.
    """

    def __init__(
        self,
        cls_cost=dict(type="FocalLossCost", weight=1.0),
        reg_cost=dict(type="BBoxBEVL1Cost", weight=1.0),
        iou_cost=dict(type="IoU3DCost", weight=1.0),
        iou_calculator=dict(type="BboxOverlaps3D"),
    ):
        self.cls_cost = build_cost(cls_cost)
        self.reg_cost = build_cost(reg_cost)
        self.iou_cost = build_cost(iou_cost)
        self.iou_calculator = build_cost(iou_calculator)

    def assign(self, bboxes, gt_bboxes, gt_labels, cls_pred, train_cfg=None, eps=1e-7):
        """Computes one-to-one matching based on the weighted costs.

        This method assign each query prediction to a ground truth or
        background. The `assigned_gt_inds` with -1 means don't care,
        0 means negative sample, and positive number is the index (1-based)
        of assigned gt.

        Args:
            bboxes (Tensor or box object): Predicted boxes with shape [num_query, D].
            gt_bboxes (Tensor or box object): Ground truth boxes with shape [num_gt, D].
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).
            cls_pred (Tensor): Predicted classification logits, shape
                [1, num_class, num_query] or [num_query, num_class].
            train_cfg (dict, optional): Training config.
            eps (int | float, optional): A value added to the denominator for
                numerical stability. Default 1e-7.

        Returns:
            :obj:`AssignResult`: The assigned result.
        """
        # Handle box objects that have .tensor attribute
        if hasattr(bboxes, 'tensor'):
            bboxes_tensor = bboxes.tensor
        else:
            bboxes_tensor = bboxes

        if hasattr(gt_bboxes, 'tensor'):
            gt_bboxes_tensor = gt_bboxes.tensor
        else:
            gt_bboxes_tensor = gt_bboxes

        num_gts, num_bboxes = gt_bboxes_tensor.size(0), bboxes_tensor.size(0)

        # 1. assign -1 by default
        assigned_gt_inds = bboxes_tensor.new_full((num_bboxes,),
                                           -1,
                                           dtype=torch.long)
        assigned_labels = bboxes_tensor.new_full((num_bboxes,),
                                          -1,
                                          dtype=torch.long)
        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)

        # 2. compute the weighted costs
        # Handle cls_pred shape: [1, num_class, num_query] -> [num_query, num_class]
        if cls_pred.dim() == 3:
            cls_pred = cls_pred[0].T

        cls_cost = self.cls_cost(cls_pred, gt_labels)
        reg_cost = self.reg_cost(bboxes, gt_bboxes, train_cfg)
        iou = self.iou_calculator(bboxes, gt_bboxes)
        iou_cost = self.iou_cost(iou)

        # weighted sum of above three costs
        cost = cls_cost + reg_cost + iou_cost

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError(
                'Please run "pip install scipy" ' "to install scipy first.")
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(bboxes_tensor.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(bboxes_tensor.device)

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]

        max_overlaps = torch.zeros_like(iou.max(1).values)
        max_overlaps[matched_row_inds] = iou[matched_row_inds, matched_col_inds]
        return AssignResult(
            num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)


# Cost type registry
COST_REGISTRY = {
    "FocalLossCost": FocalLossCost,
    "BBoxBEVL1Cost": BBoxBEVL1Cost,
    "IoU3DCost": IoU3DCost,
    "BBox3DL1Cost": BBox3DL1Cost,
    "BboxOverlaps3D": BboxOverlaps3D,
}


def build_cost(cfg):
    """Build a cost or cost calculator from config dict.

    Args:
        cfg (dict): Config dict with 'type' key and other kwargs.

    Returns:
        Cost object instance.
    """
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be a dict")
    cfg = cfg.copy()
    cost_type = cfg.pop("type")
    if cost_type not in COST_REGISTRY:
        raise KeyError(f"Unknown cost type: {cost_type}")
    return COST_REGISTRY[cost_type](**cfg)


def build_bbox_assigner(cfg):
    """Build a bbox assigner from config dict.

    Args:
        cfg (dict): Config dict with 'type' key and other kwargs.

    Returns:
        Assigner object instance.
    """
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be a dict")
    cfg = cfg.copy()
    assigner_type = cfg.pop("type")
    if assigner_type == "HungarianAssigner3D":
        return HungarianAssigner3D(**cfg)
    else:
        raise KeyError(f"Unknown assigner type: {assigner_type}")
