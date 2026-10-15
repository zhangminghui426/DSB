"""3D IoU / BEV overlap / rotated NMS ops.

Ported from mmdet3d/ops/iou3d/iou3d_utils.py (CUDA extension `iou3d_cuda`).
`boxes_overlap_bev_gpu` takes boxes in XYXYR format (x1, y1, x2, y2, r) and is
used by the Hungarian assigner (via LiDARInstance3DBoxes.overlaps) at train time.
"""

import torch

from . import iou3d_cuda

__all__ = [
    "boxes_iou_bev",
    "boxes_overlap_bev",
    "nms_gpu",
    "nms_normal_gpu",
    "iou3d_cuda",
]


def boxes_overlap_bev(boxes_a, boxes_b):
    """Calculate BEV overlap of two box sets (XYXYR format).

    Args:
        boxes_a (torch.Tensor): (M, 5) [x1, y1, x2, y2, ry].
        boxes_b (torch.Tensor): (N, 5) [x1, y1, x2, y2, ry].

    Returns:
        torch.Tensor: (M, N) intersection areas.
    """
    ans_overlap = boxes_a.new_zeros(torch.Size((boxes_a.shape[0], boxes_b.shape[0])))
    iou3d_cuda.boxes_overlap_bev_gpu(
        boxes_a.contiguous(), boxes_b.contiguous(), ans_overlap
    )
    return ans_overlap


def boxes_iou_bev(boxes_a, boxes_b):
    """Calculate boxes IoU in the bird view.

    Args:
        boxes_a (torch.Tensor): Input boxes a with shape (M, 5).
        boxes_b (torch.Tensor): Input boxes b with shape (N, 5).

    Returns:
        ans_iou (torch.Tensor): IoU result with shape (M, N).
    """
    ans_iou = boxes_a.new_zeros(torch.Size((boxes_a.shape[0], boxes_b.shape[0])))

    iou3d_cuda.boxes_iou_bev_gpu(boxes_a.contiguous(), boxes_b.contiguous(), ans_iou)

    return ans_iou


def nms_gpu(boxes, scores, thresh, pre_maxsize=None, post_max_size=None):
    """Nms function with gpu implementation.

    Args:
        boxes (torch.Tensor): Input boxes with the shape of [N, 5]
            ([x1, y1, x2, y2, ry]).
        scores (torch.Tensor): Scores of boxes with the shape of [N].
        thresh (float): Threshold.
        pre_maxsize (int): Max size of boxes before nms. Default: None.
        post_maxsize (int): Max size of boxes after nms. Default: None.

    Returns:
        torch.Tensor: Indexes after nms.
    """
    order = scores.sort(0, descending=True)[1]

    if pre_maxsize is not None:
        order = order[:pre_maxsize]
    boxes = boxes[order].contiguous()

    keep = torch.zeros(boxes.size(0), dtype=torch.long)
    num_out = iou3d_cuda.nms_gpu(boxes, keep, thresh, boxes.device.index)
    keep = order[keep[:num_out].cuda(boxes.device)].contiguous()
    if post_max_size is not None:
        keep = keep[:post_max_size]
    return keep


def nms_normal_gpu(boxes, scores, thresh):
    """Normal non maximum suppression on GPU.

    Args:
        boxes (torch.Tensor): Input boxes with shape (N, 5).
        scores (torch.Tensor): Scores of predicted boxes with shape (N).
        thresh (float): Threshold of non maximum suppression.

    Returns:
        torch.Tensor: Remaining indices with scores in descending order.
    """
    order = scores.sort(0, descending=True)[1]

    boxes = boxes[order].contiguous()

    keep = torch.zeros(boxes.size(0), dtype=torch.long)
    num_out = iou3d_cuda.nms_normal_gpu(boxes, keep, thresh, boxes.device.index)
    return order[keep[:num_out].cuda(boxes.device)].contiguous()
