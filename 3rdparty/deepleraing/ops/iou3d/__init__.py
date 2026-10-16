"""IoU3D ops — 3D box IoU computation.

Pure PyTorch fallback for CPU; CUDA kernel version optional.
"""

import torch


def boxes_overlap_bev_gpu(boxes_a, boxes_b, output):
    """BEV overlap computation (CUDA kernel stub).

    Pure PyTorch fallback: compute axis-aligned BEV overlap.
    """
    N = boxes_a.shape[0]
    M = boxes_b.shape[0]

    for i in range(N):
        for j in range(M):
            # Simplified: axis-aligned overlap (ignores rotation)
            axmin, aymin, axmax, aymax = boxes_a[i, :4]
            bxmin, bymin, bxmax, bymax = boxes_b[j, :4]

            overlap_x = max(0, min(axmax, bxmax) - max(axmin, bxmin))
            overlap_y = max(0, min(aymax, bymax) - max(aymin, bymin))
            output[i, j] = overlap_x * overlap_y


def nms_gpu(boxes, scores, thresh, pre_maxsize=None, post_max_size=None):
    """NMS on GPU (CUDA kernel stub).

    Pure PyTorch fallback.
    """
    if pre_maxsize is not None:
        order = scores.argsort(descending=True)[:pre_maxsize]
        boxes = boxes[order]
        scores = scores[order]

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort(descending=True)

    keep = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)

        if order.numel() == 1:
            break

        xx1 = torch.max(x1[i], x1[order[1:]])
        yy1 = torch.max(y1[i], y1[order[1:]])
        xx2 = torch.min(x2[i], x2[order[1:]])
        yy2 = torch.min(y2[i], y2[order[1:]])

        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = torch.where(ovr <= thresh)[0]
        order = order[inds + 1]

    if post_max_size is not None:
        keep = keep[:post_max_size]

    return torch.tensor(keep, device=boxes.device)
