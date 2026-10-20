""" circle_nms (numba-jitted) for detection post-processing.
"""

import numba
import numpy as np


@numba.jit(nopython=True)
def circle_nms(dets, thresh, post_max_size=83):
    """ Circular NMS.

    An object is only counted as positive if no other center
    with a higher confidence exists within a radius r using a
    bird-eye view distance metric.

    Args:
        dets (np.ndarray): Detection results with the shape of [N, 3]
            (x, y, score).
        thresh (float): Value of threshold.
        post_max_size (int): Max number of prediction to be kept.

    Returns:
        list: Indexes of the detections to be kept.
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    scores = dets[:, 2]
    order = scores.argsort()[::-1].astype(np.int32)  # highest->lowest
    ndets = dets.shape[0]
    suppressed = np.zeros((ndets), dtype=np.int32)
    keep = []
    for _i in range(ndets):
        i = order[_i]  # start with highest score box
        if suppressed[i] == 1:  # if any box have enough iou with this, remove it
            continue
        keep.append(i)
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            # calculate center distance between i and j box
            dist = (x1[i] - x1[j]) ** 2 + (y1[i] - y1[j]) ** 2

            if dist <= thresh:
                suppressed[j] = 1
    return keep[:post_max_size]
