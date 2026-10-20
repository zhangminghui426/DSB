"""Preprocessing — range / class filters for points and GT boxes."""

import numpy as np

class PointsRangeFilter:
    """Filter points by 3D range."""

    def __init__(self, point_cloud_range):
        self.pcd_range = np.array(point_cloud_range, dtype=np.float32)

    def __call__(self, data):
        points = data["points"]
        points_mask = points.in_range_3d(self.pcd_range)
        data["points"] = points[points_mask]
        return data


class ObjectRangeFilter:
    """Filter objects by BEV range."""

    def __init__(self, point_cloud_range):
        self.pcd_range = np.array(point_cloud_range, dtype=np.float32)

    def __call__(self, data):
        bev_range = self.pcd_range[[0, 1, 3, 4]]
        gt_bboxes_3d = data["gt_bboxes_3d"]
        gt_labels_3d = data["gt_labels_3d"]
        mask = gt_bboxes_3d.in_range_bev(bev_range)
        gt_bboxes_3d = gt_bboxes_3d[mask]
        mask_np = mask.numpy().astype(bool)
        gt_labels_3d = gt_labels_3d[mask_np]

        gt_bboxes_3d.limit_yaw(offset=0.5, period=2 * np.pi)
        data["gt_bboxes_3d"] = gt_bboxes_3d
        data["gt_labels_3d"] = gt_labels_3d
        return data


class ObjectNameFilter:
    """Filter GT objects by their names/labels."""

    def __init__(self, classes):
        self.classes = classes
        self.labels = list(range(len(self.classes)))

    def __call__(self, data):
        gt_labels_3d = data["gt_labels_3d"]
        gt_bboxes_mask = np.array(
            [n in self.labels for n in gt_labels_3d], dtype=bool
        )
        data["gt_bboxes_3d"] = data["gt_bboxes_3d"][gt_bboxes_mask]
        data["gt_labels_3d"] = data["gt_labels_3d"][gt_bboxes_mask]
        return data
