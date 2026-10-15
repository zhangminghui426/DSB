"""Preprocessing — normalization, tensor formatting and final key collection."""

import numpy as np
import torch
import torchvision

from datasets.geometry.boxes import LiDARInstance3DBoxes
from datasets.geometry.points import LiDARPoints

class ImageNormalize:
    """Normalize images using torchvision."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.compose = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, data):
        data["img"] = [self.compose(img) for img in data["img"]]
        data["img_norm_cfg"] = dict(mean=self.mean, std=self.std)
        return data


class DefaultFormatBundle3D:
    """Format 3D data — convert points to tensor, stack images, etc."""

    def __init__(self, classes, with_gt=True, with_label=True):
        self.class_names = classes
        self.with_gt = with_gt
        self.with_label = with_label

    def __call__(self, results):
        # Points -> tensor
        if "points" in results:
            if isinstance(results["points"], LiDARPoints):
                results["points"] = results["points"].tensor

        # Images -> stack into tensor
        if "img" in results:
            if isinstance(results["img"], list):
                results["img"] = torch.stack(results["img"])

        # GT labels
        if self.with_label:
            if "gt_names" in results and isinstance(results["gt_names"], np.ndarray):
                results["gt_labels"] = np.array(
                    [
                        self.class_names.index(n) if n in self.class_names else -1
                        for n in results["gt_names"]
                    ],
                    dtype=np.int64,
                )

        # Convert numpy arrays to tensors
        for key in [
            "gt_labels_3d",
            "gt_labels",
            "attr_labels",
            "depths",
        ]:
            if key in results:
                if isinstance(results[key], np.ndarray):
                    results[key] = torch.from_numpy(results[key])
                elif isinstance(results[key], list):
                    results[key] = [
                        torch.from_numpy(r) if isinstance(r, np.ndarray) else r
                        for r in results[key]
                    ]

        return results


class Collect3D:
    """Collect final keys and build metas dict."""

    def __init__(
        self,
        keys,
        meta_keys=(
            "camera_intrinsics",
            "camera2ego",
            "img_aug_matrix",
            "lidar_aug_matrix",
        ),
        meta_lis_keys=(
            "filename",
            "timestamp",
            "ori_shape",
            "img_shape",
            "lidar2image",
            "pad_shape",
            "scale_factor",
            "box_mode_3d",
            "box_type_3d",
            "img_norm_cfg",
            "token",
            "lidar_path",
        ),
    ):
        self.keys = keys
        self.meta_keys = meta_keys
        self.meta_lis_keys = meta_lis_keys

    def __call__(self, results):
        data = {}

        # Collect main keys (convert np arrays to tensors)
        for key in self.keys:
            if key not in self.meta_keys and key in results:
                val = results[key]
                if isinstance(val, np.ndarray):
                    val = torch.from_numpy(val)
                data[key] = val

        # Collect meta keys as tensors
        for key in self.meta_keys:
            if key in results:
                val = np.array(results[key])
                data[key] = torch.from_numpy(val).float()

        # Build metas dict
        metas = {}
        for key in self.meta_lis_keys:
            if key in results:
                metas[key] = results[key]
        # Add box_type_3d class reference
        if "box_type_3d" not in metas:
            metas["box_type_3d"] = LiDARInstance3DBoxes
        data["metas"] = metas

        return data


class PackOcc3DInputs:
    """Collect model inputs and metas into the flat sample dict consumed by
    datasets.dataloader.collate_fn and FlashOCC.forward.

    img_inputs stays a 7-tuple of per-sample tensors; collate transposes and
    stacks it (mmengine default_collate tuple semantics).
    """

    def __init__(self, keys, meta_keys=()):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        sample = {key: results[key] for key in self.keys if key in results}
        sample["metas"] = {key: results[key] for key in self.meta_keys if key in results}
        return sample
