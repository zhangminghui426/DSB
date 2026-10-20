"""dataloader — batch collation + DataLoader construction shared by train/test.

collate_fn dispatches on the sample contract: packed occupancy samples carry
an "img_inputs" 7-tuple (occ path); det/seg samples carry img + matrices.
"""

import numpy as np
import torch
from torch.utils.data import (
    DataLoader,
    DistributedSampler,
    RandomSampler,
    SequentialSampler,
)


def collate_fn(batch):
    """Single collate entry: occ-packed samples vs det/seg samples."""
    if len(batch) == 0:
        return {}
    if "img_inputs" in batch[0]:
        return collate_occ(batch)
    return collate_det_seg(batch)


def collate_det_seg(batch):
    """Collate a batch of samples.

    Args:
        batch: List of sample dicts from dataset __getitem__.

    Returns:
        Collated batch dict with:
        - img: (B, 6, 3, H, W) tensor (stacked)
        - points: list of (N_i, 5) tensors
        - matrices (camera_intrinsics, camera2ego, etc.): (B, 6, 4, 4) tensors
        - gt_bboxes_3d: list of LiDARInstance3DBoxes
        - gt_labels_3d: list of (M_i,) tensors
        - gt_masks_bev: (B, 6, 200, 200) tensor (stacked if present)
        - depths: (B, 6, H, W) tensor (stacked if present)
        - metas: list of dicts
    """
    if len(batch) == 0:
        return {}

    collated = {}
    keys = batch[0].keys()

    for key in keys:
        values = [sample[key] for sample in batch]

        if key == "metas":
            # Keep as list of dicts
            collated[key] = values
        elif key == "gt_bboxes_3d":
            # Keep as list of box objects
            collated[key] = values
        elif key == "gt_labels_3d":
            # Keep as list of tensors
            collated[key] = values
        elif key == "points":
            # Keep as list of tensors
            collated[key] = values
        elif key == "img":
            # Stack images: each is (6, 3, H, W) -> (B, 6, 3, H, W)
            if isinstance(values[0], torch.Tensor):
                collated[key] = torch.stack(values)
            else:
                collated[key] = values
        elif key in (
            "camera_intrinsics", "camera2ego", "lidar2ego",
            "lidar2camera", "camera2lidar", "lidar2image",
            "img_aug_matrix", "lidar_aug_matrix",
        ):
            # Stack matrices
            tensors = []
            for v in values:
                if isinstance(v, torch.Tensor):
                    tensors.append(v.float())
                elif isinstance(v, np.ndarray):
                    tensors.append(torch.from_numpy(v).float())
                else:
                    tensors.append(torch.tensor(v, dtype=torch.float32))
            collated[key] = torch.stack(tensors)
        elif key == "gt_masks_bev":
            # Stack masks: each is (6, 200, 200) -> (B, 6, 200, 200)
            if isinstance(values[0], torch.Tensor):
                collated[key] = torch.stack(values)
            elif isinstance(values[0], np.ndarray):
                tensors = [torch.from_numpy(v).long() for v in values]
                collated[key] = torch.stack(tensors)
            else:
                collated[key] = values
        elif key == "depths":
            # Stack depths: each is (6, H, W) -> (B, 6, H, W)
            if isinstance(values[0], torch.Tensor):
                collated[key] = torch.stack(values)
            elif isinstance(values[0], np.ndarray):
                tensors = [torch.from_numpy(v).float() for v in values]
                collated[key] = torch.stack(tensors)
            else:
                collated[key] = values
        elif isinstance(values[0], torch.Tensor):
            try:
                collated[key] = torch.stack(values)
            except (RuntimeError, TypeError):
                collated[key] = values
        elif isinstance(values[0], np.ndarray):
            try:
                tensors = [torch.from_numpy(v) for v in values]
                collated[key] = torch.stack(tensors)
            except (RuntimeError, TypeError):
                collated[key] = values
        else:
            collated[key] = values

    return collated



def collate_occ(batch):
    """Collate packed samples:
    - img_inputs 7-tuple -> list of 7 stacked (B, ...) tensors
      (mmengine default_collate tuple semantics)
    - fixed-shape GT tensors (gt_depth/voxel_semantics/mask_*) -> stacked
    - metas -> list[B] of dicts
    """
    if len(batch) == 0:
        return {}

    collated = {}
    for key in batch[0].keys():
        values = [sample[key] for sample in batch]
        if key == "metas":
            collated[key] = values
        elif key == "img_inputs":
            collated[key] = [torch.stack(items) for items in zip(*values)]
        elif isinstance(values[0], torch.Tensor):
            try:
                collated[key] = torch.stack(values)
            except (RuntimeError, TypeError):
                collated[key] = values
        else:
            collated[key] = values
    return collated



def build_dataloader(dataset, batch_size, num_workers, distributed=False,
                     shuffle=False, seed=0, drop_last=False):
    """DataLoader with sampler selection (distributed / random / sequential)."""
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle, seed=seed)
    elif shuffle:
        g = torch.Generator()
        g.manual_seed(seed)
        sampler = RandomSampler(dataset, generator=g)
    else:
        sampler = SequentialSampler(dataset)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=drop_last,
    )
