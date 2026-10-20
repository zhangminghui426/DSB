"""PyTorch Dataset: 读取 PCD + 标注, 体素化, 生成 anchor 匹配目标。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from configs.default import Config, get_config
from models.second import generate_anchors
from preprocess.voxelize import points_to_voxels
from utils.pcd_reader import read_pcd


class SECONDDataset(Dataset):
    def __init__(self, cfg: Config, split: str = "train"):
        self.cfg = cfg
        self.split = split
        self.voxel_cfg = cfg.voxel
        self.samples = self._load_samples()

        # 预生成 anchors (Dy, Dx, num_rot, 7)
        self.anchors = generate_anchors(
            grid_size=cfg.voxel.grid_size,
            voxel_size=cfg.voxel.voxel_size,
            point_cloud_range=cfg.voxel.point_cloud_range,
            anchor_sizes=cfg.model.anchor_sizes,
            anchor_rotations=cfg.model.anchor_rotations,
        )  # (Dy, Dx, num_rot, 7)
        self.num_rot = len(cfg.model.anchor_rotations)
        # 展平为 (N_anchor, 7)
        self.anchors_flat = self.anchors.reshape(-1, 7)

    def _load_samples(self) -> list[dict]:
        samples_path = Path(self.cfg.dataset.annotation_file).parent / "samples.json"
        with open(samples_path, "r") as f:
            samples = json.load(f)
        # 11 帧: 7:3 划分, 前 8 训练, 后 3 验证
        if self.split == "train":
            return samples[:8]
        else:
            return samples[8:]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        pcd_path = sample["pcd_path"]

        # 读取点云
        arr = read_pcd(pcd_path)
        xyz = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float32)
        rgb_uint = arr["rgb"].astype(np.uint32)
        intensity = ((rgb_uint >> 16) & np.uint32(0xFF)).astype(np.float32) / 255.0
        points = np.concatenate([xyz, intensity[:, None]], axis=1).astype(np.float32)

        # 体素化
        vf, vc, np_pts = points_to_voxels(
            points,
            self.voxel_cfg.voxel_size,
            self.voxel_cfg.point_cloud_range,
            self.voxel_cfg.max_num_points_per_voxel,
            self.voxel_cfg.max_num_voxels,
        )

        # 标注框
        gt_boxes = np.array(sample["gt_boxes"], dtype=np.float32)  # (K, 7)
        gt_labels = np.array(sample["gt_labels"], dtype=np.int64)  # (K,)

        # anchor 匹配
        cls_target, loc_target = self._match_anchors(gt_boxes, gt_labels)

        return {
            "voxel_features": torch.from_numpy(vf),  # (M, T, C)
            "voxel_coords": torch.from_numpy(vc),    # (M, 3)
            "num_points": torch.from_numpy(np_pts),  # (M,)
            "cls_target": cls_target,               # (Dy, Dx, num_rot)
            "loc_target": loc_target,               # (Dy, Dx, num_rot, 7)
            "gt_boxes": torch.from_numpy(gt_boxes),
        }

    def _match_anchors(
        self, gt_boxes: np.ndarray, gt_labels: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """简化的 anchor 匹配: 计算 BEV IoU。

        Returns:
            cls_target: (Dy, Dx, num_rot) 0=负, 1=正, -1=忽略。
            loc_target: (Dy, Dx, num_rot, 7) 编码后的回归目标。
        """
        Dy, Dx, num_rot, _ = self.anchors.shape
        anchors = self.anchors.numpy()  # (Dy, Dx, num_rot, 7)

        # 初始化: 全部负样本
        cls_target = np.zeros((Dy, Dx, num_rot), dtype=np.float32)
        loc_target = np.zeros((Dy, Dx, num_rot, 7), dtype=np.float32)

        for k, gt in enumerate(gt_boxes):
            # 计算每个 anchor 与 GT 的 BEV IoU (2D 轴对齐近似)
            # anchor: [x, y, z, w, l, h, yaw]
            # 简化: 不考虑旋转, 用中心点距离 + 重叠面积
            gt_x, gt_y = gt[0], gt[1]
            gt_w, gt_l = gt[3], gt[4]
            gt_area = gt_w * gt_l

            ax = anchors[..., 0]
            ay = anchors[..., 1]
            aw = anchors[..., 3]
            al = anchors[..., 4]
            a_area = aw * al

            # 重叠宽高
            overlap_w = np.minimum(ax + aw / 2, gt_x + gt_w / 2) - np.maximum(ax - aw / 2, gt_x - gt_w / 2)
            overlap_l = np.minimum(ay + al / 2, gt_y + gt_l / 2) - np.maximum(ay - al / 2, gt_y - gt_l / 2)
            overlap_w = np.maximum(overlap_w, 0)
            overlap_l = np.maximum(overlap_l, 0)
            overlap_area = overlap_w * overlap_l
            iou = overlap_area / (a_area + gt_area - overlap_area + 1e-6)

            # 正样本
            pos_mask = iou >= self.cfg.train.pos_iou_thr
            neg_mask = iou < self.cfg.train.neg_iou_thr

            cls_target[pos_mask] = 1.0
            cls_target[~pos_mask & ~neg_mask] = -1.0  # 忽略中间区域

            # 对正样本计算回归目标 (简化编码: 直接回归偏移)
            # dx = (gt_x - ax) / ax_w, dy = (gt_y - ay) / ay_l, dz = gt_z - az
            # dw = log(gt_w / aw), dl = log(gt_l / al), dh = log(gt_h / ah), dyaw = gt_yaw - a_yaw
            pos_idx = np.where(pos_mask)
            if len(pos_idx[0]) > 0:
                a_pos = anchors[pos_idx]  # (N, 7)
                gt_exp = gt[None, :].repeat(len(pos_idx[0]), axis=0)
                dx = (gt_exp[:, 0] - a_pos[:, 0]) / (a_pos[:, 3] + 1e-6)
                dy = (gt_exp[:, 1] - a_pos[:, 1]) / (a_pos[:, 4] + 1e-6)
                dz = gt_exp[:, 2] - a_pos[:, 2]
                dw = np.log(gt_exp[:, 3] / (a_pos[:, 3] + 1e-6) + 1e-6)
                dl = np.log(gt_exp[:, 4] / (a_pos[:, 4] + 1e-6) + 1e-6)
                dh = np.log(gt_exp[:, 5] / (a_pos[:, 5] + 1e-6) + 1e-6)
                dyaw = gt_exp[:, 6] - a_pos[:, 6]
                loc_target[pos_idx] = np.stack([dx, dy, dz, dw, dl, dh, dyaw], axis=-1)

        return torch.from_numpy(cls_target), torch.from_numpy(loc_target)


def collate_batch(batch: list[dict]) -> dict:
    """自定义 batch: 将不同数量的体素 padding 到统一 M。"""
    max_M = max(b["voxel_features"].shape[0] for b in batch)
    T = batch[0]["voxel_features"].shape[1]
    C = batch[0]["voxel_features"].shape[2]
    B = len(batch)

    vf = torch.zeros(B, max_M, T, C)
    vc = torch.zeros(B, max_M, 3, dtype=torch.int32)
    np_pts = torch.zeros(B, max_M, dtype=torch.int32)
    cls_t = torch.stack([b["cls_target"] for b in batch])
    loc_t = torch.stack([b["loc_target"] for b in batch])

    for i, b in enumerate(batch):
        m = b["voxel_features"].shape[0]
        vf[i, :m] = b["voxel_features"]
        vc[i, :m] = b["voxel_coords"]
        np_pts[i, :m] = b["num_points"]

    return {
        "voxel_features": vf,
        "voxel_coords": vc,
        "num_points": np_pts,
        "cls_target": cls_t,
        "loc_target": loc_t,
    }
