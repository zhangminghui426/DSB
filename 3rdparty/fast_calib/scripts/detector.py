"""标定板检测器接口和 SECOND 实现。"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent.parent
_CALIB_BOARD_DIR = _PROJECT_ROOT / "3rdparty" / "deepleraing" / "models" / "calib_board_detector"
if str(_CALIB_BOARD_DIR) not in sys.path:
    sys.path.insert(0, str(_CALIB_BOARD_DIR))


class CalibrationBoardDetector:
    """标定板检测器基类, 输出 AABB ROI。"""

    def detect(self, pc_xyzi: np.ndarray) -> Tuple[bool, Optional[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]]]:
        raise NotImplementedError("Implement detect() for board detector")


class ROIBoardDetector(CalibrationBoardDetector):
    """手动 ROI 检测器, 直接返回预设的 ROI。"""

    def __init__(self, roi: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]):
        self.roi = roi

    def detect(self, pc_xyzi: np.ndarray):
        return True, self.roi


# ==================== SECOND 检测器 ====================

import torch
import yaml

from models.vfe import VoxelFeatureEncoder
from models.backbone import SpMiddleFPN
from models.rpn import RPN
from models.second import SECOND, generate_anchors


def _points_to_voxels(points, voxel_size, point_cloud_range, max_points_per_voxel, max_num_voxels):
    """硬体素化。"""
    x_min, y_min, z_min, x_max, y_max, z_max = point_cloud_range
    vx, vy, vz = voxel_size
    mask = (
        (points[:, 0] >= x_min) & (points[:, 0] < x_max)
        & (points[:, 1] >= y_min) & (points[:, 1] < y_max)
        & (points[:, 2] >= z_min) & (points[:, 2] < z_max)
    )
    pts = points[mask]
    if pts.shape[0] == 0:
        M = 0
        C = points.shape[1]
        return (
            np.zeros((M, max_points_per_voxel, C), dtype=np.float32),
            np.zeros((M, 3), dtype=np.int32),
            np.zeros((M,), dtype=np.int32),
        )
    ix = np.floor((pts[:, 0] - x_min) / vx).astype(np.int32)
    iy = np.floor((pts[:, 1] - y_min) / vy).astype(np.int32)
    iz = np.floor((pts[:, 2] - z_min) / vz).astype(np.int32)
    grid_dx = int(round((x_max - x_min) / vx))
    grid_dy = int(round((y_max - y_min) / vy))
    voxel_keys = ix * grid_dy * grid_dx + iy * grid_dx + iz
    order = np.argsort(voxel_keys)
    sorted_keys = voxel_keys[order]
    sorted_pts = pts[order]
    diff = np.diff(sorted_keys)
    split_idx = np.where(diff != 0)[0] + 1
    starts = np.concatenate([[0], split_idx])
    ends = np.concatenate([split_idx, [len(sorted_keys)]])
    num_voxels = len(starts)
    C = pts.shape[1]
    voxel_features = np.zeros((num_voxels, max_points_per_voxel, C), dtype=np.float32)
    voxel_coords = np.zeros((num_voxels, 3), dtype=np.int32)
    num_points = np.zeros((num_voxels,), dtype=np.int32)
    for i in range(num_voxels):
        s, e = starts[i], ends[i]
        n = min(e - s, max_points_per_voxel)
        voxel_features[i, :n] = sorted_pts[s : s + n]
        num_points[i] = n
        p0 = sorted_pts[s]
        voxel_coords[i, 0] = int(np.floor((p0[0] - x_min) / vx))
        voxel_coords[i, 1] = int(np.floor((p0[1] - y_min) / vy))
        voxel_coords[i, 2] = int(np.floor((p0[2] - z_min) / vz))
    if num_voxels > max_num_voxels:
        voxel_features = voxel_features[:max_num_voxels]
        voxel_coords = voxel_coords[:max_num_voxels]
        num_points = num_points[:max_num_voxels]
    return voxel_features, voxel_coords, num_points


def _load_config(model_path: str):
    """加载模型配置: 优先用权重旁边同名 yaml, 否则用项目默认配置。"""
    model_path = Path(model_path)
    sidecar_yaml = model_path.with_suffix(".yaml")
    default_yaml = _THIS_DIR.parent / "config" / "second_config.yaml"

    if sidecar_yaml.exists():
        cfg_path = sidecar_yaml
        print(f"  [CONFIG] 使用权重旁边配置: {sidecar_yaml}")
    else:
        cfg_path = default_yaml
        print(f"  [CONFIG] 使用默认配置: {default_yaml}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


class SECONDBoardDetector(CalibrationBoardDetector):
    """SECOND 标定板检测器, 输出 3D 框 ROI。"""

    def __init__(self, model_path: str, score_thr: float = None):
        self.device = torch.device("cpu")

        # 加载配置
        cfg = _load_config(model_path)
        self.voxel_size = tuple(cfg["voxel_size"])
        self.point_cloud_range = tuple(cfg["point_cloud_range"])
        self.max_points_per_voxel = cfg["max_points_per_voxel"]
        self.max_num_voxels = cfg["max_num_voxels"]
        self.grid_size = tuple(cfg["grid_size"])
        self.vfe_filters = tuple(cfg["vfe_filters"])
        self.sparse_filters = tuple(cfg["sparse_filters"])
        self.upsample_filters = tuple(cfg["upsample_filters"])
        self.num_anchor_per_loc = cfg["num_anchor_per_loc"]
        self.num_class = cfg["num_class"]
        self.anchor_sizes = [tuple(s) for s in cfg["anchor_sizes"]]
        self.anchor_rotations = tuple(cfg["anchor_rotations"])
        self.score_thr = score_thr if score_thr is not None else cfg["score_thr"]

        # 构建模型
        self.model = SECOND(
            in_channels=4,
            vfe_filters=self.vfe_filters,
            sparse_filters=self.sparse_filters,
            upsample_filters=self.upsample_filters,
            grid_size=self.grid_size,
            num_anchor_per_loc=self.num_anchor_per_loc,
            num_class=self.num_class,
        ).to(self.device)

        ckpt = torch.load(model_path, map_location=self.device, weights_only=True)

        self.model.load_state_dict(ckpt)
        self.model.eval()

        # 预生成 anchors
        self.anchors = generate_anchors(
            self.grid_size, self.voxel_size, self.point_cloud_range,
            self.anchor_sizes, self.anchor_rotations,
        )  # (Dy, Dx, num_rot, 7)

    def _decode(self, cls_pred, loc_pred):
        """解码最高置信度的框, 返回 (cx, cy, cz, w, l, h, yaw, score)。"""
        num_rot = self.anchors.shape[2]
        Dy, Dx = cls_pred.shape[1], cls_pred.shape[2]
        loc = loc_pred.view(num_rot, 7, Dy, Dx).permute(2, 3, 0, 1).contiguous()
        scores = torch.sigmoid(cls_pred.permute(1, 2, 0).contiguous())

        ax, ay, az = self.anchors[..., 0], self.anchors[..., 1], self.anchors[..., 2]
        aw, al, ah = self.anchors[..., 3], self.anchors[..., 4], self.anchors[..., 5]
        ayaw = self.anchors[..., 6]
        dx, dy, dz, dw, dl, dh, dyaw = loc.unbind(dim=-1)

        pred_x = ax + dx * aw
        pred_y = ay + dy * al
        pred_z = az + dz
        dw = torch.clamp(dw, max=4.0)
        dl = torch.clamp(dl, max=4.0)
        dh = torch.clamp(dh, max=4.0)
        pred_w = aw * torch.exp(dw)
        pred_l = al * torch.exp(dl)
        pred_h = ah * torch.exp(dh)
        pred_yaw = ayaw + dyaw

        boxes = torch.stack([pred_x, pred_y, pred_z, pred_w, pred_l, pred_h, pred_yaw], dim=-1)
        boxes = boxes.cpu().numpy().reshape(-1, 7)
        scores = scores.cpu().numpy().reshape(-1)

        # 过滤不合理的 box
        x_min_pc, y_min_pc, z_min_pc, x_max_pc, y_max_pc, z_max_pc = self.point_cloud_range
        valid_mask = (
            (boxes[:, 0] >= x_min_pc) & (boxes[:, 0] <= x_max_pc)
            & (boxes[:, 1] >= y_min_pc) & (boxes[:, 1] <= y_max_pc)
            & (boxes[:, 2] >= z_min_pc) & (boxes[:, 2] <= z_max_pc)
            & (boxes[:, 3] < 5.0) & (boxes[:, 4] < 5.0) & (boxes[:, 5] < 5.0)
        )

        mask = (scores > self.score_thr) & valid_mask
        if mask.sum() == 0:
            return None
        boxes, scores = boxes[mask], scores[mask]
        best = np.argmax(scores)
        return boxes[best], scores[best]

    def detect(self, pc_xyzi: np.ndarray):
        """检测标定板, 返回 (ok, ROI)。

        Args:
            pc_xyzi: (N, 4) [x, y, z, intensity]。

        Returns:
            ok: 是否检测到标定板。
            ROI: ((xmin,xmax),(ymin,ymax),(zmin,zmax)) 或 None。
        """
        if pc_xyzi.shape[0] < 100:
            return False, None

        # 只取 x, y, z, intensity 4 维
        pc_xyzi = pc_xyzi[:, :4].copy()

        # intensity 归一化
        inten = pc_xyzi[:, 3]
        if inten.max() > 1.5:
            pc_xyzi[:, 3] = inten / 255.0

        # 体素化
        vf, vc, np_pts = _points_to_voxels(
            pc_xyzi,
            voxel_size=self.voxel_size,
            point_cloud_range=self.point_cloud_range,
            max_points_per_voxel=self.max_points_per_voxel,
            max_num_voxels=self.max_num_voxels,
        )
        if vf.shape[0] == 0:
            return False, None

        vf_t = torch.from_numpy(vf).unsqueeze(0).to(self.device)
        vc_t = torch.from_numpy(vc).unsqueeze(0).to(self.device)
        np_t = torch.from_numpy(np_pts).unsqueeze(0).to(self.device)

        with torch.no_grad():
            cls_pred, loc_pred = self.model(
                vf_t, vc_t, np_t, self.voxel_size, self.point_cloud_range,
            )

        result = self._decode(cls_pred[0], loc_pred[0])
        if result is None:
            return False, None

        box, score = result
        cx, cy, cz, w, l, h, yaw = box

        xmin = cx - w / 2
        xmax = cx + w / 2
        ymin = cy - l / 2
        ymax = cy + l / 2
        zmin = cz - h / 2
        zmax = cz + h / 2

        roi = (
            (float(xmin), float(xmax)),
            (float(ymin), float(ymax)),
            (float(zmin), float(zmax)),
        )
        return True, roi
