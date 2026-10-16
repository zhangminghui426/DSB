"""推理 + 可视化脚本 (matplotlib 版, 无需 GUI)。

渲染:
  - 点云 (灰色散点)
  - GT 标注框 (绿色线框)
  - 预测框 (红色线框, 置信度高于阈值)

用法:
  python infer_visualize.py [--frame 0] [--score-thr 0.3]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import torch

from configs.default import get_config
from models.second import SECOND, generate_anchors
from preprocess.voxelize import points_to_voxels
from utils.pcd_reader import read_pcd


def decode_predictions(
    cls_pred: torch.Tensor,
    loc_pred: torch.Tensor,
    anchors: torch.Tensor,
    score_thr: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    num_rot = anchors.shape[2]
    Dy, Dx = cls_pred.shape[1], cls_pred.shape[2]
    loc = loc_pred.view(num_rot, 7, Dy, Dx).permute(2, 3, 0, 1).contiguous()
    scores = torch.sigmoid(cls_pred.permute(1, 2, 0).contiguous())

    ax = anchors[..., 0]
    ay = anchors[..., 1]
    az = anchors[..., 2]
    aw = anchors[..., 3]
    al = anchors[..., 4]
    ah = anchors[..., 5]
    ayaw = anchors[..., 6]
    dx, dy, dz, dw, dl, dh, dyaw = loc.unbind(dim=-1)

    pred_x = ax + dx * aw
    pred_y = ay + dy * al
    pred_z = az + dz
    pred_w = aw * torch.exp(dw)
    pred_l = al * torch.exp(dl)
    pred_h = ah * torch.exp(dh)
    pred_yaw = ayaw + dyaw

    boxes = torch.stack([pred_x, pred_y, pred_z, pred_w, pred_l, pred_h, pred_yaw], dim=-1)
    boxes = boxes.cpu().numpy().reshape(-1, 7)
    scores = scores.cpu().numpy().reshape(-1)
    mask = scores > score_thr
    return boxes[mask], scores[mask]


def box_to_corners(box: np.ndarray) -> np.ndarray:
    """[x,y,z,w,l,h,yaw] -> 8 个角点 (8, 3)。"""
    cx, cy, cz, w, l, h, yaw = box
    # 未旋转的 8 角点
    x = w / 2
    y = l / 2
    z = h / 2
    corners = np.array([
        [-x, -y, -z], [x, -y, -z], [x, y, -z], [-x, y, -z],
        [-x, -y, z],  [x, -y, z],  [x, y, z],  [-x, y, z],
    ])
    # 绕 z 轴旋转
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    corners = corners @ R.T + np.array([cx, cy, cz])
    return corners


def draw_box(ax, box: np.ndarray, color: str, linewidth: float = 1.5):
    """画 3D 框的 12 条边。"""
    corners = box_to_corners(box)
    edges = [
        (0,1),(1,2),(2,3),(3,0),
        (4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7),
    ]
    for i, j in edges:
        ax.plot3D(
            [corners[i, 0], corners[j, 0]],
            [corners[i, 1], corners[j, 1]],
            [corners[i, 2], corners[j, 2]],
            color=color, linewidth=linewidth,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--score-thr", type=float, default=0.3)
    parser.add_argument("--top-k", type=int, default=5, help="只画置信度最高的N个框")
    parser.add_argument("--output", type=str, default="output/vis_result.png")
    args = parser.parse_args()

    cfg = get_config()
    device = torch.device("cpu")

    model = SECOND(
        in_channels=4,
        vfe_filters=cfg.model.vfe_num_filters,
        sparse_filters=cfg.model.sparse_num_filters,
        upsample_filters=cfg.model.rpn_upsample_filters,
        grid_size=cfg.voxel.grid_size,
        num_anchor_per_loc=cfg.model.num_anchor_per_loc,
        num_class=cfg.model.num_class,
    ).to(device)
    ckpt_path = Path(cfg.train.output_dir) / "best_model.pth"
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        print(f"加载模型: {ckpt_path}")
    model.eval()

    with open(Path(cfg.dataset.annotation_file).parent / "samples.json") as f:
        samples = json.load(f)
    sample = samples[args.frame]
    print(f"可视化帧: {sample['frame_id']}")

    arr = read_pcd(sample["pcd_path"])
    xyz = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float32)
    rgb_uint = arr["rgb"].astype(np.uint32)
    intensity = ((rgb_uint >> 16) & np.uint32(0xFF)).astype(np.float32) / 255.0
    points = np.concatenate([xyz, intensity[:, None]], axis=1).astype(np.float32)

    vf, vc, np_pts = points_to_voxels(
        points, cfg.voxel.voxel_size, cfg.voxel.point_cloud_range,
        cfg.voxel.max_num_points_per_voxel, cfg.voxel.max_num_voxels,
    )
    vf_t = torch.from_numpy(vf).unsqueeze(0)
    vc_t = torch.from_numpy(vc).unsqueeze(0)
    np_t = torch.from_numpy(np_pts).unsqueeze(0)

    with torch.no_grad():
        cls_pred, loc_pred = model(vf_t, vc_t, np_t, cfg.voxel.voxel_size, cfg.voxel.point_cloud_range)
    cls_pred = cls_pred[0]
    loc_pred = loc_pred[0]

    anchors = generate_anchors(
        cfg.voxel.grid_size, cfg.voxel.voxel_size, cfg.voxel.point_cloud_range,
        cfg.model.anchor_sizes, cfg.model.anchor_rotations,
    )
    pred_boxes, pred_scores = decode_predictions(cls_pred, loc_pred, anchors, args.score_thr)
    gt_boxes = np.array(sample["gt_boxes"])

    # 只保留 top-k 个置信度最高的预测
    if len(pred_boxes) > args.top_k:
        top_idx = np.argsort(pred_scores)[::-1][:args.top_k]
        pred_boxes = pred_boxes[top_idx]
        pred_scores = pred_scores[top_idx]

    print(f"GT 框数: {len(gt_boxes)}, 预测框数 (top-{args.top_k}): {len(pred_boxes)}")
    for i, box in enumerate(gt_boxes):
        cx, cy, cz, w, l, h, yaw = box
        print(f"  GT[{i}]   x=[{cx-w/2:.3f}, {cx+w/2:.3f}]  y=[{cy-l/2:.3f}, {cy+l/2:.3f}]  z=[{cz-h/2:.3f}, {cz+h/2:.3f}]")
    for i, (box, score) in enumerate(zip(pred_boxes, pred_scores)):
        cx, cy, cz, w, l, h, yaw = box
        print(f"  pred[{i}] score={score:.4f}  x=[{cx-w/2:.3f}, {cx+w/2:.3f}]  y=[{cy-l/2:.3f}, {cy+l/2:.3f}]  z=[{cz-h/2:.3f}, {cz+h/2:.3f}]")

    # 画图
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    # 点云 (检测范围内, 抽样减少点数)
    pc_mask = (
        (points[:, 0] >= cfg.voxel.point_cloud_range[0])
        & (points[:, 0] < cfg.voxel.point_cloud_range[3])
        & (points[:, 1] >= cfg.voxel.point_cloud_range[1])
        & (points[:, 1] < cfg.voxel.point_cloud_range[4])
    )
    pc = points[pc_mask, :3]
    if len(pc) > 5000:
        idx = np.random.choice(len(pc), 5000, replace=False)
        pc = pc[idx]
    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c="gray", s=1, alpha=0.5)

    # 在原点画 XYZ 参考坐标轴 (箭头)
    axis_len = 1.5
    origin = np.array([0.0, 0.0, -2.8])
    # X 轴 (红色, 向前)
    ax.quiver(origin[0], origin[1], origin[2],
              axis_len, 0, 0, color="r", arrow_length_ratio=0.15, linewidth=2)
    ax.text(origin[0] + axis_len + 0.1, origin[1], origin[2], "X", color="r", fontsize=14, fontweight="bold")
    # Y 轴 (绿色, 向左)
    ax.quiver(origin[0], origin[1], origin[2],
              0, axis_len, 0, color="g", arrow_length_ratio=0.15, linewidth=2)
    ax.text(origin[0], origin[1] + axis_len + 0.1, origin[2], "Y", color="g", fontsize=14, fontweight="bold")
    # Z 轴 (蓝色, 向上)
    ax.quiver(origin[0], origin[1], origin[2],
              0, 0, axis_len, color="b", arrow_length_ratio=0.15, linewidth=2)
    ax.text(origin[0], origin[1], origin[2] + axis_len + 0.1, "Z", color="b", fontsize=14, fontweight="bold")
    # 原点标记
    ax.scatter([origin[0]], [origin[1]], [origin[2]], c="k", s=30, marker="o")

    # GT 框 (绿色)
    for box in gt_boxes:
        draw_box(ax, box, "g", linewidth=2.5)

    # 预测框 (红色)
    for box, score in zip(pred_boxes, pred_scores):
        draw_box(ax, box, "r", linewidth=1.0)

    ax.set_xlabel("X forward (m)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Y left (m)", fontsize=12, fontweight="bold")
    ax.set_zlabel("Z up (m)", fontsize=12, fontweight="bold")
    ax.set_title(f"{sample['frame_id']}  GT=green, Pred=red(top-{len(pred_boxes)})")
    # 视角
    ax.view_init(elev=25, azim=-60)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"已保存: {out_path}")


if __name__ == "__main__":
    main()
