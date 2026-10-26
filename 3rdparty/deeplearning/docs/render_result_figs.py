"""Run real inference on nuScenes sample 0 and render the figures used by the
architecture diagram (docs/figs/).

Usage (3rdparty/deeplearning 下):
  python docs/render_result_figs.py
"""

import os
import os.path as osp
import sys

_HERE = osp.dirname(osp.abspath(__file__))
_DL_ROOT = osp.dirname(_HERE)
if _DL_ROOT not in sys.path:
    sys.path.insert(0, _DL_ROOT)

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image

from datasets import build_dataset, collate_fn
from datasets.definitions import OCC_CLASS_NAMES
from deploy.scripts.common import load_tensor
from models import build_model
from scripts.config import load_config
from scripts.render import (camera_collage, compose_det_frame, compose_seg_frame,
                            denormalize_img, render_det_bev, render_seg_bev)
from scripts.render import DET_CLASSES, DET_COLORS
from scripts.trainer import load_model_checkpoint, to_device
from scripts.vis_occ import render_occ_offscreen
from scripts.visualize import CAM_NAMES, CAM_ORDER, shifted_corners

OUT = osp.join(_HERE, "figs")
DATA = "outputs/example-data"
SAMPLE = 0
CAMS = ["0-FRONT.jpg", "1-FRONT_RIGHT.jpg", "2-FRONT_LEFT.jpg",
        "3-BACK.jpg", "4-BACK_LEFT.jpg", "5-BACK_RIGHT.jpg"]
COLLAGE_ORDER = [2, 0, 1, 4, 3, 5]  # FL, F, FR, BL, B, BR


def _load_model(config, ckpt):
    cfg = load_config(config)
    model = build_model(cfg.model).cuda().eval()
    load_model_checkpoint(model, ckpt, strict=True)
    return cfg, model


def _load_batch(cfg):
    ds = build_dataset(cfg.data.val)
    return to_device(collate_fn([ds[SAMPLE]]), "cuda"), ds


def run_seg():
    cfg, model = _load_model("configs/seg_fusion_bev256d2_lss.py",
                             "ckpts/bevfusion-seg-bf.pth")
    batch, _ = _load_batch(cfg)
    with torch.no_grad():
        out = model(**batch)[0]
    return out, batch


def run_occ():
    cfg, model = _load_model("configs/flashocc_r50.py",
                             "ckpts/flashocc-r50-256x704-bff.pth")
    batch, ds = _load_batch(cfg)
    with torch.no_grad():
        out = model(**batch)[0]
    return out["pred_occ"], ds.data_list[SAMPLE]


def render_detection(points):
    """Official 6-camera + centered BEV layout (visualize.py render_det)."""
    l2i = load_tensor(osp.join(DATA, "lidar2image.tensor")).astype(np.float32)
    boxes = np.load("outputs/model_diff/torch/boxes.npy")
    keep = boxes[:, 9] > 0.3
    boxes = boxes[keep][np.argsort(-boxes[keep, 9])][:40]
    corners, centers = shifted_corners(boxes[:, :9])
    colors = [DET_COLORS[int(l) % len(DET_COLORS)] for l in boxes[:, 10]]
    labels = [f"{DET_CLASSES[int(l)]} {s:.2f}"
              for l, s in zip(boxes[:, 10], boxes[:, 9])]

    bev = render_det_bev(points, corners, centers, colors, labels,
                         pc_range=54, size=(520, 520))
    cam_imgs = [Image.open(osp.join(DATA, CAMS[c])).convert("RGB") for c in CAM_ORDER]
    frame = compose_det_frame(
        cam_imgs, [corners] * 6, [l2i[c] for c in CAM_ORDER],
        [colors] * 6, [labels] * 6, bev,
        f"BEVFusion detection | score>0.3 | {len(boxes)} boxes",
        cam_names=CAM_NAMES)
    frame.save(osp.join(OUT, "det_result.png"))
    print("[fig] det_result.png")


def render_lidar_input(points, path):
    """LiDAR BEV with height color + 10 m range rings."""
    keep = (np.abs(points[:, 0]) < 54) & (np.abs(points[:, 1]) < 54)
    p = points[keep]
    fig = plt.figure(figsize=(5.6, 5.6), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("#0b0f14")
    fig.patch.set_facecolor("#0b0f14")
    ax.scatter(p[:, 0], p[:, 1], s=0.9, c=p[:, 2], cmap="turbo",
               vmin=-2.5, vmax=2.5, alpha=0.95, linewidths=0)
    for r in (10, 20, 30, 40, 50):
        ax.add_patch(plt.Circle((0, 0), r, fill=False, color="#334155",
                                lw=0.6, ls=(0, (2, 3))))
    ax.plot([0], [0], marker="^", color="red", ms=6)
    ax.set_xlim(-54, 54)
    ax.set_ylim(-54, 54)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(path, facecolor="#0b0f14")
    plt.close(fig)
    print("[fig]", osp.basename(path))


def main():
    os.makedirs(OUT, exist_ok=True)

    imgs = [cv2.imread(osp.join(DATA, CAMS[i])) for i in COLLAGE_ORDER]
    Image.fromarray(camera_collage(imgs, cam_w=600)[:, :, ::-1]).save(
        osp.join(OUT, "cameras.png"))
    print("[fig] cameras.png")

    occ, occ_info = run_occ()
    points = load_tensor(osp.join(DATA, "points.tensor")).astype(np.float32)
    render_lidar_input(points, osp.join(OUT, "lidar_bev.png"))

    render_detection(points)

    # segmentation: official render_seg layout (camera + pred/GT BEV)
    out, batch = run_seg()
    pred = (out["masks_bev"].numpy() > 0.5).astype(np.uint8)
    gt = (out["gt_masks_bev"].numpy() > 0).astype(np.uint8)
    pred_bev = render_seg_bev(points, pred, "prediction (sigmoid>0.5)")
    gt_bev = render_seg_bev(points, gt, "ground truth (NuScenesMap)")
    cam = denormalize_img(batch["img"][0][0])
    compose_seg_frame(cam, pred_bev, gt_bev,
                      "BEV segmentation | pred vs GT").save(
        osp.join(OUT, "seg_result.png"))
    print("[fig] seg_result.png")

    # occupancy: vis_occ headless render (camera-visible non-free voxels)
    labels = np.load(osp.join(occ_info["occ_path"], "labels.npz"))
    show = (occ != len(OCC_CLASS_NAMES) - 1) & labels["mask_camera"]
    render_occ_offscreen(occ, show, [0.4, 0.4, 0.4],
                         osp.join(OUT, "occ_result.png"),
                         size=(900, 800), crop_m=22)
    print("[fig] occ_result.png")


if __name__ == "__main__":
    main()
