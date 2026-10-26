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
from datasets.definitions import OBJECT_CLASSES, OCC_CLASS_NAMES, OCC_COLOR_MAP
from deploy.scripts.common import load_tensor
from models import build_model
from scripts.config import load_config
from scripts.render import (camera_collage, draw_boxes_on_image, render_bev,
                            render_det_bev, render_seg_bev)
from scripts.trainer import load_model_checkpoint, to_device
from scripts.visualize import shifted_corners

OUT = osp.join(_HERE, "figs")
DATA = "outputs/example-data"
SAMPLE = 0
CAMS = ["0-FRONT.jpg", "1-FRONT_RIGHT.jpg", "2-FRONT_LEFT.jpg",
        "3-BACK.jpg", "4-BACK_LEFT.jpg", "5-BACK_RIGHT.jpg"]
COLLAGE_ORDER = [2, 0, 1, 4, 3, 5]  # FL, F, FR, BL, B, BR
DET_COLORS = ["#1f77b4", "#ff7f0e", "#8c564b", "#d62728", "#9467bd",
              "#7f7f7f", "#bcbd22", "#17becf", "#55c255", "#e377c2"]


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
        out = model(**batch)
    return out[0]["masks_bev"].cpu().numpy()


def run_occ():
    cfg, model = _load_model("configs/flashocc_r50.py",
                             "ckpts/flashocc-r50-256x704-bff.pth")
    batch, ds = _load_batch(cfg)
    with torch.no_grad():
        out = model(**batch)
    return out[0]["pred_occ"], ds.data_list[SAMPLE]


def render_detection():
    points = load_tensor(osp.join(DATA, "points.tensor")).astype(np.float32)
    l2i = load_tensor(osp.join(DATA, "lidar2image.tensor")).astype(np.float32)
    boxes = np.load("outputs/model_diff/torch/boxes.npy")
    keep = boxes[:, 9] > 0.3
    boxes = boxes[keep][np.argsort(-boxes[keep, 9])][:30]
    corners, centers = shifted_corners(boxes[:, :9])
    pal = [DET_COLORS[int(l) % len(DET_COLORS)] for l in boxes[:, 10]]
    labels = [f"{OBJECT_CLASSES[int(l)]} {s:.2f}"
              for l, s in zip(boxes[:, 10], boxes[:, 9])]

    bev = render_det_bev(points, corners, centers, pal, labels,
                         pc_range=54, size=(560, 560))
    cam = Image.open(osp.join(DATA, CAMS[0])).convert("RGB")
    cam = draw_boxes_on_image(cam, corners, l2i[0], pal, labels)
    cam = cam.resize((int(cam.size[0] * bev.size[1] / cam.size[1]), bev.size[1]))
    out = Image.new("RGB", (bev.size[0] + cam.size[0] + 8, bev.size[1]), "#101010")
    out.paste(bev, (0, 0))
    out.paste(cam, (bev.size[0] + 8, 0))
    out.save(osp.join(OUT, "det_result.png"))
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


def render_occ_3d(occ, path):
    """Isometric view: translucent ground layers + opaque object voxels."""
    small = occ[50:150, 50:150, :][::2, ::2, :]  # +-20 m
    ground = (small >= 11) & (small <= 16)
    obj = (small >= 1) & (small <= 10)
    colors = np.asarray(OCC_COLOR_MAP, dtype=np.float32) / 255.0

    def face_for(mask):
        face = np.zeros(mask.shape + (3,), np.float32)
        for c in range(len(OCC_CLASS_NAMES)):
            face[mask & (small == c)] = colors[c]
        return face

    fig = plt.figure(figsize=(5.2, 4.8), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    ax.voxels(ground, facecolors=face_for(ground), edgecolors=None, alpha=0.22)
    ax.voxels(obj, facecolors=face_for(obj), edgecolors=None, alpha=1.0)
    ax.set_box_aspect((1, 1, 0.55))
    ax.view_init(elev=35, azim=-60)
    ax.set_xlim(0, small.shape[0])
    ax.set_ylim(0, small.shape[1])
    ax.set_zlim(0, small.shape[2])
    ax.axis("off")
    fig.patch.set_alpha(0)
    fig.savefig(path, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print("[fig]", osp.basename(path))


def main():
    os.makedirs(OUT, exist_ok=True)

    imgs = [cv2.imread(osp.join(DATA, CAMS[i])) for i in COLLAGE_ORDER]
    collage = camera_collage(imgs, cam_w=600)
    Image.fromarray(collage[:, :, ::-1]).save(osp.join(OUT, "cameras.png"))
    print("[fig] cameras.png")

    occ, _ = run_occ()
    points = load_tensor(osp.join(DATA, "points.tensor")).astype(np.float32)
    render_lidar_input(points, osp.join(OUT, "lidar_bev.png"))

    render_detection()

    masks = run_seg()
    seg_mask = (masks > 0.5).astype(np.uint8)[:, 40:160, 40:160]  # +-30 m
    render_seg_bev(points, seg_mask, "", size=(560, 560), pc_range=30).save(
        osp.join(OUT, "seg_result.png"))
    print("[fig] seg_result.png")

    Image.fromarray(render_bev(occ, 560)[:, :, ::-1]).save(osp.join(OUT, "occ_bev.png"))
    print("[fig] occ_bev.png")
    render_occ_3d(occ, osp.join(OUT, "occ_3d.png"))


if __name__ == "__main__":
    main()
