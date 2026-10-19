"""Visualize TRT inference results: single-frame jpg + rotating-BEV gif.

Runs the TRT pipeline (deploy.inferencer) on one val sample — or reuses an
existing npz dump (boxes/scores/labels) — then renders to <out>/ (default
outputs/vis/, kept separate from the model/ckpt tree):

  sample{N}_det.jpg       single frame: BEV + 6 camera views with 3D boxes
  sample{N}_bev_rot.gif   BEV rotating around z (24 frames, 360 deg)

Rendering reuses scripts/render.py (same drawing as scripts/visualize.py).

Usage:
  python deploy/visualize.py CONFIG CKPT --engine_dir outputs/trt \
      --sample-index 5 [--npz outputs/trt/sample5.npz] \
      [--out outputs/vis] [--score_thr 0.3] [--max_boxes 40]
"""

import argparse
import os
import os.path as osp
import sys

sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), ".."))

import numpy as np
import torch

from datasets import build_dataset, collate_fn
from scripts.config import load_config
from scripts.render import (
    DET_CLASSES,
    DET_COLORS,
    compose_det_frame,
    denormalize_img,
    render_det_bev,
    save_gif,
)
from scripts.trainer import to_device
from scripts.visualize import CAM_ORDER, CAM_NAMES, shifted_corners


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config")
    p.add_argument("checkpoint", nargs="?", default=None,
                   help="needed for the online (TRT) mode; ignored with --npz")
    p.add_argument("--engine_dir", default=osp.join("outputs", "trt"))
    p.add_argument("--sample-index", type=int, default=0)
    p.add_argument("--npz", default=None,
                   help="reuse a dumped npz (boxes/scores/labels) instead of "
                        "running the TRT pipeline")
    p.add_argument("--out", default=osp.join("outputs", "vis"))
    p.add_argument("--score_thr", type=float, default=0.3)
    p.add_argument("--max_boxes", type=int, default=40)
    p.add_argument("--rot-frames", type=int, default=24,
                   help="frames of the rotating-BEV gif (360 deg sweep)")
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--options", nargs="*", default=[])
    return p.parse_args()


def rot_z(ang_deg):
    a = np.deg2rad(ang_deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], np.float32)


def render_rot_gif(points, corners, centers, colors, labels_text, pc_range,
                   path, frames=24, fps=8):
    """BEV sweep 0..360 deg: rotate points + box corners, one gif frame each."""
    rot_frames = []
    step = 360.0 / frames
    for i in range(frames):
        ang = i * step
        R = rot_z(ang)
        p = points.copy()
        p[:, :3] = p[:, :3] @ R.T
        c = corners.copy()
        c = c @ R.T
        cen = centers.copy()
        cen = cen @ R.T
        rot_frames.append(render_det_bev(p, c, cen, colors, labels_text,
                                         pc_range=pc_range))
    save_gif(rot_frames, path, fps=fps)


@torch.no_grad()
def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda"

    cfg = load_config(args.config, options=args.options)
    ds = build_dataset(cfg.data.val)
    sample = ds[args.sample_index]
    batch = to_device(collate_fn([sample]), device)

    # ---- boxes/scores/labels: TRT pipeline or reused npz -----------------
    if args.npz:
        assert osp.exists(args.npz), f"npz not found: {args.npz}"
        d = np.load(args.npz)
        boxes = torch.from_numpy(d["boxes"]).float()
        scores = torch.from_numpy(d["scores"]).float()
        labels = torch.from_numpy(d["labels"]).long()
        mode = f"npz {args.npz}"
    else:
        assert args.checkpoint, "checkpoint required for the online TRT mode"
        from models import build_model
        from scripts.trainer import load_model_checkpoint

        model = build_model(cfg.model).cuda().eval()
        load_model_checkpoint(model, args.checkpoint, strict=True)

        from deploy.inferencer import DetInferencer
        from deploy.runtime import load_engines

        engines = load_engines("det", osp.join("outputs", "onnx"),
                               args.engine_dir, build=False)
        inf = DetInferencer(model, engines, "det")
        preds, _ = inf.infer(batch)
        boxes, scores, labels = inf.decode(preds, batch["metas"])
        mode = f"TRT engines {args.engine_dir}"

    # ---- select boxes ----------------------------------------------------
    box_t = boxes.tensor if hasattr(boxes, "tensor") else boxes  # 3DBoxes -> raw
    scores = scores.float().cpu().numpy()
    labels = labels.cpu().numpy()
    keep = scores >= args.score_thr
    order = np.argsort(-scores[keep])[: args.max_boxes]
    sel = np.where(keep)[0][order]
    n_sel = len(sel)

    if n_sel > 0:
        corners, centers = shifted_corners(
            box_t[torch.from_numpy(sel)].cpu().numpy())
        colors = [DET_COLORS[int(labels[i]) % len(DET_COLORS)] for i in sel]
        labels_text = [f"{DET_CLASSES[int(labels[i])]} {scores[i]:.2f}"
                       for i in sel]
    else:
        corners = np.zeros((0, 8, 3), np.float32)
        centers = np.zeros((0, 3), np.float32)
        colors, labels_text = [], None

    points = batch["points"][0].cpu().numpy()
    pc_range = max(abs(float(np.asarray(
        sample["metas"].get("pc_range", [-54, 54]))[0])), 54.0)

    # ---- single frame: BEV + 6 cameras -----------------------------------
    bev = render_det_bev(points, corners, centers, colors, labels_text,
                         pc_range=pc_range)
    lidar2image = batch["lidar2image"][0].cpu().numpy()  # (6,4,4)
    cam_imgs, cam_corners, cam_P, cam_colors, cam_labels = [], [], [], [], []
    for c in CAM_ORDER:
        cam_imgs.append(denormalize_img(batch["img"][0][c]))
        cam_corners.append(corners)
        cam_P.append(lidar2image[c])
        cam_colors.append(colors)
        cam_labels.append(labels_text)

    title = (f"TRT sample {args.sample_index} | {mode} | "
             f"dets(score>={args.score_thr}): {n_sel}")
    frame = compose_det_frame(cam_imgs, cam_corners, cam_P, cam_colors,
                              cam_labels, bev, title, cam_names=CAM_NAMES)

    stem = osp.join(args.out, f"sample{args.sample_index}")
    jpg = f"{stem}_det.jpg"
    frame.save(jpg, quality=90)
    print(f"saved {jpg}  ({frame.size[0]}x{frame.size[1]})")

    gif = f"{stem}_bev_rot.gif"
    render_rot_gif(points, corners, centers, colors, labels_text, pc_range,
                   gif, frames=args.rot_frames, fps=args.fps)
    print(f"saved {gif}  ({args.rot_frames} frames)")


if __name__ == "__main__":
    main()
