"""Visualize TRT inference results: per-frame jpgs + temporal-sequence gif.

Runs the TRT pipeline (deploy.python.pipeline) over consecutive val samples — or
reuses one existing npz dump (single frame) — and renders to <out>/ (default
outputs/vis/, kept separate from the model/ckpt tree):

  sample{N}_det.jpg           BEV + 6 camera views with 3D boxes (per frame)
  sample{start}-{end}_seq.gif temporal sequence (frames x fps), when --frames > 1

Rendering reuses scripts/render.py (same drawing as scripts/visualize.py).

Usage (run from the model repo root 3rdparty/deeplearning):
  python deploy/python/visualize.py CONFIG CKPT --engine_dir outputs/trt \
      --sample-index 5 --frames 1 [--stride 1] \
      [--npz outputs/trt/sample5.npz] \
      [--out outputs/vis] [--score_thr 0.3] [--max_boxes 40]
"""

import argparse
import os
import os.path as osp
import sys

# deeplearning 根（上 3 级）：models/datasets/scripts 与 deploy.python 包都在其下
_DL_ROOT = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.insert(0, _DL_ROOT)

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
                   help="可选；TRT 推理直接用 engine，仅传入时加载权重（一般不需要）")
    p.add_argument("--engine_dir", default=osp.join("outputs", "trt"))
    p.add_argument("--sample-index", type=int, default=0,
                   help="first frame index (consecutive samples = same-scene sweeps)")
    p.add_argument("--frames", type=int, default=1,
                   help="consecutive frames to render; >1 also writes a seq gif")
    p.add_argument("--stride", type=int, default=1, help="frame subsample stride")
    p.add_argument("--npz", default=None,
                   help="reuse a dumped npz (boxes/scores/labels) instead of "
                        "running the TRT pipeline; single frame only")
    p.add_argument("--out", default=osp.join("outputs", "vis"))
    p.add_argument("--score_thr", type=float, default=0.3)
    p.add_argument("--max_boxes", type=int, default=40)
    p.add_argument("--fps", type=int, default=4, help="seq gif frame rate")
    p.add_argument("--options", nargs="*", default=[])
    return p.parse_args()


@torch.no_grad()
def render_frame(batch, sample, idx, boxes, scores, labels, args, mode):
    """Select boxes, compose the BEV + 6-camera frame, return a PIL frame."""
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

    title = (f"TRT sample {idx} | {mode} | "
             f"dets(score>={args.score_thr}): {n_sel}")
    return compose_det_frame(cam_imgs, cam_corners, cam_P, cam_colors,
                             cam_labels, bev, title, cam_names=CAM_NAMES)


@torch.no_grad()
def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda"

    assert not (args.npz and args.frames > 1), \
        "--npz mode is single-frame; drop --frames or run without --npz"
    indices = list(range(args.sample_index,
                         args.sample_index + args.frames * args.stride,
                         args.stride))

    cfg = load_config(args.config, options=args.options)
    ds = build_dataset(cfg.data.val)

    # online mode: load model + engines once, reuse for every frame
    inf = None
    if not args.npz:
        from models import build_model

        # TRT 推理直接用 engine；模型骨架提供结构信息，权重非必需
        model = build_model(cfg.model).cuda().eval()
        if args.checkpoint:
            from scripts.trainer import load_model_checkpoint
            load_model_checkpoint(model, args.checkpoint, strict=True)

        from deploy.python.inference import Inferencer
        from deploy.python.postprocess import Postprocessor
        from deploy.python.preprocess import Preprocessor
        from deploy.python.runtime import load_engines

        engines = load_engines("det", osp.join("outputs", "onnx"),
                               args.engine_dir, build=False)
        pre = Preprocessor(model)
        inf = Inferencer(model, engines, "det")
        post = Postprocessor(model)

    frames_pil = []
    for idx in indices:
        sample = ds[idx]
        batch = to_device(collate_fn([sample]), device)

        if args.npz:
            d = np.load(args.npz)
            boxes = torch.from_numpy(d["boxes"]).float()
            scores = torch.from_numpy(d["scores"]).float()
            labels = torch.from_numpy(d["labels"]).long()
            mode = f"npz {args.npz}"
        else:
            preds, _ = inf.infer(batch, pre(batch))
            boxes, scores, labels = post.decode(preds, batch["metas"])
            mode = f"TRT engines {args.engine_dir}"

        frame = render_frame(batch, sample, idx, boxes, scores, labels,
                             args, mode)
        jpg = osp.join(args.out, f"sample{idx}_det.jpg")
        frame.save(jpg, quality=90)
        print(f"saved {jpg}  ({frame.size[0]}x{frame.size[1]})")
        frames_pil.append(frame)

    if len(frames_pil) > 1:
        gif = osp.join(args.out,
                       f"sample{indices[0]}-{indices[-1]}_seq.gif")
        save_gif(frames_pil, gif, fps=args.fps)
        print(f"saved {gif}  ({len(frames_pil)} frames, fps {args.fps})")


if __name__ == "__main__":
    main()
