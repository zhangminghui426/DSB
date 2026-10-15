"""Visualization entry point for all tasks.

--task det / seg: run inference and render per-scene GIF animations
(6 camera panels with projected 3D boxes / BEV mask pred-vs-GT).
--task occ: BEV semantic rendering, offline from pred.npz dumps
(--results_dir) or online from config+checkpoint (+ surround RGB strip).

Usage (from repo root):
    python scripts/visualize.py configs/det_convfuser.py ckpts/bevfusion-det-bf.pth --task det --show_gt
    python scripts/visualize.py configs/seg_fusion_bev256d2_lss.py ckpts/bevfusion-seg-bf.pth --task seg
    python scripts/visualize.py --task occ --results_dir outputs/results --out outputs/vis
    python scripts/visualize.py configs/flashocc_r50.py ckpts/flashocc-r50-256x704-bff.pth --task occ --with_rgb

Outputs: {out}/{task}_{scene}.gif (det/seg) or {out}/<token>-*.jpg (occ).
"""

import argparse
import glob
import os
import os.path as osp
import sys

sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), ".."))

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from datasets import collate_fn
from scripts.config import load_config
from scripts.render import (
    DET_CLASSES,
    DET_COLORS,
    compose_det_frame,
    compose_seg_frame,
    denormalize_img,
    render_bev,
    render_det_bev,
    render_seg_bev,
    save_gif,
    surround_rgb,
)
from scripts.trainer import load_model_checkpoint, to_device
from models import build_model

# camera order in the batch: [FRONT, FRONT_RIGHT, FRONT_LEFT, BACK, BACK_LEFT, BACK_RIGHT]
# display order: FL, F, FR / BL, B, BR (two rows x three columns)
CAM_FRONT = 0  # CAM_FRONT index in the batch 6-cam order
CAM_ORDER = [2, 0, 1, 4, 3, 5]
CAM_NAMES = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config")
    p.add_argument("checkpoint")
    p.add_argument("--task", default="det", choices=["det", "seg"])
    p.add_argument("--out", default=osp.join("outputs", "vis"))
    p.add_argument("--score_thr", type=float, default=0.3)
    p.add_argument("--max_boxes", type=int, default=40)
    p.add_argument("--stride", type=int, default=1, help="frame subsample stride")
    p.add_argument("--fps", type=int, default=4)
    p.add_argument("--max_frames", type=int, default=0, help="0 = all")
    p.add_argument("--show_gt", action="store_true", help="draw GT boxes (det, dashed white)")
    args = p.parse_args()
    return args


def build_scene_map(dataset):
    """token -> scene name via nuscenes-devkit."""
    from nuscenes.nuscenes import NuScenes

    nusc = NuScenes(version=dataset.version, dataroot=dataset.dataset_root, verbose=False)
    t2s = {}
    for scene in nusc.scene:
        tok = scene["first_sample_token"]
        while tok != "":
            t2s[tok] = scene["name"]
            tok = nusc.get("sample", tok)["next"]
    return t2s


@torch.no_grad()
def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda"

    cfg = load_config(args.config)
    dataset = dataset_full = None
    from datasets import build_dataset

    dataset = build_dataset(cfg.data.test)
    model = build_model(cfg.model).to(device)
    msg, _ = load_model_checkpoint(model, args.checkpoint, strict=True)
    assert not msg.missing_keys and not msg.unexpected_keys, msg
    model.eval()
    print(f"loaded {args.checkpoint} (task={args.task}), {len(dataset)} samples")

    token2scene = build_scene_map(dataset)
    scene_frames = {}  # scene -> list of PIL frames

    indices = list(range(0, len(dataset), args.stride))
    if args.max_frames:
        indices = indices[: args.max_frames]

    for idx in tqdm(indices, desc="render"):
        sample = dataset[idx]
        if sample is None:
            continue
        batch = to_device(collate_fn([sample]), device)
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(**batch)[0]

        token = sample["metas"]["token"]
        scene = token2scene.get(token, "unknown")
        ts = dataset.data_infos[idx]["timestamp"] / 1e6

        if args.task == "det":
            frame = render_det(sample, batch, out, args, ts, scene, idx)
        else:
            frame = render_seg(sample, batch, out, ts, scene, idx)
        scene_frames.setdefault(scene, []).append(frame)

    for scene, frames in scene_frames.items():
        path = osp.join(args.out, f"{args.task}_{scene}.gif")
        save_gif(frames, path, fps=args.fps)
        size_mb = osp.getsize(path) / 1e6
        print(f"saved {path}  ({len(frames)} frames, {size_mb:.1f} MB)")


def shifted_corners(box_tensor: np.ndarray):
    """Official visualize.py convention: decoded/GT box z is gravity center
    while LiDARInstance3DBoxes.corners expects bottom-center, so shift
    z -= h/2 before building corners (otherwise boxes float h/2 too high)."""
    from datasets.geometry.boxes import LiDARInstance3DBoxes

    t = box_tensor.copy()
    t[..., 2] -= t[..., 5] / 2
    boxes = LiDARInstance3DBoxes(torch.from_numpy(t).float(), box_dim=t.shape[-1])
    return boxes.corners.cpu().numpy(), t[:, :3]


def render_det(sample, batch, out, args, ts, scene, idx):
    boxes = out["boxes_3d"]
    scores = out["scores_3d"].float().cpu().numpy()
    labels = out["labels_3d"].cpu().numpy()

    keep = scores >= args.score_thr
    order = np.argsort(-scores[keep])[: args.max_boxes]
    sel = np.where(keep)[0][order]

    if len(sel) > 0:
        b = boxes[torch.from_numpy(sel).to(boxes.tensor.device)]
        corners, centers = shifted_corners(b.tensor.cpu().numpy())
        colors = [DET_COLORS[int(labels[i]) % len(DET_COLORS)] for i in sel]
        labels_text = [f"{DET_CLASSES[int(labels[i])]} {scores[i]:.2f}" for i in sel]
    else:
        corners = np.zeros((0, 8, 3), np.float32)
        centers = np.zeros((0, 3), np.float32)
        colors, labels_text = [], None

    points = batch["points"][0].cpu().numpy()

    gt_corners = None
    if args.show_gt and len(batch["gt_bboxes_3d"][0]) > 0:
        gt_corners, _ = shifted_corners(batch["gt_bboxes_3d"][0].tensor.cpu().numpy())

    pc_range = max(abs(float(np.asarray(sample["metas"].get("pc_range", [-54, 54])[0]))), 54.0)
    bev = render_det_bev(points, corners, centers, colors, labels_text,
                         pc_range=pc_range, gt_corners=gt_corners)

    # camera panels: official visualize.py path — ORIGINAL images +
    # lidar2image (validated against mmdet3d/core/utils/visualize.py);
    # all 6 surround cameras (model input is 6 x 256 x 704 after augmentation)
    lidar2image = batch["lidar2image"][0].cpu().numpy()  # (6,4,4)
    cam_imgs, cam_corners, cam_P, cam_colors, cam_labels = [], [], [], [], []
    for c in CAM_ORDER:
        fn = sample["metas"]["filename"][c]
        if not osp.exists(fn):
            fn = osp.join("data/nuscenes", fn)
        cam_imgs.append(Image.open(fn).convert("RGB"))
        cam_corners.append(corners)
        cam_P.append(lidar2image[c])
        cam_colors.append(colors)
        cam_labels.append(labels_text)

    title = (
        f"{scene} | frame {idx} | {ts:.2f}s | dets(score>={args.score_thr}): {len(sel)}"
        + ("  [white dashed = GT]" if gt_corners is not None else "")
    )
    return compose_det_frame(
        cam_imgs, cam_corners, cam_P, cam_colors, cam_labels, bev, title, cam_names=CAM_NAMES
    )


def render_seg(sample, batch, out, ts, scene, idx):
    pred = (out["masks_bev"].numpy() > 0.5).astype(np.uint8)  # (6,200,200)
    gt = out["gt_masks_bev"].numpy()
    gt = (gt > 0).astype(np.uint8) if gt.dtype != bool else gt.astype(np.uint8)

    points = batch["points"][0].cpu().numpy()
    pred_bev = render_seg_bev(points, pred, "prediction (sigmoid>0.5)")
    gt_bev = render_seg_bev(points, gt, "ground truth (NuScenesMap)")
    cam = denormalize_img(batch["img"][0][CAM_FRONT])
    title = f"{scene} | frame {idx} | {ts:.2f}s | BEV segmentation pred vs GT"
    return compose_seg_frame(cam, pred_bev, gt_bev, title)




def run_det_seg(args):
    os.makedirs(args.out, exist_ok=True)
    device = "cuda"

    cfg = load_config(args.config)
    dataset = dataset_full = None
    from datasets import build_dataset

    dataset = build_dataset(cfg.data.test)
    model = build_model(cfg.model).to(device)
    msg, _ = load_model_checkpoint(model, args.checkpoint, strict=True)
    assert not msg.missing_keys and not msg.unexpected_keys, msg
    model.eval()
    print(f"loaded {args.checkpoint} (task={args.task}), {len(dataset)} samples")

    token2scene = build_scene_map(dataset)
    scene_frames = {}  # scene -> list of PIL frames

    indices = list(range(0, len(dataset), args.stride))
    if args.max_frames:
        indices = indices[: args.max_frames]

    for idx in tqdm(indices, desc="render"):
        sample = dataset[idx]
        if sample is None:
            continue
        batch = to_device(collate_fn([sample]), device)
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(**batch)[0]

        token = sample["metas"]["token"]
        scene = token2scene.get(token, "unknown")
        ts = dataset.data_infos[idx]["timestamp"] / 1e6

        if args.task == "det":
            frame = render_det(sample, batch, out, args, ts, scene, idx)
        else:
            frame = render_seg(sample, batch, out, ts, scene, idx)
        scene_frames.setdefault(scene, []).append(frame)

    for scene, frames in scene_frames.items():
        path = osp.join(args.out, f"{args.task}_{scene}.gif")
        save_gif(frames, path, fps=args.fps)
        size_mb = osp.getsize(path) / 1e6
        print(f"saved {path}  ({len(frames)} frames, {size_mb:.1f} MB)")


def shifted_corners(box_tensor: np.ndarray):
    """Official visualize.py convention: decoded/GT box z is gravity center
    while LiDARInstance3DBoxes.corners expects bottom-center, so shift
    z -= h/2 before building corners (otherwise boxes float h/2 too high)."""
    from datasets.geometry.boxes import LiDARInstance3DBoxes

    t = box_tensor.copy()
    t[..., 2] -= t[..., 5] / 2
    boxes = LiDARInstance3DBoxes(torch.from_numpy(t).float(), box_dim=t.shape[-1])
    return boxes.corners.cpu().numpy(), t[:, :3]


def render_det(sample, batch, out, args, ts, scene, idx):
    boxes = out["boxes_3d"]
    scores = out["scores_3d"].float().cpu().numpy()
    labels = out["labels_3d"].cpu().numpy()

    keep = scores >= args.score_thr
    order = np.argsort(-scores[keep])[: args.max_boxes]
    sel = np.where(keep)[0][order]

    if len(sel) > 0:
        b = boxes[torch.from_numpy(sel).to(boxes.tensor.device)]
        corners, centers = shifted_corners(b.tensor.cpu().numpy())
        colors = [DET_COLORS[int(labels[i]) % len(DET_COLORS)] for i in sel]
        labels_text = [f"{DET_CLASSES[int(labels[i])]} {scores[i]:.2f}" for i in sel]
    else:
        corners = np.zeros((0, 8, 3), np.float32)
        centers = np.zeros((0, 3), np.float32)
        colors, labels_text = [], None

    points = batch["points"][0].cpu().numpy()

    gt_corners = None
    if args.show_gt and len(batch["gt_bboxes_3d"][0]) > 0:
        gt_corners, _ = shifted_corners(batch["gt_bboxes_3d"][0].tensor.cpu().numpy())

    pc_range = max(abs(float(np.asarray(sample["metas"].get("pc_range", [-54, 54])[0]))), 54.0)
    bev = render_det_bev(points, corners, centers, colors, labels_text,
                         pc_range=pc_range, gt_corners=gt_corners)

    # camera panels: official visualize.py path — ORIGINAL images +
    # lidar2image (validated against mmdet3d/core/utils/visualize.py);
    # all 6 surround cameras (model input is 6 x 256 x 704 after augmentation)
    lidar2image = batch["lidar2image"][0].cpu().numpy()  # (6,4,4)
    cam_imgs, cam_corners, cam_P, cam_colors, cam_labels = [], [], [], [], []
    for c in CAM_ORDER:
        fn = sample["metas"]["filename"][c]
        if not osp.exists(fn):
            fn = osp.join("data/nuscenes", fn)
        cam_imgs.append(Image.open(fn).convert("RGB"))
        cam_corners.append(corners)
        cam_P.append(lidar2image[c])
        cam_colors.append(colors)
        cam_labels.append(labels_text)

    title = (
        f"{scene} | frame {idx} | {ts:.2f}s | dets(score>={args.score_thr}): {len(sel)}"
        + ("  [white dashed = GT]" if gt_corners is not None else "")
    )
    return compose_det_frame(
        cam_imgs, cam_corners, cam_P, cam_colors, cam_labels, bev, title, cam_names=CAM_NAMES
    )


def render_seg(sample, batch, out, ts, scene, idx):
    pred = (out["masks_bev"].numpy() > 0.5).astype(np.uint8)  # (6,200,200)
    gt = out["gt_masks_bev"].numpy()
    gt = (gt > 0).astype(np.uint8) if gt.dtype != bool else gt.astype(np.uint8)

    points = batch["points"][0].cpu().numpy()
    pred_bev = render_seg_bev(points, pred, "prediction (sigmoid>0.5)")
    gt_bev = render_seg_bev(points, gt, "ground truth (NuScenesMap)")
    cam = denormalize_img(batch["img"][0][CAM_FRONT])
    title = f"{scene} | frame {idx} | {ts:.2f}s | BEV segmentation pred vs GT"
    return compose_seg_frame(cam, pred_bev, gt_bev, title)




# ---- occupancy modes ----

def render_offline(results_dir, out_dir, max_frames=None, bev_size=800):
    npzs = sorted(glob.glob(osp.join(results_dir, "*", "*", "pred.npz")))
    if max_frames:
        npzs = npzs[:max_frames]
    if not npzs:
        raise SystemExit(f"no pred.npz found under {results_dir}")
    os.makedirs(out_dir, exist_ok=True)
    for path in npzs:
        data = np.load(path)
        token = str(data["sample_token"])
        cv2.imwrite(osp.join(out_dir, f"{token}-pred.jpg"),
                    render_bev(data["pred"], bev_size))
        cv2.imwrite(osp.join(out_dir, f"{token}-gt.jpg"),
                    render_bev(data["gt"], bev_size))
    print(f"rendered {len(npzs)} tokens -> {out_dir}")


def render_online(config, checkpoint, out_dir, max_frames, with_rgb, stride,
                  bev_size=800):
    from datasets import build_dataset, collate_fn
    from models import build_model
    from scripts.config import load_config
    from scripts.trainer import load_model_checkpoint, to_device

    cfg = load_config(config)
    occ_cfg = (cfg.model.get("encoders") or {}).get("occ") or {}
    if occ_cfg.get("type") == "OccEncoderStereo4D":
        occ_cfg["align_after_view_transfromation"] = True  # official test protocol

    dataset = build_dataset(cfg.data.test)
    model = build_model(cfg.model).cuda().eval()
    msg, _ = load_model_checkpoint(model, checkpoint, strict=True)
    print(f"loaded {checkpoint}: missing={len(msg.missing_keys)} "
          f"unexpected={len(msg.unexpected_keys)}")
    os.makedirs(out_dir, exist_ok=True)

    indices = list(range(0, len(dataset), stride))
    if max_frames:
        indices = indices[:max_frames]
    for idx in indices:
        sample = dataset[idx]
        batch = to_device(collate_fn([sample]), "cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            out = model(**batch)[0]
        token = sample["metas"]["token"]
        cv2.imwrite(osp.join(out_dir, f"{token}-pred.jpg"),
                    render_bev(out["pred_occ"].numpy(), bev_size))
        npz = np.load(osp.join(sample["metas"]["occ_gt_path"], "labels.npz"))
        cv2.imwrite(osp.join(out_dir, f"{token}-gt.jpg"),
                    render_bev(npz["semantics"], bev_size))
        if with_rgb:
            cv2.imwrite(osp.join(out_dir, f"{token}-rgb.jpg"),
                        surround_rgb(batch["img_inputs"][0][0]))
        print(f"  {token} rendered")
    print(f"rendered {len(indices)} tokens -> {out_dir}")




def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", nargs="?", default=None)
    p.add_argument("checkpoint", nargs="?", default=None)
    p.add_argument("--task", default="det", choices=["det", "seg", "occ"])
    p.add_argument("--out", default=osp.join("outputs", "vis"))
    p.add_argument("--score_thr", type=float, default=0.3)
    p.add_argument("--max_boxes", type=int, default=40)
    p.add_argument("--stride", type=int, default=1, help="frame subsample stride")
    p.add_argument("--fps", type=int, default=4)
    p.add_argument("--max_frames", type=int, default=0, help="0 = all")
    p.add_argument("--show_gt", action="store_true", help="draw GT boxes (det, dashed white)")
    p.add_argument("--results_dir", default=None,
                   help="occ offline mode: directory with <scene>/<token>/pred.npz")
    p.add_argument("--bev_size", type=int, default=800,
                   help="occ: size of the square BEV rendering")
    p.add_argument("--with_rgb", action="store_true",
                   help="occ online: also write the surround-view RGB strip")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    if args.task == "occ":
        if args.results_dir:
            render_offline(args.results_dir, args.out, args.max_frames, args.bev_size)
            return
        if not (args.config and args.checkpoint):
            raise SystemExit("occ online mode needs CONFIG CHECKPOINT")
        render_online(args.config, args.checkpoint, args.out, args.max_frames,
                      args.with_rgb, args.stride, args.bev_size)
        return
    run_det_seg(args)


if __name__ == "__main__":
    main()
