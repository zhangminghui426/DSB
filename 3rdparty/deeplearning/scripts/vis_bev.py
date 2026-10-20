"""Batch BEV visualization of occupancy results (no GUI, no open3d).

Port of the original FlashOCC tools/analysis_tools/vis_bev.py; the rendering
helpers live in scripts/render.py. For each sample with a pred.npz dump
(scripts/test.py --save_dir, or the official FlashOCC results dump) the
following files are written to --save_path (prefix '<idx>_<scene>_<token>'):

    <prefix>_overall.jpg   6-cam collage (left) + pred/gt BEV panels + legend
    <prefix>_bev_pred.png  predicted occupancy BEV (top-most non-free voxel)
    <prefix>_bev_gt.png    GT occupancy BEV (when labels.npz is found)
    <prefix>_bev_img.png   camera images back-projected to BEV (ground plane)
    <prefix>_bev_lidar.png LIDAR_TOP point cloud rasterized top-down (height)

All BEV images share one orientation: +x (ego forward) up, +y left, 40 m x
40 m around the ego vehicle.

Usage:
    python scripts/vis_bev.py outputs/results --root_path data/nuscenes \
        --save_path outputs/vis --draw_gt
"""

import argparse
import os
import os.path as osp
import pickle
import sys

sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), ".."))

import cv2
import numpy as np

from scripts.render import (VIEWS_BOT, VIEWS_TOP, camera_collage, label,
                            make_legend, render_bev, render_image_bev,
                            render_lidar_bev)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("res", help="results dir with <scene>/<token>/pred.npz")
    p.add_argument("--root_path", default="data/nuscenes",
                   help="nuScenes data root (with the infos pkl)")
    p.add_argument("--version", default="val",
                   help="infos pkl split: train / val")
    p.add_argument("--save_path", default="outputs/vis",
                   help="directory to save composed images")
    p.add_argument("--draw_gt", action="store_true",
                   help="also render the GT occupancy BEV next to pred")
    p.add_argument("--max_frames", type=int, default=-1,
                   help="limit number of frames (-1 = all)")
    p.add_argument("--cam_width", type=int, default=480,
                   help="width of each resized camera view")
    p.add_argument("--bev_size", type=int, default=480,
                   help="size of the square BEV rendering")
    p.add_argument("--no_legend", action="store_true",
                   help="do not draw the class color legend")
    return p.parse_args()


def main():
    args = parse_args()
    info_path = osp.join(args.root_path, f"bevdetv2-nuscenes_infos_{args.version}.pkl")
    with open(info_path, "rb") as f:
        infos = pickle.load(f)["infos"]
    if args.max_frames > 0:
        infos = infos[:args.max_frames]
    os.makedirs(args.save_path, exist_ok=True)
    print(f"saving composed images to {args.save_path} ({len(infos)} frames)")

    cam_w = args.cam_width
    n_done, n_skip = 0, 0

    for cnt, info in enumerate(infos):
        scene, token = info["scene_name"], info["token"]
        pred_path = osp.join(args.res, scene, token, "pred.npz")
        if not osp.exists(pred_path):
            print(f"[{cnt}] skip {scene}/{token}: no pred.npz")
            n_skip += 1
            continue
        pred_occ = np.load(pred_path)["pred"]
        base = osp.join(args.save_path, f"{cnt:04d}_{scene}_{token}")

        # GT npz: occ_path may be repo-root relative already -> fall back to raw
        gt_occ = None
        gt_npz = osp.join(args.root_path, info["occ_path"], "labels.npz")
        if not osp.exists(gt_npz):
            gt_npz = osp.join(info["occ_path"], "labels.npz")
        if osp.exists(gt_npz):
            gt_occ = np.load(gt_npz)["semantics"]

        # the 6 camera frames are read once and shared by the ground-plane
        # back-projection and the collage
        cam_imgs = {v: cv2.imread(info["cams"][v]["data_path"])
                    for v in VIEWS_TOP + VIEWS_BOT}

        # ---- standalone BEV parts ----
        cv2.imwrite(base + "_bev_pred.png", render_bev(pred_occ, args.bev_size))
        if gt_occ is not None:
            cv2.imwrite(base + "_bev_gt.png", render_bev(gt_occ, args.bev_size))
        cv2.imwrite(base + "_bev_img.png",
                    render_image_bev(info, args.bev_size, imgs=cam_imgs))
        cv2.imwrite(base + "_bev_lidar.png", render_lidar_bev(info, args.bev_size))

        # ---- left: 6 camera views (bottom row mirrored) ----
        left = camera_collage(cam_imgs, cam_w)  # (2*cam_h, 3*cam_w)

        # ---- right: BEV panel(s) + legend ----
        panels = []
        p = render_bev(pred_occ, args.bev_size)
        label(p, "pred", (8, 28), 0.8)
        panels.append(p)
        if args.draw_gt and gt_occ is not None:
            gt = render_bev(gt_occ, args.bev_size)
            label(gt, "gt", (8, 28), 0.8)
            panels.append(gt)
        right = np.concatenate(panels, axis=1)
        if not args.no_legend:
            legend = make_legend(right.shape[0])
            right = np.concatenate([right, legend], axis=1)

        # pad both sides to a common height, then hconcat
        h = max(left.shape[0], right.shape[0])
        if right.shape[0] < h:
            right = cv2.copyMakeBorder(right, 0, h - right.shape[0], 0, 0,
                                       cv2.BORDER_CONSTANT, value=(255, 255, 255))
        if left.shape[0] < h:
            left = cv2.copyMakeBorder(left, 0, h - left.shape[0], 0, 0,
                                      cv2.BORDER_CONSTANT, value=(0, 0, 0))
        canvas = np.concatenate([left, right], axis=1)
        label(canvas, f"{scene}  {token}", (8, canvas.shape[0] - 12), 0.6)

        cv2.imwrite(base + "_overall.jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
        n_done += 1
        if cnt % 20 == 0:
            print(f"{cnt + 1}/{len(infos)}")

    print(f"done: {n_done} saved, {n_skip} skipped -> {args.save_path}")


if __name__ == "__main__":
    main()
