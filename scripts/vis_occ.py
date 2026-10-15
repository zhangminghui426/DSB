"""Interactive 3D voxel visualization of occupancy results (open3d).

Port of the original FlashOCC tools/analysis_tools/vis_occ.py (mm deps
removed; palette shared via datasets/definitions.py). Each sample renders
the predicted voxels (camera-mask visible, non-free) as colored boxes and
captures the offscreen buffer. Per frame:

    <save_path>/<scene>/<token>/img{0..5}.png  raw camera views
    <save_path>/<scene>/<token>/occ.png        open3d voxel capture
    <save_path>/<scene>/<token>/overall.png    cameras + occ composite

(--format video writes <save_path>/<video_prefix>.mp4 instead.)
With --draw_gt the GT voxels are drawn side by side (y-offset by one grid).
With --interactive a visible window opens per frame (drag to rotate, scroll
to zoom, close it to capture and continue); default is batch offscreen.

Requires a display (X11). Results dir = scripts/test.py --save_dir or the
official FlashOCC dump (<scene>/<token>/pred.npz).

Usage:
    python scripts/vis_occ.py outputs/results --root_path data/nuscenes \
        --save_path outputs/vis --draw_gt --vis_frames 8
"""

import argparse
import os
import os.path as osp
import pickle
import sys

sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), ".."))

import cv2
import numpy as np
import torch

try:  # open3d is only needed for this entry point (optional in the merged env)
    import open3d as o3d
except ImportError:
    o3d = None

from datasets.definitions import NUM_CLASSES, OCC_COLOR_MAP

FREE_LABEL = NUM_CLASSES - 1  # 17
# Occ grid geometry (matches grid_config in configs/common.py)
VOXEL_SIZE = [0.4, 0.4, 0.4]
POINT_CLOUD_RANGE = [-40, -40, -1, 40, 40, 5.4]
COLORS17 = np.array(OCC_COLOR_MAP[:FREE_LABEL], dtype=np.float32) / 255.0  # RGB


def voxel2points(voxel, occ_show, voxel_size):
    """(Dx,Dy,Dz) labels + bool mask -> occupied voxel corner points.

    Returns: points (N,3) xyz at the voxel-range origin, labels (N,),
    occIdx tuple of (x_idx, y_idx, z_idx) index tensors.
    """
    occIdx = torch.where(occ_show)
    points = torch.cat(
        (occIdx[0][:, None] * voxel_size[0] + POINT_CLOUD_RANGE[0],
         occIdx[1][:, None] * voxel_size[1] + POINT_CLOUD_RANGE[1],
         occIdx[2][:, None] * voxel_size[2] + POINT_CLOUD_RANGE[2]), dim=1)
    return points, voxel[occIdx], occIdx


def voxel_profile(voxel, voxel_size):
    """(N,3) voxel origins -> (N,7) boxes (center x, y, z-dz/2, vx, vy, vz, 0)."""
    centers = torch.cat(
        (voxel[:, :2], voxel[:, 2][:, None] - voxel_size[2] / 2), dim=1)
    wlh = torch.cat((torch.tensor(voxel_size[0]).repeat(centers.shape[0])[:, None],
                     torch.tensor(voxel_size[1]).repeat(centers.shape[0])[:, None],
                     torch.tensor(voxel_size[2]).repeat(centers.shape[0])[:, None]),
                    dim=1)
    yaw = torch.full_like(centers[:, 0:1], 0)
    return torch.cat((centers, wlh, yaw), dim=1)


def compute_box_3d(center, size, heading_angle):
    """Box centers/sizes/yaw -> (N,8,3) corners (axis-aligned voxel boxes)."""
    h, w, l = size[:, 2], size[:, 0], size[:, 1]
    center[:, 2] = center[:, 2] + h / 2
    l, w, h = (l / 2).unsqueeze(1), (w / 2).unsqueeze(1), (h / 2).unsqueeze(1)
    x_corners = torch.cat([-l, l, l, -l, -l, l, l, -l], dim=1)[..., None]
    y_corners = torch.cat([w, w, -w, -w, w, w, -w, -w], dim=1)[..., None]
    z_corners = torch.cat([h, h, h, h, -h, -h, -h, -h], dim=1)[..., None]
    corners_3d = torch.cat([x_corners, y_corners, z_corners], dim=2)
    corners_3d[..., 0] += center[:, 0:1]
    corners_3d[..., 1] += center[:, 1:2]
    corners_3d[..., 2] += center[:, 2:3]
    return corners_3d


def show_point_cloud(points, colors=True, points_colors=None, voxelize=False,
                     bbox_corners=None, linesets=None, vis=None,
                     offset=(0, 0, 0), large_voxel=True, voxel_size=0.4):
    """Add a voxel grid (or raw pcd) + optional wireframe + coord frame to vis."""
    if vis is None:
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window()
    if isinstance(offset, (list, tuple)):
        offset = np.array(offset)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points + offset)
    if colors:
        pcd.colors = o3d.utility.Vector3dVector(points_colors[:, :3])
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1, origin=[0, 0, 0])

    voxelGrid = o3d.geometry.VoxelGrid.create_from_point_cloud(
        pcd, voxel_size=voxel_size)
    vis.add_geometry(voxelGrid if large_voxel else pcd)

    if voxelize:
        line_sets = o3d.geometry.LineSet()
        line_sets.points = o3d.utility.Vector3dVector(
            bbox_corners.reshape((-1, 3)) + offset)
        line_sets.lines = o3d.utility.Vector2iVector(linesets.reshape((-1, 2)))
        line_sets.paint_uniform_color((0, 0, 0))
        vis.add_geometry(line_sets)

    vis.add_geometry(mesh_frame)
    return vis


def show_occ(occ_state, occ_show, voxel_size, vis=None, offset=(0, 0, 0)):
    """Draw one occupancy grid: colored voxel boxes + black wireframe edges."""
    pcd, labels, occIdx = voxel2points(occ_state, occ_show, voxel_size)
    pcds_colors = COLORS17[labels % len(COLORS17)]  # (N,3)

    bboxes = voxel_profile(pcd, voxel_size)  # (N,7)
    bboxes_corners = compute_box_3d(bboxes[:, 0:3], bboxes[:, 3:6],
                                    bboxes[:, 6:7])  # (N,8,3)

    bases_ = torch.arange(0, bboxes_corners.shape[0] * 8, 8)
    edges = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0],
                          [4, 5], [5, 6], [6, 7], [7, 4],
                          [0, 4], [1, 5], [2, 6], [3, 7]])  # 12 edges per box
    edges = edges.reshape((1, 12, 2)).repeat(bboxes_corners.shape[0], 1, 1)
    edges = edges + bases_[:, None, None]  # global corner ids

    return show_point_cloud(points=pcd.numpy(), colors=True,
                            points_colors=pcds_colors, voxelize=True,
                            bbox_corners=bboxes_corners.numpy(),
                            linesets=edges.numpy(), vis=vis, offset=offset,
                            large_voxel=True, voxel_size=voxel_size[0])


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
                   help="directory to save visualization results")
    p.add_argument("--canva_size", type=int, default=1000,
                   help="size of the occ canvas in the composite (pixel)")
    p.add_argument("--vis_frames", type=int, default=500,
                   help="number of frames for visualization")
    p.add_argument("--scale_factor", type=int, default=4,
                   help="trade-off between image-view and bev canvas size")
    p.add_argument("--draw_gt", action="store_true",
                   help="also draw the GT voxels (y-offset by one grid)")
    p.add_argument("--interactive", action="store_true",
                   help="open a visible window per frame (close to capture)")
    p.add_argument("--format", default="image", choices=["video", "image"],
                   help="output format")
    p.add_argument("--fps", type=int, default=10, help="frame rate of video")
    p.add_argument("--video_prefix", default="vis", help="name of video")
    return p.parse_args()


def main():
    if o3d is None:
        raise SystemExit("open3d is required: pip install open3d")
    args = parse_args()
    info_path = osp.join(args.root_path, f"bevdetv2-nuscenes_infos_{args.version}.pkl")
    with open(info_path, "rb") as f:
        infos = pickle.load(f)["infos"]
    os.makedirs(args.save_path, exist_ok=True)
    print(f"saving visualized result to {args.save_path}")

    scale_factor = args.scale_factor
    canva_size = args.canva_size
    if args.format == "video":
        fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
        vout = cv2.VideoWriter(
            osp.join(args.save_path, f"{args.video_prefix}.mp4"), fourcc,
            args.fps, (int(1600 / scale_factor * 3),
                       int(900 / scale_factor * 2 + canva_size)))

    views = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
             "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]
    print("start visualizing results")

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(visible=args.interactive)

    n = min(args.vis_frames, len(infos))
    for cnt, info in enumerate(infos[:n]):
        if cnt % 10 == 0:
            print(f"{cnt}/{n}")
        scene_name = info["scene_name"]
        sample_token = info["token"]

        pred_path = osp.join(args.res, scene_name, sample_token, "pred.npz")
        if not osp.exists(pred_path):
            print(f"[{cnt}] skip {scene_name}/{sample_token}: no pred.npz")
            continue
        pred_occ = np.load(pred_path)["pred"]
        # GT npz: occ_path may be repo-root relative already -> fall back to raw
        gt_npz_path = osp.join(args.root_path, info["occ_path"], "labels.npz")
        if not osp.exists(gt_npz_path):
            gt_npz_path = osp.join(info["occ_path"], "labels.npz")
        gt_data = np.load(gt_npz_path)
        voxel_label = gt_data["semantics"]
        camera_mask = gt_data["mask_camera"]

        imgs = [cv2.imread(info["cams"][v]["data_path"]) for v in views]

        # pred voxels (camera-visible, non-free); GT drawn with a y-offset
        voxel_show = np.logical_and(pred_occ != FREE_LABEL, camera_mask)
        vis = show_occ(torch.from_numpy(pred_occ), torch.from_numpy(voxel_show),
                       voxel_size=VOXEL_SIZE, vis=vis)
        if args.draw_gt:
            voxel_show = np.logical_and(voxel_label != FREE_LABEL, camera_mask)
            vis = show_occ(torch.from_numpy(voxel_label),
                           torch.from_numpy(voxel_show),
                           voxel_size=VOXEL_SIZE, vis=vis,
                           offset=[0, voxel_label.shape[0] * VOXEL_SIZE[0] * 1.2, 0])

        view_control = vis.get_view_control()
        view_control.set_lookat(np.array([-0.185, 0.513, 3.485]))
        view_control.set_front(np.array([-0.974, -0.055, 0.221]))
        view_control.set_up(np.array([0.221, 0.014, 0.975]))
        view_control.set_zoom(0.08)
        opt = vis.get_render_option()
        opt.background_color = np.asarray([1, 1, 1])
        opt.line_width = 5

        vis.poll_events()
        vis.update_renderer()
        if args.interactive:
            vis.run()  # close the window to capture + next frame

        occ_canvas = np.asarray(vis.capture_screen_float_buffer(do_render=True))
        occ_canvas = ((occ_canvas * 255).astype(np.uint8))[..., [2, 1, 0]]
        occ_canvas_resize = cv2.resize(occ_canvas, (canva_size, canva_size),
                                       interpolation=cv2.INTER_CUBIC)
        vis.clear_geometries()

        # composite: front-3 cams top, mirrored back-3 bottom, occ centered
        big_img = np.zeros((900 * 2 + canva_size * scale_factor, 1600 * 3, 3),
                           dtype=np.uint8)
        big_img[:900, :, :] = np.concatenate(imgs[:3], axis=1)
        img_back = np.concatenate(
            [imgs[3][:, ::-1, :], imgs[4][:, ::-1, :], imgs[5][:, ::-1, :]],
            axis=1)
        big_img[900 + canva_size * scale_factor:, :, :] = img_back
        big_img = cv2.resize(big_img, (int(1600 / scale_factor * 3),
                                       int(900 / scale_factor * 2 + canva_size)))
        w_begin = int((1600 * 3 / scale_factor - canva_size) // 2)
        big_img[int(900 / scale_factor):int(900 / scale_factor) + canva_size,
                w_begin:w_begin + canva_size, :] = occ_canvas_resize

        if args.format == "image":
            out_dir = osp.join(args.save_path, scene_name, sample_token)
            os.makedirs(out_dir, exist_ok=True)
            for i, img in enumerate(imgs):
                cv2.imwrite(osp.join(out_dir, f"img{i}.png"), img)
            cv2.imwrite(osp.join(out_dir, "occ.png"), occ_canvas)
            cv2.imwrite(osp.join(out_dir, "overall.png"), big_img)
        else:
            cv2.putText(big_img, f"{cnt}", (5, 15),
                        fontFace=cv2.FONT_HERSHEY_COMPLEX, color=(0, 0, 0),
                        fontScale=0.5)
            cv2.putText(big_img, scene_name, (5, 35),
                        fontFace=cv2.FONT_HERSHEY_COMPLEX, color=(0, 0, 0),
                        fontScale=0.5)
            cv2.putText(big_img, sample_token[:5], (5, 55),
                        fontFace=cv2.FONT_HERSHEY_COMPLEX, color=(0, 0, 0),
                        fontScale=0.5)
            vout.write(big_img)

    if args.format == "video":
        vout.release()
    vis.destroy_window()


if __name__ == "__main__":
    main()
