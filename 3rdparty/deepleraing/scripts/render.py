"""Rendering library for the visualization entry points.

det/seg: camera-panel + BEV matplotlib composites and GIF assembly
(BEVFusion visualize protocol). occ: class-colored BEV rasters, ground-plane
camera back-projection, lidar height raster and the 6-cam collage
(FlashOCC analysis_tools protocol). All occ BEV images share one orientation:
+x (ego forward) up, +y left, 40 m x 40 m around the ego vehicle.
"""

import os.path as osp

import cv2
import numpy as np
import torch
import matplotlib
from pyquaternion import Quaternion

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image, ImageDraw

from datasets.definitions import (MAP_CLASSES, OBJECT_CLASSES, OCC_CLASS_NAMES,
                                  OCC_COLOR_MAP)

DET_CLASSES = list(OBJECT_CLASSES)
DET_COLORS = [
    "#1f77b4", "#ff7f0e", "#8c564b", "#d62728", "#9467bd", "#7f7f7f",
    "#bcbd22", "#17becf", "#55c255", "#e377c2",
]

SEG_CLASSES = list(MAP_CLASSES)
SEG_COLORS = [
    "#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3", "#ccb974",
]
# draw rare classes on top of common ones
SEG_DRAW_ORDER = [0, 4, 2, 5, 1, 3]

IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# LiDARInstance3DBoxes.corners order:
# (x0y0z0, x0y0z1, x0y1z1, x0y1z0, x1y0z0, x1y0z1, x1y1z1, x1y1z0)
# bottom ring: 0-3-7-4; top ring: 1-2-6-5; verticals: (0,1)(3,2)(7,6)(4,5)
BOX_EDGES = [
    (0, 3), (3, 7), (7, 4), (4, 0),
    (1, 2), (2, 6), (6, 5), (5, 1),
    (0, 1), (3, 2), (7, 6), (4, 5),
]
BEV_RING = [0, 3, 7, 4]  # bottom-face polygon in order
FRONT_PAIR = (4, 7)  # front (x1-side) bottom edge → heading



def denormalize_img(img: torch.Tensor) -> Image.Image:
    """ (3,H,W) normalized tensor -> RGB PIL image."""
    arr = img.detach().float().cpu().numpy().transpose(1, 2, 0)
    arr = arr * IMG_STD + IMG_MEAN
    arr = np.clip(arr, 0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8))


def project_corners(corners: np.ndarray, P: np.ndarray):
    """ (N,8,3) lidar corners + (4,4) projection -> uv (N,8,2), depth (N,8)."""
    N = corners.shape[0]
    homo = np.concatenate([corners.reshape(-1, 3), np.ones((N * 8, 1))], axis=1)
    p = (P @ homo.T).T.reshape(N, 8, 4)  # (N,8,4)
    depth = p[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = p[..., :2] / np.maximum(depth[..., None], 1e-6)
    return uv, depth


def draw_boxes_on_image(
    pil_img: Image.Image,
    corners: np.ndarray,
    P: np.ndarray,
    colors,
    labels_text=None,
    line_width=2,
) -> Image.Image:
    """ Draw projected 3D boxes (12 edges) onto a camera image."""
    img = pil_img.copy()
    draw = ImageDraw.Draw(img)
    W, H = img.size
    if corners.shape[0] == 0:
        return img
    uv, depth = project_corners(corners, P)
    # official convention: keep a box only if ALL 8 corners are in front
    for i in range(corners.shape[0]):
        if not np.all(depth[i] > 0):
            continue
        for a, b in BOX_EDGES:
            draw.line(
                [tuple(uv[i, a]), tuple(uv[i, b])],
                fill=colors[i],
                width=line_width,
            )
        if labels_text is not None:
            vis = [k for k in range(8) if 0 <= uv[i, k, 0] < W and 0 <= uv[i, k, 1] < H]
            if vis:
                k = vis[0]
                draw.text((uv[i, k, 0] + 3, uv[i, k, 1] - 12), labels_text[i], fill=colors[i])
    return img


def bev_to_pil(fig) -> Image.Image:
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    img = Image.fromarray(np.asarray(buf)[:, :, :3])
    plt.close(fig)
    return img


def render_det_bev(
    points: np.ndarray,
    corners: np.ndarray,
    centers: np.ndarray,
    colors,
    labels_text=None,
    pc_range=54.0,
    size=(640, 640),
    gt_corners: np.ndarray = None,
) -> Image.Image:
    """ BEV scatter of lidar + rotated box outlines with heading ticks.

    corners: (N, 8, 3) full 3D corners (LiDARInstance3DBoxes.corners order).
    """
    dpi = 100
    fig = plt.figure(figsize=(size[0] / dpi, size[1] / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("#101010")
    fig.patch.set_facecolor("#101010")

    if points.shape[0] > 40000:
        sel = np.random.choice(points.shape[0], 40000, replace=False)
        points = points[sel]
    keep = (np.abs(points[:, 0]) < pc_range) & (np.abs(points[:, 1]) < pc_range)
    # bright colormap so low-z points stay visible after GIF palette quantization
    ax.scatter(
        points[keep, 0], points[keep, 1], s=0.7, c=points[keep, 2], cmap="plasma",
        vmin=-2.5, vmax=2.5, marker=".", linewidths=0, alpha=0.85,
    )

    for i in range(corners.shape[0]):
        poly = corners[i][BEV_RING][:, :2]
        ax.plot(
            np.append(poly[:, 0], poly[0, 0]),
            np.append(poly[:, 1], poly[0, 1]),
            color=colors[i], lw=1.4, zorder=3,
        )
        # heading tick: center -> front edge midpoint
        front_mid = (corners[i][FRONT_PAIR[0], :2] + corners[i][FRONT_PAIR[1], :2]) / 2
        ax.plot(
            [centers[i, 0], front_mid[0]], [centers[i, 1], front_mid[1]],
            color=colors[i], lw=1.0, zorder=4,
        )
        if labels_text is not None:
            ax.text(centers[i, 0], centers[i, 1] + 0.8, labels_text[i],
                    color=colors[i], fontsize=5.5, zorder=5)

    if gt_corners is not None:
        for i in range(gt_corners.shape[0]):
            poly = gt_corners[i][BEV_RING][:, :2]
            ax.plot(
                np.append(poly[:, 0], poly[0, 0]),
                np.append(poly[:, 1], poly[0, 1]),
                color="white", lw=1.0, ls="--", alpha=0.9, zorder=2,
            )

    # legend for classes present in this frame
    if labels_text:
        seen = {}
        for lt, c in zip(labels_text, colors):
            seen.setdefault(lt.split(" ")[0], c)
        for j, (name, c) in enumerate(seen.items()):
            ax.text(
                -pc_range + 1.5, pc_range - 3 - j * 2.6, f"■ {name}",
                color=c, fontsize=7, zorder=7,
            )

    # ego marker
    ax.plot([0], [0], marker="^", color="red", markersize=6, zorder=6)
    ax.set_xlim(-pc_range, pc_range)
    ax.set_ylim(-pc_range, pc_range)
    ax.set_aspect("equal")
    ax.axis("off")
    return bev_to_pil(fig)


def mask_to_rgba(masks: np.ndarray) -> np.ndarray:
    """ (6,200,200) 0/1 -> (200,200,4) RGBA float in [0,1], x-first convention:
    masks[k][x_idx, y_idx]; returned image rows = y, cols = x for imshow
    with extent (x, y)."""
    H = masks.shape[1]
    W = masks.shape[2]
    rgba = np.zeros((H, W, 4), dtype=np.float32)
    for k in SEG_DRAW_ORDER:
        if k >= masks.shape[0]:
            continue
        m = masks[k].astype(bool)  # (x, y)
        color = np.array(
            [int(SEG_COLORS[k][i : i + 2], 16) / 255.0 for i in (1, 3, 5)] + [0.62],
            dtype=np.float32,
        )
        m_t = m.T  # (y, x) for imshow
        rgba[m_t] = color
    return rgba


def render_seg_bev(points: np.ndarray, masks: np.ndarray, title: str, size=(520, 520), pc_range=50.0) -> Image.Image:
    dpi = 100
    fig = plt.figure(figsize=(size[0] / dpi, size[1] / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0.04, 1, 0.92])
    ax.set_facecolor("#101010")
    fig.patch.set_facecolor("#101010")

    if points.shape[0] > 20000:
        sel = np.random.choice(points.shape[0], 20000, replace=False)
        points = points[sel]
    keep = (np.abs(points[:, 0]) < pc_range) & (np.abs(points[:, 1]) < pc_range)
    ax.scatter(points[keep, 0], points[keep, 1], s=0.5, c="#666666", marker=".", linewidths=0)

    rgba = mask_to_rgba(masks)
    ax.imshow(
        rgba,
        extent=(-pc_range, pc_range, -pc_range, pc_range),
        origin="lower",
        interpolation="nearest",
        zorder=2,
    )
    ax.plot([0], [0], marker="^", color="red", markersize=6, zorder=6)
    ax.set_xlim(-pc_range, pc_range)
    ax.set_ylim(-pc_range, pc_range)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, color="white", fontsize=8, pad=2)
    return bev_to_pil(fig)


def compose_det_frame(
    cam_imgs,
    cam_corners,
    cam_P,
    cam_colors,
    cam_labels,
    bev_img: Image.Image,
    title: str,
    cam_width=480,
    cam_names=None,
) -> Image.Image:
    """ Left: BEV; right: all camera views (2 cols x N/2 rows) with projected
    3D boxes; title bar."""
    title_h = 26
    bev = bev_img
    W = bev.size[0]
    cams = []
    for img, corners, P, colors, labels in zip(cam_imgs, cam_corners, cam_P, cam_colors, cam_labels):
        im = draw_boxes_on_image(img, corners, P, colors, labels)
        scale = cam_width / im.size[0]
        im = im.resize((cam_width, max(1, int(im.size[1] * scale))))
        if cam_names is not None:
            d = ImageDraw.Draw(im)
            d.text((4, 2), cam_names[len(cams)], fill="#ffff00")
        cams.append(im)
    n = len(cams)
    cw = cams[0].size[0]
    ch = cams[0].size[1]
    gap = 4
    if n == 6:
        # three rows: front (FL,F,FR) / BEV centered / back (BL,B,BR)
        grid_w = 3 * cw + 2 * gap
        bev_sz = min(grid_w, 520)
        bev_p = bev.resize((bev_sz, bev_sz))
        row2_h = bev_sz
        H = title_h + ch + gap + row2_h + gap + ch
        frame = Image.new("RGB", (grid_w, H), "#181818")
        for i in range(3):
            frame.paste(cams[i], (i * (cw + gap), title_h))
        frame.paste(bev_p, ((grid_w - bev_sz) // 2, title_h + ch + gap))
        for i in range(3):
            frame.paste(cams[3 + i], (i * (cw + gap), title_h + ch + gap + row2_h + gap))
    else:
        # fallback: BEV left, cameras in a grid on the right
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols
        grid_w = ncols * cw + (ncols - 1) * gap
        grid_h = nrows * ch + (nrows - 1) * gap
        H = max(bev.size[1], grid_h) + title_h
        frame = Image.new("RGB", (W + grid_w + 8, H), "#181818")
        frame.paste(bev, (0, title_h))
        for i, c in enumerate(cams):
            r, col = divmod(i, ncols)
            frame.paste(c, (W + 8 + col * (cw + gap), title_h + r * (ch + gap)))
    draw = ImageDraw.Draw(frame)
    draw.text((8, 7), title, fill="#f0f0f0")
    return frame


def compose_seg_frame(cam_img: Image.Image, pred_bev: Image.Image, gt_bev: Image.Image, title: str) -> Image.Image:
    title_h = 26
    W = pred_bev.size[0] + gt_bev.size[0] + 8
    H = max(pred_bev.size[1], gt_bev.size[1], cam_img.size[1]) + title_h
    # scale camera to combined width
    scale = W / cam_img.size[0]
    cam = cam_img.resize((W, max(1, int(cam_img.size[1] * scale))))
    H = cam.size[1] + pred_bev.size[1] + title_h
    frame = Image.new("RGB", (W, H), "#181818")
    frame.paste(cam, (0, title_h))
    frame.paste(pred_bev, (0, title_h + cam.size[1]))
    frame.paste(gt_bev, (pred_bev.size[0] + 8, title_h + cam.size[1]))
    draw = ImageDraw.Draw(frame)
    draw.text((8, 7), title, fill="#f0f0f0")
    # legend
    lx = pred_bev.size[0] + 14
    ly = title_h + cam.size[1] + 6
    for k, name in enumerate(SEG_CLASSES):
        draw.rectangle([lx, ly + k * 13, lx + 10, ly + k * 13 + 9], fill=SEG_COLORS[k])
        draw.text((lx + 14, ly + k * 13 - 1), name, fill="#dddddd")
    return frame


def save_gif(frames, path: str, fps: int = 4):
    """ Save PIL frames as an animated GIF."""
    assert len(frames) > 0
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),
        loop=0,
        optimize=False,
    )


# ---- occupancy BEV renderers ----

FREE_LABEL = len(OCC_CLASS_NAMES) - 1  # 17
COLORS = np.array(OCC_COLOR_MAP, dtype=np.uint8)  # RGB, 18 rows

VIEWS_TOP = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"]
VIEWS_BOT = ["CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]


def render_bev(occ, size):
    """ (Dx, Dy, Dz) labels -> (size, size, 3) BGR BEV image.

    z-collapse: per (x, y) column the top-most non-free voxel wins; empty
    columns render as free. Double flip gives +x-up / +y-left orientation.
    """
    occ = np.asarray(occ)
    dx, dy, dz = occ.shape
    valid = occ != FREE_LABEL
    z_idx = np.broadcast_to(np.arange(dz).reshape(1, 1, dz), (dx, dy, dz)) * valid
    top_z = np.argmax(z_idx, axis=2)
    bev = np.take_along_axis(occ, top_z[..., None], axis=2)[..., 0]
    bev = np.where(valid.any(axis=2), bev, FREE_LABEL)
    rgb = COLORS[bev.reshape(-1).astype(np.int32)].reshape(dx, dy, 3)
    img = rgb[::-1, ::-1, :]
    img = cv2.resize(img.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)
    return np.ascontiguousarray(img[..., ::-1])  # RGB -> BGR


def render_image_bev(info, size, z0=-1.7, rng=40.0, imgs=None):
    """ BEV-ize the raw camera images via ground-plane back-projection.

    Each BEV cell (x, y, z0) is projected into all 6 cameras; the pixel of
    the camera whose optical axis is closest to the point is sampled.
    imgs: optional {view: BGR array} to reuse already-loaded frames.
    """
    n = size
    xs = np.linspace(-rng + rng / n, rng - rng / n, n)
    gx, gy = np.meshgrid(xs, xs, indexing="ij")
    p = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, z0), np.ones(gx.size)], -1)
    color = np.zeros((gx.size, 3), np.uint8)
    score = np.full(gx.size, np.inf)
    # The occ grid frame is vehicle-aligned, so BEV cells are projected with
    # the camera sensor2ego pose directly (the raw pkl lidar2ego quaternion
    # encodes the un-corrected sensor mount and must NOT be applied here).
    for v in VIEWS_TOP + VIEWS_BOT:
        cam = info["cams"][v]
        img = imgs[v] if imgs is not None else cv2.imread(cam["data_path"])
        if img is None:
            continue
        H, W = img.shape[:2]
        R_s2e = Quaternion(cam["sensor2ego_rotation"]).rotation_matrix
        t_s2e = np.asarray(cam["sensor2ego_translation"], np.float64)
        K = np.asarray(cam["cam_intrinsic"], np.float64)
        pc = (p[:, :3] - t_s2e) @ R_s2e  # vehicle frame -> cam
        zc = pc[:, 2]
        uv = (K @ pc.T).T
        with np.errstate(divide="ignore", invalid="ignore"):
            u = np.where(zc > 0.5, uv[:, 0] / np.maximum(zc, 1e-6), -1)
            w = np.where(zc > 0.5, uv[:, 1] / np.maximum(zc, 1e-6), -1)
        ok = (u >= 0) & (u < W - 1) & (w >= 0) & (w < H - 1)
        s = np.where(ok, np.abs(u - W / 2) / W + np.abs(w - H / 2) / H, np.inf)
        upd = s < score
        idx = np.where(upd)[0]
        if idx.size:
            color[idx] = img[w[idx].astype(int), u[idx].astype(int)]
            score[idx] = s[idx]
    bev = color.reshape(n, n, 3)[::-1, ::-1]  # same orientation as render_bev
    return np.ascontiguousarray(bev)


def render_lidar_bev(info, size, rng=40.0):
    """ Top-down raster of LIDAR_TOP points, colored by height (turbo)."""
    path = info["lidar_path"]
    if not osp.exists(path):
        return np.zeros((size, size, 3), np.uint8)
    pts = np.fromfile(path, dtype=np.float32).reshape(-1, 5)  # x,y,z,i,ring
    keep = (np.abs(pts[:, 0]) < rng) & (np.abs(pts[:, 1]) < rng)
    pts = pts[keep]
    if not pts.size:
        return np.zeros((size, size, 3), np.uint8)
    ix = ((pts[:, 0] + rng) / (2 * rng) * size).astype(int).clip(0, size - 1)
    iy = ((pts[:, 1] + rng) / (2 * rng) * size).astype(int).clip(0, size - 1)
    zmax = np.full((size, size), -np.inf, np.float32)
    np.maximum.at(zmax, (ix, iy), pts[:, 2])
    cnt = np.zeros((size, size), np.int32)
    np.add.at(cnt, (ix, iy), 1)
    empty = cnt == 0
    z = np.where(empty, 0.0, zmax)
    zn = np.clip((z + 2.0) / 4.0, 0, 1)  # -2 m .. +2 m
    bev = cv2.applyColorMap((zn * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    bev[empty] = (0, 0, 0)
    bev = bev[::-1, ::-1]
    return np.ascontiguousarray(bev)


def camera_collage(imgs, cam_w=None, with_labels=True):
    """ 6 BGR views (FL/F/FR/BL/B/BR, dict by view or list) -> 2x3 collage.

    Bottom row mirrored so the surround reads as one panorama; cam_w=None
    keeps native resolution (all 6 views share a shape).
    """
    cam_h = None if cam_w is None else int(round(cam_w * 900 / 1600))
    rows = []
    for i, (views, mirror) in enumerate(((VIEWS_TOP, False), (VIEWS_BOT, True))):
        tiles = []
        for j, v in enumerate(views):
            img = imgs[v] if isinstance(imgs, dict) else imgs[i * 3 + j]
            if img is None:
                img = np.zeros((cam_h or 270, cam_w or 480, 3), np.uint8)
            if cam_w is not None:
                img = cv2.resize(img, (cam_w, cam_h))
            if mirror:
                img = np.ascontiguousarray(img[:, ::-1])
            if with_labels:
                label(img, v.replace("CAM_", ""), (6, 20), 0.55)
            tiles.append(img)
        rows.append(np.concatenate(tiles, axis=1))
    return np.concatenate(rows, axis=0)


def surround_rgb(imgs):
    """ (6, 3, H, W) normalized imgs -> denormalized BGR surround strip.

    Inverse of the BGR-quirk normalization: channel 0 denormalizes to B, so
    the array is already BGR. Rows: FL/F/FR over (mirrored) BL/B/BR.
    """
    from datasets.preprocess.loading import MEAN, STD

    img = imgs.cpu().numpy().astype(np.float32) if hasattr(imgs, "cpu") else np.asarray(imgs, np.float32)
    img = img * STD.reshape(1, 3, 1, 1) + MEAN.reshape(1, 3, 1, 1)
    img = np.clip(img, 0, 255).astype(np.uint8).transpose(0, 2, 3, 1)
    return camera_collage(list(img), with_labels=False)


def make_legend(row_h, n_entries=17):
    """ Vertical class legend strip (BGR)."""
    w = 230
    h = max(row_h, n_entries * 18 + 8)
    canvas = np.full((h, w, 3), 255, np.uint8)
    for i in range(n_entries):
        y = 6 + i * 18
        canvas[y:y + 12, 6:18] = COLORS[i][::-1].astype(np.uint8)
        cv2.putText(canvas, OCC_CLASS_NAMES[i], (24, y + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas


def label(img, text, org=(8, 24), scale=0.7):
    """ Outlined text (white halo + black core) readable on any background."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)
    return img
