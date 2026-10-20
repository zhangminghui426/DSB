"""Preprocessing — loading transforms: images / point clouds / BEV-seg GT / depth GT."""

import glob
import logging
import os
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pyquaternion import Quaternion

from datasets.devkit_compat import apply_compat
from datasets.geometry.points import LiDARPoints

logger = logging.getLogger(__name__)


def reduce_LiDAR_beams(pts, reduce_beams_to=32):
    """Reduce LiDAR beams to simulate lower-beam sensors.

    Args:
        pts: Points array (numpy or torch) with shape (N, D).
        reduce_beams_to: Target number of beams (16, 4, or 1).

    Returns:
        numpy.ndarray: Filtered points.
    """
    if isinstance(pts, np.ndarray):
        pts = torch.from_numpy(pts)
    radius = torch.sqrt(pts[:, 0].pow(2) + pts[:, 1].pow(2) + pts[:, 2].pow(2))
    sine_theta = pts[:, 2] / radius
    theta = torch.asin(sine_theta.clamp(-1, 1))
    phi = torch.atan2(pts[:, 1], pts[:, 0])

    top_ang = 0.1862
    down_ang = -0.5353

    beam_range = torch.zeros(32)
    beam_range[0] = top_ang
    beam_range[31] = down_ang

    for i in range(1, 31):
        beam_range[i] = beam_range[i - 1] - 0.023275

    num_pts = pts.size(0)
    mask = torch.zeros(num_pts)
    if reduce_beams_to == 16:
        for id in [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]:
            beam_mask = (theta < (beam_range[id - 1] - 0.012)) * (
                theta > (beam_range[id] - 0.012)
            )
            mask = mask + beam_mask
        mask = mask.bool()
    elif reduce_beams_to == 4:
        for id in [7, 9, 11, 13]:
            beam_mask = (theta < (beam_range[id - 1] - 0.012)) * (
                theta > (beam_range[id] - 0.012)
            )
            mask = mask + beam_mask
        mask = mask.bool()
    elif reduce_beams_to == 1:
        chosen_beam_id = 9
        mask = (theta < (beam_range[chosen_beam_id - 1] - 0.012)) * (
            theta > (beam_range[chosen_beam_id] - 0.012)
        )
    else:
        raise NotImplementedError(
            f"reduce_beams_to={reduce_beams_to} not supported"
        )
    points = pts[mask]
    return points.numpy()


class LoadMultiViewImageFromFiles:
    """Load multi-view images from files using PIL."""

    def __init__(self, to_float32=False, color_type="unchanged"):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        filename = results["image_paths"]
        images = []
        for name in filename:
            images.append(Image.open(name))

        results["filename"] = filename
        results["img"] = images
        results["img_shape"] = images[0].size  # (W, H)
        results["ori_shape"] = images[0].size
        results["pad_shape"] = images[0].size
        results["scale_factor"] = 1.0
        return results


class LoadPointsFromFile:
    """Load point cloud from file."""

    def __init__(
        self,
        coord_type,
        load_dim=6,
        use_dim=None,
        shift_height=False,
        use_color=False,
        load_augmented=None,
        reduce_beams=None,
    ):
        self.coord_type = coord_type
        self.load_dim = load_dim
        if use_dim is None:
            use_dim = list(range(load_dim))
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        self.use_dim = use_dim
        self.shift_height = shift_height
        self.use_color = use_color
        self.load_augmented = load_augmented
        self.reduce_beams = reduce_beams

    def __call__(self, results):
        lidar_path = results["lidar_path"]
        if not os.path.exists(lidar_path):
            raise FileNotFoundError(f"Lidar file not found: {lidar_path}")

        if lidar_path.endswith(".npy"):
            points = np.load(lidar_path)
        else:
            points = np.fromfile(lidar_path, dtype=np.float32)

        points = points.reshape(-1, self.load_dim)

        # reduce beams if needed
        if self.reduce_beams is not None and self.reduce_beams < 32:
            points = reduce_LiDAR_beams(points, self.reduce_beams)

        points = points[:, self.use_dim]

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3], np.expand_dims(height, 1), points[:, 3:]], 1
            )

        results["points"] = LiDARPoints(points, points_dim=points.shape[-1])
        return results


class LoadPointsFromMultiSweeps:
    """Load points from multiple sweeps."""

    def __init__(
        self,
        sweeps_num=10,
        load_dim=5,
        use_dim=None,
        pad_empty_sweeps=False,
        remove_close=False,
        test_mode=False,
        load_augmented=None,
        reduce_beams=None,
    ):
        self.load_dim = load_dim
        self.sweeps_num = sweeps_num
        if use_dim is None:
            use_dim = list(range(load_dim))
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        self.use_dim = use_dim
        self.pad_empty_sweeps = pad_empty_sweeps
        self.remove_close = remove_close
        self.test_mode = test_mode
        self.load_augmented = load_augmented
        self.reduce_beams = reduce_beams

    def load_points(self, lidar_path):
        if lidar_path.endswith(".npy"):
            return np.load(lidar_path)
        return np.fromfile(lidar_path, dtype=np.float32)

    def remove_close_points(self, points, radius=1.0):
        if isinstance(points, np.ndarray):
            pts = points
        else:
            pts = points.tensor.numpy()
        x_filt = np.abs(pts[:, 0]) < radius
        y_filt = np.abs(pts[:, 1]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        return points[not_close]

    def __call__(self, results):
        points = results["points"]
        points = points[:, self.use_dim]
        points.tensor[:, 4] = 0
        sweep_points_list = [points]
        ts = results["timestamp"] / 1e6

        if self.pad_empty_sweeps and len(results["sweeps"]) == 0:
            for _ in range(self.sweeps_num):
                if self.remove_close:
                    sweep_points_list.append(self.remove_close_points(points))
                else:
                    sweep_points_list.append(points)
        else:
            if len(results["sweeps"]) <= self.sweeps_num:
                choices = np.arange(len(results["sweeps"]))
            elif self.test_mode:
                choices = np.arange(self.sweeps_num)
            else:
                if not self.load_augmented:
                    choices = np.random.choice(
                        len(results["sweeps"]), self.sweeps_num, replace=False
                    )
                else:
                    choices = np.random.choice(
                        len(results["sweeps"]) - 1, self.sweeps_num, replace=False
                    )

            for idx in choices:
                sweep = results["sweeps"][idx]
                # sweep paths are already rebased in NuScenesDataset.get_data_info
                points_sweep = self.load_points(sweep["data_path"])
                points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)

                if self.reduce_beams is not None and self.reduce_beams < 32:
                    points_sweep = reduce_LiDAR_beams(
                        points_sweep, self.reduce_beams
                    )

                if self.remove_close:
                    x_filt = np.abs(points_sweep[:, 0]) < 1.0
                    y_filt = np.abs(points_sweep[:, 1]) < 1.0
                    not_close = np.logical_not(np.logical_and(x_filt, y_filt))
                    points_sweep = points_sweep[not_close]

                points_sweep = points_sweep[:, self.use_dim]
                sweep_ts = sweep["timestamp"] / 1e6
                points_sweep[:, :3] = (
                    points_sweep[:, :3] @ sweep["sensor2lidar_rotation"].T
                )
                points_sweep[:, :3] += sweep["sensor2lidar_translation"]
                points_sweep[:, 4] = ts - sweep_ts
                sweep_pt = LiDARPoints(
                    points_sweep, points_dim=points_sweep.shape[-1]
                )
                sweep_points_list.append(sweep_pt)

        points = LiDARPoints.cat(sweep_points_list)
        results["points"] = points
        return results


class LoadBEVSegmentation:
    """Load BEV segmentation masks -> gt_masks_bev (C, H, W) int64.

    Backends:
    - 'nuscenes_map' (default): rasterize NuScenesMap layers at runtime.
      Requires maps/expansion/*.json (map expansion pack) under dataset_root.
    - 'occ_npz': approximate fallback derived from Occ3D-style occupancy GT
      (gts/scene-*/{token}/labels.npz with 'semantics' class-id voxels).
      Only 2 of the 6 bevfusion map classes can be derived
      (drivable_area <- driveable_surface, walkway <- sidewalk);
      ped_crossing / stop_line / carpark_area / divider stay EMPTY.
      Use only for pipeline smoke tests without the map expansion pack.
    """

    # Occ3D-nuScenes (occ34) semantic class ids
    OCC_DRIVEABLE_SURFACE = 11
    OCC_SIDEWALK = 13

    # Class-level cache for npz index
    _npz_index_cache = {}

    def __init__(
        self,
        dataset_root: str,
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        classes: Tuple[str, ...],
        backend: str = "nuscenes_map",
    ):
        self.dataset_root = dataset_root
        patch_h = ybound[1] - ybound[0]
        patch_w = xbound[1] - xbound[0]
        canvas_h = int(patch_h / ybound[2])
        canvas_w = int(patch_w / xbound[2])
        self.patch_size = (patch_h, patch_w)
        self.canvas_size = (canvas_h, canvas_w)
        self.classes = classes
        self.xbound = xbound
        self.ybound = ybound
        self.backend = backend

        if backend == "nuscenes_map":
            apply_compat()  # shapely 2.x fixes for devkit map rasterization
            from nuscenes.map_expansion.map_api import NuScenesMap
            from nuscenes.map_expansion.map_api import locations as LOCATIONS

            self.maps = {}
            for location in LOCATIONS:
                self.maps[location] = NuScenesMap(dataset_root, location)

    def build_npz_index(self):
        """Build token -> npz_path index from gts/ directory."""
        cache_key = self.dataset_root
        if cache_key in LoadBEVSegmentation._npz_index_cache:
            return LoadBEVSegmentation._npz_index_cache[cache_key]

        gts_dir = os.path.join(self.dataset_root, "gts")
        index = {}
        if os.path.exists(gts_dir):
            npz_files = glob.glob(os.path.join(gts_dir, "*", "*", "labels.npz"))
            for fpath in npz_files:
                token = os.path.basename(os.path.dirname(fpath))
                index[token] = fpath

        LoadBEVSegmentation._npz_index_cache[cache_key] = index
        return index

    def __call__(self, data):
        token = data.get("token", "")

        if self.backend in ("npz", "occ_npz"):
            return self.load_npz(data, token)
        elif self.backend == "nuscenes_map":
            return self.load_nuscenes_map(data)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def load_npz(self, data, token):
        """Approximate BEV mask from Occ3D-style occupancy GT.

        labels.npz['semantics'] is (X?, Y?, Z=16) uint8 class ids (occ34:
        11=driveable_surface, 13=sidewalk, 17=free/invalid). We project to
        BEV (any-hit over z) and map the two derivable classes; the other
        four bevfusion map classes have NO occupancy counterpart and stay 0.
        Grid orientation: transpose + flip-x (empirically validated against
        lidar point coverage in tests/test_dataset.py).
        """
        logger.warning(
            "LoadBEVSegmentation occ_npz backend: only drivable_area/walkway "
            "are derivable from occupancy GT; other map classes are empty. "
            "Prefer backend='nuscenes_map' with the map expansion pack."
        )
        index = self.build_npz_index()
        num_classes = len(self.classes)
        if token not in index:
            data["gt_masks_bev"] = np.zeros(
                (num_classes, *self.canvas_size), dtype=np.int64
            )
            return data

        npz_data = np.load(index[token])
        semantics = npz_data["semantics"]  # (200, 200, 16) uint8 class ids

        # BEV projection: any voxel hit along z
        occ_class_ids = {
            "drivable_area": self.OCC_DRIVEABLE_SURFACE,
            "drivable_area*": self.OCC_DRIVEABLE_SURFACE,
            "walkway": self.OCC_SIDEWALK,
        }
        labels = np.zeros((num_classes, *self.canvas_size), dtype=np.int64)
        for k, name in enumerate(self.classes):
            cid = occ_class_ids.get(name)
            if cid is None:
                continue  # not derivable from occupancy GT
            bev = (semantics == cid).any(axis=2).astype(np.int64)  # (200, 200)
            # orientation: transpose + flip first axis (validated in tests)
            labels[k] = bev.T[::-1, :]

        # Warp mask by lidar_aug_matrix if present
        lidar_aug_matrix = data.get("lidar_aug_matrix", None)
        if lidar_aug_matrix is not None:
            labels = self.warp_mask(labels, lidar_aug_matrix)

        data["gt_masks_bev"] = labels
        return data

    def warp_mask(self, labels, lidar_aug_matrix):
        """Warp BEV mask by inverse lidar augmentation matrix.

        Maps augmented-frame grid cells back to unaugmented frame.
        """
        if isinstance(lidar_aug_matrix, np.ndarray):
            A = torch.from_numpy(lidar_aug_matrix).float()
        else:
            A = lidar_aug_matrix.float()

        # Check if identity
        if torch.allclose(A, torch.eye(4), atol=1e-6):
            return labels

        inv_A = torch.inverse(A)

        C, H, W = labels.shape
        mask_tensor = torch.from_numpy(labels).float().unsqueeze(0)  # (1, C, H, W)

        # Build grid: for each output cell, find where to sample in input
        # Grid spans [-50, 50] in both x and y
        x_min, x_max = self.xbound[0], self.xbound[1]
        y_min, y_max = self.ybound[0], self.ybound[1]

        # Create grid of augmented-frame coordinates
        xs = torch.linspace(
            x_min + self.xbound[2] / 2,
            x_max - self.xbound[2] / 2,
            W,
        )
        ys = torch.linspace(
            y_min + self.ybound[2] / 2,
            y_max - self.ybound[2] / 2,
            H,
        )
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

        # Augmented frame coords -> unaugmented frame via inv(A)
        ones = torch.ones_like(grid_x)
        aug_coords = torch.stack([grid_x, grid_y, torch.zeros_like(grid_x), ones], dim=-1)
        # Apply inverse augmentation (BEV part: 2D affine)
        unaug_coords = aug_coords @ inv_A.T  # (H, W, 4)

        # Normalize to [-1, 1] for grid_sample
        norm_x = (unaug_coords[..., 0] - x_min) / (x_max - x_min) * 2 - 1
        norm_y = (unaug_coords[..., 1] - y_min) / (y_max - y_min) * 2 - 1

        # grid_sample expects (N, H, W, 2) with (x, y) = (W, H) order
        grid = torch.stack([norm_y, norm_x], dim=-1).unsqueeze(0)  # (1, H, W, 2)

        warped = F.grid_sample(
            mask_tensor, grid, mode="nearest", padding_mode="zeros",
            align_corners=False,
        )
        labels = warped.squeeze(0).numpy().astype(np.int64)
        return labels

    def load_nuscenes_map(self, data):
        """Load BEV mask using NuScenesMap API."""
        location = data.get("location", None)
        if location is None or location not in self.maps:
            num_classes = len(self.classes)
            labels = np.zeros(
                (num_classes, *self.canvas_size), dtype=np.int64
            )
            data["gt_masks_bev"] = labels
            return data

        lidar2point = data["lidar_aug_matrix"]
        point2lidar = np.linalg.inv(lidar2point)
        lidar2ego = data["lidar2ego"]
        ego2global = data["ego2global"]
        lidar2global = ego2global @ lidar2ego @ point2lidar

        map_pose = lidar2global[:2, 3]
        patch_box = (
            map_pose[0],
            map_pose[1],
            self.patch_size[0],
            self.patch_size[1],
        )

        rotation = lidar2global[:3, :3]
        v = np.dot(rotation, np.array([1, 0, 0]))
        yaw = np.arctan2(v[1], v[0])
        patch_angle = yaw / np.pi * 180

        mappings = {}
        for name in self.classes:
            if name == "drivable_area*":
                mappings[name] = ["road_segment", "lane"]
            elif name == "divider":
                mappings[name] = ["road_divider", "lane_divider"]
            else:
                mappings[name] = [name]

        layer_names = []
        for name in mappings:
            layer_names.extend(mappings[name])
        layer_names = list(set(layer_names))

        masks = self.maps[location].get_map_mask(
            patch_box=patch_box,
            patch_angle=patch_angle,
            layer_names=layer_names,
            canvas_size=self.canvas_size,
        )
        masks = masks.transpose(0, 2, 1)
        masks = masks.astype(bool)

        num_classes = len(self.classes)
        labels = np.zeros((num_classes, *self.canvas_size), dtype=np.int64)
        for k, name in enumerate(self.classes):
            for layer_name in mappings[name]:
                index = layer_names.index(layer_name)
                labels[k, masks[index]] = 1

        data["gt_masks_bev"] = labels
        return data


class GTDepth:
    """Project LiDAR points onto image planes -> sparse per-camera depth.

    Configured after Collect3D: consumes the collected dict and appends
    'depths' (6, H, W).
    """

    def __init__(self, keyframe_only=False):
        self.keyframe_only = keyframe_only

    def __call__(self, data):
        sensor2ego = np.array(data["camera2ego"])
        cam_intrinsic = np.array(data["camera_intrinsics"])
        img_aug_matrix = np.array(data["img_aug_matrix"])
        bev_aug_matrix = np.array(data["lidar_aug_matrix"])
        lidar2ego = np.array(data["lidar2ego"])
        camera2lidar = np.array(data["camera2lidar"])
        lidar2image = np.array(data["lidar2image"])

        # Convert to torch
        sensor2ego = torch.from_numpy(sensor2ego).float()
        cam_intrinsic = torch.from_numpy(cam_intrinsic).float()
        img_aug_matrix = torch.from_numpy(img_aug_matrix).float()
        bev_aug_matrix = torch.from_numpy(bev_aug_matrix).float()
        lidar2ego = torch.from_numpy(lidar2ego).float()
        camera2lidar = torch.from_numpy(camera2lidar).float()
        lidar2image = torch.from_numpy(lidar2image).float()

        points = data["points"]
        if isinstance(points, LiDARPoints):
            points = points.tensor
        elif isinstance(points, torch.Tensor):
            pass
        else:
            points = torch.from_numpy(np.asarray(points)).float()

        # Get img shape
        img = data["img"]
        if isinstance(img, list):
            if isinstance(img[0], torch.Tensor):
                img_h, img_w = img[0].shape[-2:]
            else:
                img_h, img_w = img[0].size[1], img[0].size[0]
        elif isinstance(img, torch.Tensor):
            img_h, img_w = img.shape[-2:]
        else:
            img_h, img_w = img.shape[-2:]

        if self.keyframe_only and points.shape[1] > 4:
            points = points[points[:, 4] == 0]

        num_cams = sensor2ego.shape[0]
        depth = torch.zeros(num_cams, img_h, img_w)

        cur_coords = points[:, :3]

        # inverse aug
        cur_coords = cur_coords - bev_aug_matrix[:3, 3]
        cur_coords = torch.inverse(bev_aug_matrix[:3, :3]).matmul(
            cur_coords.transpose(1, 0)
        )
        # lidar2image
        cur_coords = lidar2image[:, :3, :3].matmul(cur_coords)
        cur_coords += lidar2image[:, :3, 3].reshape(-1, 3, 1)
        # get 2d coords
        dist = cur_coords[:, 2, :].clone()
        cur_coords[:, 2, :] = torch.clamp(cur_coords[:, 2, :], 1e-5, 1e5)
        cur_coords[:, :2, :] /= cur_coords[:, 2:3, :]

        # imgaug
        cur_coords = img_aug_matrix[:, :3, :3].matmul(cur_coords)
        cur_coords += img_aug_matrix[:, :3, 3].reshape(-1, 3, 1)
        cur_coords = cur_coords[:, :2, :].transpose(1, 2)

        # normalize coords for grid sample
        cur_coords = cur_coords[..., [1, 0]]

        on_img = (
            (cur_coords[..., 0] < img_h)
            & (cur_coords[..., 0] >= 0)
            & (cur_coords[..., 1] < img_w)
            & (cur_coords[..., 1] >= 0)
        )
        for c in range(on_img.shape[0]):
            masked_coords = cur_coords[c, on_img[c]].long()
            masked_dist = dist[c, on_img[c]]
            depth[c, masked_coords[:, 0], masked_coords[:, 1]] = masked_dist

        data["depths"] = depth
        return data


# ---- occupancy (BEVDet/FlashOCC protocol; BGR-quirk normalization is
# load-bearing, see mmlab_normalize docstring) ----

MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


def mmlab_normalize(img):
    """PIL RGB image -> (3, H, W) float tensor, BGR-quirk normalized (see above)."""
    img = np.array(img).astype(np.float32)
    img = img[:, :, [2, 1, 0]]  # cv2.cvtColor(BGR2RGB) on RGB data == ch0<->ch2 swap
    img = (img - MEAN) / STD
    return torch.from_numpy(img).float().permute(2, 0, 1).contiguous()


class PrepareImageInputs:
    """Load + augment the 6 camera images; build the (imgs, sensor2egos,
    ego2globals, intrins, post_rots, post_trans) 6-tuple. sequential=True
    appends adjacent frames sharing the same augmentation (4D variants)."""

    def __init__(self, data_config, is_train=False, sequential=False):
        self.is_train = is_train
        self.data_config = data_config
        self.sequential = sequential

    def choose_cams(self):
        if self.is_train and self.data_config["Ncams"] < len(self.data_config["cams"]):
            return np.random.choice(
                self.data_config["cams"], self.data_config["Ncams"], replace=False
            )
        return self.data_config["cams"]

    def sample_augmentation(self, H, W, flip=None, scale=None):
        fH, fW = self.data_config["input_size"]
        if self.is_train:
            resize = float(fW) / float(W)
            resize += np.random.uniform(*self.data_config["resize"])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_config["crop_h"])) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = self.data_config["flip"] and np.random.choice([0, 1])
            rotate = np.random.uniform(*self.data_config["rot"])
        else:
            resize = float(fW) / float(W)
            resize += scale if scale is not None else self.data_config.get("resize_test", 0.0)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_config["crop_h"])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False if flip is None else flip
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def img_transform_core(self, img, resize_dims, crop, flip, rotate):
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        return img

    def get_rot(self, h):
        return torch.Tensor([[np.cos(h), np.sin(h)], [-np.sin(h), np.cos(h)]])

    def img_transform(self, img, post_rot, post_tran, resize, resize_dims, crop, flip, rotate):
        img = self.img_transform_core(img, resize_dims, crop, flip, rotate)

        # post-homography of the augmentation chain
        post_rot *= resize
        post_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            post_rot = A.matmul(post_rot)
            post_tran = A.matmul(post_tran) + b
        A = self.get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b
        return img, post_rot, post_tran

    def get_sensor_transforms(self, info, cam_name):
        """-> sensor2ego (4,4), ego2global (4,4) from quaternion records."""
        w, x, y, z = info["cams"][cam_name]["sensor2ego_rotation"]
        sensor2ego_rot = torch.Tensor(Quaternion(w, x, y, z).rotation_matrix)
        sensor2ego_tran = torch.Tensor(info["cams"][cam_name]["sensor2ego_translation"])
        sensor2ego = sensor2ego_rot.new_zeros((4, 4))
        sensor2ego[3, 3] = 1
        sensor2ego[:3, :3] = sensor2ego_rot
        sensor2ego[:3, -1] = sensor2ego_tran

        w, x, y, z = info["cams"][cam_name]["ego2global_rotation"]
        ego2global_rot = torch.Tensor(Quaternion(w, x, y, z).rotation_matrix)
        ego2global_tran = torch.Tensor(info["cams"][cam_name]["ego2global_translation"])
        ego2global = ego2global_rot.new_zeros((4, 4))
        ego2global[3, 3] = 1
        ego2global[:3, :3] = ego2global_rot
        ego2global[:3, -1] = ego2global_tran
        return sensor2ego, ego2global

    def get_inputs(self, results, flip=None, scale=None):
        imgs, sensor2egos, ego2globals = [], [], []
        intrins, post_rots, post_trans = [], [], []
        cam_names = self.choose_cams()
        results["cam_names"] = cam_names

        for cam_name in cam_names:
            cam_data = results["curr"]["cams"][cam_name]
            img = Image.open(cam_data["data_path"])

            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)
            intrin = torch.Tensor(cam_data["cam_intrinsic"])
            sensor2ego, ego2global = self.get_sensor_transforms(results["curr"], cam_name)

            resize, resize_dims, crop, flip, rotate = self.sample_augmentation(
                H=img.height, W=img.width, flip=flip, scale=scale
            )
            img, post_rot2, post_tran2 = self.img_transform(
                img, post_rot, post_tran,
                resize=resize, resize_dims=resize_dims, crop=crop, flip=flip, rotate=rotate,
            )

            # embed the 2x2 augmentation homography in 3x3
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            imgs.append(mmlab_normalize(img))

            if self.sequential:
                assert "adjacent" in results
                for adj_info in results["adjacent"]:
                    img_adjacent = Image.open(adj_info["cams"][cam_name]["data_path"])
                    img_adjacent = self.img_transform_core(
                        img_adjacent, resize_dims=resize_dims, crop=crop, flip=flip, rotate=rotate
                    )
                    imgs.append(mmlab_normalize(img_adjacent))

            intrins.append(intrin)
            sensor2egos.append(sensor2ego)
            ego2globals.append(ego2global)
            post_rots.append(post_rot)
            post_trans.append(post_tran)

        if self.sequential:
            for adj_info in results["adjacent"]:
                # adjacent frames share augmentation and intrinsics
                post_trans.extend(post_trans[: len(cam_names)])
                post_rots.extend(post_rots[: len(cam_names)])
                intrins.extend(intrins[: len(cam_names)])
                for cam_name in cam_names:
                    sensor2ego, ego2global = self.get_sensor_transforms(adj_info, cam_name)
                    sensor2egos.append(sensor2ego)
                    ego2globals.append(ego2global)

        return (
            torch.stack(imgs),
            torch.stack(sensor2egos),
            torch.stack(ego2globals),
            torch.stack(intrins),
            torch.stack(post_rots),
            torch.stack(post_trans),
        )

    def __call__(self, results):
        results["img_inputs"] = self.get_inputs(results)
        return results


class LoadAnnotationsBEVDepth:
    """Sample the BDA (BEV data augmentation) matrix and append it as the 7th
    img_inputs element; record flip flags for LoadOccGTFromFile.

    The occupancy flow consumes no detection boxes (PackOcc3DInputs drops
    them), so the original box transformation is intentionally omitted.
    bda = flip_mat @ (scale_mat @ rot_mat) — order matters.
    """

    def __init__(self, bda_aug_conf, is_train=True):
        self.bda_aug_conf = bda_aug_conf
        self.is_train = is_train

    def sample_bda_augmentation(self):
        if self.is_train:
            rotate_bda = np.random.uniform(*self.bda_aug_conf["rot_lim"])
            scale_bda = np.random.uniform(*self.bda_aug_conf["scale_lim"])
            flip_dx = np.random.uniform() < self.bda_aug_conf["flip_dx_ratio"]
            flip_dy = np.random.uniform() < self.bda_aug_conf["flip_dy_ratio"]
        else:
            rotate_bda = 0
            scale_bda = 1.0
            flip_dx = False
            flip_dy = False
        return rotate_bda, scale_bda, flip_dx, flip_dy

    def bda_matrix(self, rotate_angle, scale_ratio, flip_dx, flip_dy):
        rotate_angle = torch.tensor(rotate_angle / 180 * np.pi)
        rot_sin = torch.sin(rotate_angle)
        rot_cos = torch.cos(rotate_angle)
        rot_mat = torch.Tensor([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]])
        scale_mat = torch.Tensor(
            [[scale_ratio, 0, 0], [0, scale_ratio, 0], [0, 0, scale_ratio]]
        )
        flip_mat = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if flip_dx:  # flip along y axis
            flip_mat = flip_mat @ torch.Tensor([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if flip_dy:  # flip along x axis
            flip_mat = flip_mat @ torch.Tensor([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
        return flip_mat @ (scale_mat @ rot_mat)

    def __call__(self, results):
        rotate_bda, scale_bda, flip_dx, flip_dy = self.sample_bda_augmentation()
        bda_rot = self.bda_matrix(rotate_bda, scale_bda, flip_dx, flip_dy)

        imgs, sensor2egos, ego2globals, intrins = results["img_inputs"][:4]
        post_rots, post_trans = results["img_inputs"][4:]
        results["img_inputs"] = (
            imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda_rot
        )
        results["flip_dx"] = flip_dx
        results["flip_dy"] = flip_dy
        results["rotate_bda"] = rotate_bda
        results["scale_bda"] = scale_bda
        return results



class PointToMultiViewDepth:
    """Project lidar points into each (augmented) camera view -> sparse
    per-pixel metric depth maps (N_cams, H/downsample, W/downsample)."""

    def __init__(self, grid_config, downsample=1):
        self.downsample = downsample
        self.grid_config = grid_config

    def points2depthmap(self, points, height, width):
        height, width = height // self.downsample, width // self.downsample
        depth_map = torch.zeros((height, width), dtype=torch.float32)
        coor = torch.round(points[:, :2] / self.downsample)
        depth = points[:, 2]
        kept1 = (
            (coor[:, 0] >= 0) & (coor[:, 0] < width)
            & (coor[:, 1] >= 0) & (coor[:, 1] < height)
            & (depth < self.grid_config["depth"][1])
            & (depth >= self.grid_config["depth"][0])
        )
        coor, depth = coor[kept1], depth[kept1]
        ranks = coor[:, 0] + coor[:, 1] * width
        sort = (ranks + depth / 100.0).argsort()  # nearest-first per pixel
        coor, depth, ranks = coor[sort], depth[sort], ranks[sort]
        kept2 = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
        kept2[1:] = ranks[1:] != ranks[:-1]
        coor, depth = coor[kept2], depth[kept2]
        coor = coor.to(torch.long)
        depth_map[coor[:, 1], coor[:, 0]] = depth
        return depth_map

    def __call__(self, results):
        pts = results["points"]
        points_lidar = pts.tensor if hasattr(pts, "tensor") else pts  # LiDARPoints shim
        imgs, sensor2egos, ego2globals, intrins = results["img_inputs"][:4]
        post_rots, post_trans, bda = results["img_inputs"][4:]
        depth_map_list = []
        for cid in range(len(results["cam_names"])):
            cam_name = results["cam_names"][cid]
            # lidar and camera are not strictly synchronized, so the chain is
            # lidar -> lidar_ego -> global -> cam_ego -> cam (not lidar->ego->cam)
            lidar2lidarego = np.eye(4, dtype=np.float32)
            lidar2lidarego[:3, :3] = Quaternion(results["curr"]["lidar2ego_rotation"]).rotation_matrix
            lidar2lidarego[:3, 3] = results["curr"]["lidar2ego_translation"]
            lidar2lidarego = torch.from_numpy(lidar2lidarego)

            lidarego2global = np.eye(4, dtype=np.float32)
            lidarego2global[:3, :3] = Quaternion(results["curr"]["ego2global_rotation"]).rotation_matrix
            lidarego2global[:3, 3] = results["curr"]["ego2global_translation"]
            lidarego2global = torch.from_numpy(lidarego2global)

            cam2camego = np.eye(4, dtype=np.float32)
            cam2camego[:3, :3] = Quaternion(
                results["curr"]["cams"][cam_name]["sensor2ego_rotation"]
            ).rotation_matrix
            cam2camego[:3, 3] = results["curr"]["cams"][cam_name]["sensor2ego_translation"]
            cam2camego = torch.from_numpy(cam2camego)

            camego2global = np.eye(4, dtype=np.float32)
            camego2global[:3, :3] = Quaternion(
                results["curr"]["cams"][cam_name]["ego2global_rotation"]
            ).rotation_matrix
            camego2global[:3, 3] = results["curr"]["cams"][cam_name]["ego2global_translation"]
            camego2global = torch.from_numpy(camego2global)

            cam2img = np.eye(4, dtype=np.float32)
            cam2img = torch.from_numpy(cam2img)
            cam2img[:3, :3] = intrins[cid]

            lidar2cam = torch.inverse(camego2global.matmul(cam2camego)).matmul(
                lidarego2global.matmul(lidar2lidarego)
            )
            lidar2img = cam2img.matmul(lidar2cam)
            points_img = points_lidar[:, :3].matmul(lidar2img[:3, :3].T) + lidar2img[
                :3, 3
            ].unsqueeze(0)  # (N, 3): (u*d, v*d, d)
            points_img = torch.cat(
                [points_img[:, :2] / points_img[:, 2:3], points_img[:, 2:3]], 1
            )  # (N, 3): (u, v, d)

            # apply the image augmentation homography
            points_img = points_img.matmul(post_rots[cid].T) + post_trans[cid : cid + 1, :]
            depth_map_list.append(self.points2depthmap(points_img, imgs.shape[2], imgs.shape[3]))

        results["gt_depth"] = torch.stack(depth_map_list)
        return results


class LoadOccGTFromFile:
    """Load Occ3D-style labels.npz (semantics/mask_lidar/mask_camera,
    (200,200,16) uint8) and apply the BDA flips to keep GT aligned."""

    def __call__(self, results):
        occ_gt_path = os.path.join(results["occ_gt_path"], "labels.npz")
        occ_labels = np.load(occ_gt_path)
        semantics = torch.from_numpy(occ_labels["semantics"])
        mask_lidar = torch.from_numpy(occ_labels["mask_lidar"])
        mask_camera = torch.from_numpy(occ_labels["mask_camera"])

        if results.get("flip_dx", False):
            semantics = torch.flip(semantics, [0])
            mask_lidar = torch.flip(mask_lidar, [0])
            mask_camera = torch.flip(mask_camera, [0])
        if results.get("flip_dy", False):
            semantics = torch.flip(semantics, [1])
            mask_lidar = torch.flip(mask_lidar, [1])
            mask_camera = torch.flip(mask_camera, [1])

        results["voxel_semantics"] = semantics
        results["mask_lidar"] = mask_lidar
        results["mask_camera"] = mask_camera
        return results
