"""Preprocessing — augmentation: GT-object paste, image/point aug, grid mask."""

import copy
import os
import pickle

import numpy as np
import torch
from PIL import Image

from datasets.geometry.box_np_ops import (
    box_collision_test,
    center_to_corner_box2d,
    points_in_rbbox,
)
from datasets.geometry.points import LiDARPoints
from datasets.preprocess.loading import reduce_LiDAR_beams

class BatchSampler:
    """Batch sampler for GT database."""

    def __init__(self, sampled_list, name=None, shuffle=True):
        self._sampled_list = sampled_list
        self._indices = np.arange(len(sampled_list))
        if shuffle:
            np.random.shuffle(self._indices)
        self._idx = 0
        self._example_num = len(sampled_list)
        self._shuffle = shuffle

    def take_indices(self, num):
        if self._idx + num >= self._example_num:
            ret = self._indices[self._idx :].copy()
            self.reset()
            if len(ret) < num:
                ret2 = self._indices[: num - len(ret)]
                self._idx = num - len(ret)
                ret = np.concatenate([ret, ret2])
        else:
            ret = self._indices[self._idx : self._idx + num]
            self._idx += num
        return ret

    def reset(self):
        if self._shuffle:
            np.random.shuffle(self._indices)
        self._idx = 0

    def sample(self, num):
        indices = self.take_indices(num)
        return [self._sampled_list[i] for i in indices]


class DataBaseSampler:
    """Ground truth database sampler for ObjectPaste."""

    def __init__(
        self,
        info_path,
        dataset_root,
        rate,
        prepare,
        sample_groups,
        classes=None,
        points_loader=None,
    ):
        self.dataset_root = dataset_root
        self.rate = rate
        self.prepare = prepare
        self.classes = classes or []
        self.cat2label = {name: i for i, name in enumerate(self.classes)}
        self.label2cat = {i: name for i, name in enumerate(self.classes)}

        # Build points loader
        if points_loader is None:
            points_loader = dict(
                type="LoadPointsFromFile",
                coord_type="LIDAR",
                load_dim=4,
                use_dim=[0, 1, 2, 3],
            )
        self.points_loader = build_points_loader(points_loader)

        # Load db infos
        with open(info_path, "rb") as f:
            db_infos = pickle.load(f)

        # Apply preparation filters
        for prep_func, val in prepare.items():
            db_infos = getattr(self, prep_func)(db_infos, val)

        self.db_infos = db_infos

        # Build sample groups
        self.sample_groups = []
        for name, num in sample_groups.items():
            self.sample_groups.append({name: int(num)})

        self.sample_classes = []
        self.sample_max_nums = []
        for group_info in self.sample_groups:
            self.sample_classes += list(group_info.keys())
            self.sample_max_nums += list(group_info.values())

        self.sampler_dict = {}
        for k, v in self.db_infos.items():
            self.sampler_dict[k] = BatchSampler(v, k, shuffle=True)

    @staticmethod
    def filter_by_difficulty(db_infos, removed_difficulty):
        new_db_infos = {}
        for key, dinfos in db_infos.items():
            new_db_infos[key] = [
                info
                for info in dinfos
                if info.get("difficulty", -1) not in removed_difficulty
            ]
        return new_db_infos

    @staticmethod
    def filter_by_min_points(db_infos, min_gt_points_dict):
        for name, min_num in min_gt_points_dict.items():
            min_num = int(min_num)
            if min_num > 0:
                filtered_infos = []
                for info in db_infos.get(name, []):
                    if info["num_points_in_gt"] >= min_num:
                        filtered_infos.append(info)
                db_infos[name] = filtered_infos
        return db_infos

    def sample_all(self, gt_bboxes, gt_labels, img=None):
        sampled_num_dict = {}
        sample_num_per_class = []
        for class_name, max_sample_num in zip(
            self.sample_classes, self.sample_max_nums
        ):
            class_label = self.cat2label[class_name]
            sampled_num = int(
                max_sample_num - np.sum([n == class_label for n in gt_labels])
            )
            sampled_num = np.round(self.rate * sampled_num).astype(np.int64)
            sampled_num_dict[class_name] = sampled_num
            sample_num_per_class.append(sampled_num)

        sampled = []
        sampled_gt_bboxes = []
        avoid_coll_boxes = gt_bboxes

        for class_name, sampled_num in zip(
            self.sample_classes, sample_num_per_class
        ):
            if sampled_num > 0:
                if class_name not in self.sampler_dict:
                    continue
                sampled_cls = self.sample_class_v2(
                    class_name, sampled_num, avoid_coll_boxes
                )
                sampled += sampled_cls
                if len(sampled_cls) > 0:
                    if len(sampled_cls) == 1:
                        sampled_gt_box = sampled_cls[0]["box3d_lidar"][
                            np.newaxis, ...
                        ]
                    else:
                        sampled_gt_box = np.stack(
                            [s["box3d_lidar"] for s in sampled_cls], axis=0
                        )
                    sampled_gt_bboxes += [sampled_gt_box]
                    avoid_coll_boxes = np.concatenate(
                        [avoid_coll_boxes, sampled_gt_box], axis=0
                    )

        ret = None
        if len(sampled) > 0:
            sampled_gt_bboxes = np.concatenate(sampled_gt_bboxes, axis=0)
            s_points_list = []
            for info in sampled:
                file_path = (
                    os.path.join(self.dataset_root, info["path"])
                    if self.dataset_root
                    else info["path"]
                )
                s_points = self.points_loader(file_path)
                s_points.translate(info["box3d_lidar"][:3])
                s_points_list.append(s_points)

            gt_labels_out = np.array(
                [self.cat2label[s["name"]] for s in sampled], dtype=np.int64
            )
            ret = {
                "gt_labels_3d": gt_labels_out,
                "gt_bboxes_3d": sampled_gt_bboxes,
                "points": LiDARPoints.cat(s_points_list),
                "group_ids": np.arange(
                    gt_bboxes.shape[0], gt_bboxes.shape[0] + len(sampled)
                ),
            }

        return ret

    def sample_class_v2(self, name, num, gt_bboxes):
        sampled = self.sampler_dict[name].sample(num)
        sampled = copy.deepcopy(sampled)
        num_gt = gt_bboxes.shape[0]
        num_sampled = len(sampled)
        gt_bboxes_bv = center_to_corner_box2d(
            gt_bboxes[:, 0:2], gt_bboxes[:, 3:5], gt_bboxes[:, 6]
        )

        sp_boxes = np.stack([i["box3d_lidar"] for i in sampled], axis=0)
        boxes = np.concatenate([gt_bboxes, sp_boxes], axis=0).copy()

        sp_boxes_new = boxes[gt_bboxes.shape[0] :]
        sp_boxes_bv = center_to_corner_box2d(
            sp_boxes_new[:, 0:2], sp_boxes_new[:, 3:5], sp_boxes_new[:, 6]
        )

        total_bv = np.concatenate([gt_bboxes_bv, sp_boxes_bv], axis=0)
        coll_mat = box_collision_test(total_bv, total_bv)
        diag = np.arange(total_bv.shape[0])
        coll_mat[diag, diag] = False

        valid_samples = []
        for i in range(num_gt, num_gt + num_sampled):
            if coll_mat[i].any():
                coll_mat[i] = False
                coll_mat[:, i] = False
            else:
                valid_samples.append(sampled[i - num_gt])
        return valid_samples


def build_points_loader(cfg):
    """Build a simple points loader function from config dict."""
    load_dim = cfg.get("load_dim", 5)
    use_dim = cfg.get("use_dim", [0, 1, 2, 3, 4])
    if isinstance(use_dim, int):
        use_dim = list(range(use_dim))
    reduce_beams = cfg.get("reduce_beams", None)

    def loader(file_path):
        if file_path.endswith(".npy"):
            points = np.load(file_path)
        else:
            points = np.fromfile(file_path, dtype=np.float32)
        points = points.reshape(-1, load_dim)
        if reduce_beams is not None and reduce_beams < 32:
            points = reduce_LiDAR_beams(points, reduce_beams)
        points = points[:, use_dim]
        return LiDARPoints(points, points_dim=points.shape[-1])

    return loader


# ===========================================================================
# Augmentation transforms
# ===========================================================================


class ObjectPaste:
    """Sample GT objects to the data using DataBaseSampler."""

    def __init__(self, db_sampler, sample_2d=False, stop_epoch=None):
        self.sampler_cfg = db_sampler
        self.sample_2d = sample_2d
        self.stop_epoch = stop_epoch
        self.epoch = -1
        # Build db_sampler lazily (needs dataset_root from data dict)
        self._db_sampler = None
        self._built = False

    def set_epoch(self, epoch):
        self.epoch = epoch

    def ensure_built(self, dataset_root):
        if not self._built:
            cfg = copy.deepcopy(self.sampler_cfg)
            cfg.pop("type", None)
            # Resolve info_path relative to dataset_root
            info_path = cfg.get("info_path", "")
            if not os.path.isabs(info_path):
                info_path = os.path.join(dataset_root, info_path)
            cfg["info_path"] = info_path
            cfg["dataset_root"] = dataset_root
            self._db_sampler = DataBaseSampler(**cfg)
            self._built = True

    @staticmethod
    def remove_points_in_boxes(points, boxes):
        masks = points_in_rbbox(points.coord.numpy(), boxes)
        points = points[np.logical_not(masks.any(-1))]
        return points

    def __call__(self, data):
        if self.stop_epoch is not None and self.epoch >= self.stop_epoch:
            return data

        dataset_root = data.get("_dataset_root", "data/nuscenes/")
        self.ensure_built(dataset_root)

        gt_bboxes_3d = data["gt_bboxes_3d"]
        gt_labels_3d = data["gt_labels_3d"]
        points = data["points"]

        sampled_dict = self._db_sampler.sample_all(
            gt_bboxes_3d.tensor.numpy(), gt_labels_3d, img=None
        )

        if sampled_dict is not None:
            sampled_gt_bboxes_3d = sampled_dict["gt_bboxes_3d"]
            sampled_points = sampled_dict["points"]
            sampled_gt_labels = sampled_dict["gt_labels_3d"]

            gt_labels_3d = np.concatenate(
                [gt_labels_3d, sampled_gt_labels], axis=0
            )
            gt_bboxes_3d = gt_bboxes_3d.new_box(
                np.concatenate(
                    [gt_bboxes_3d.tensor.numpy(), sampled_gt_bboxes_3d]
                )
            )

            points = self.remove_points_in_boxes(points, sampled_gt_bboxes_3d)
            points = LiDARPoints.cat([sampled_points, points])

        data["gt_bboxes_3d"] = gt_bboxes_3d
        data["gt_labels_3d"] = gt_labels_3d.astype(np.int64)
        data["points"] = points
        return data


class ImageAug3D:
    """3D image augmentation (resize, crop, flip, rotate)."""

    def __init__(
        self, final_dim, resize_lim, bot_pct_lim, rot_lim, rand_flip, is_train
    ):
        self.final_dim = final_dim
        self.resize_lim = resize_lim
        self.bot_pct_lim = bot_pct_lim
        self.rand_flip = rand_flip
        self.rot_lim = rot_lim
        self.is_train = is_train

    def sample_augmentation(self, results):
        W, H = results["ori_shape"]
        fH, fW = self.final_dim
        if self.is_train:
            resize = np.random.uniform(*self.resize_lim)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.bot_pct_lim)) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.rand_flip and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.rot_lim)
        else:
            resize = np.mean(self.resize_lim)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.bot_pct_lim)) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def img_transform(
        self, img, rotation, translation, resize, resize_dims, crop, flip, rotate
    ):
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # post-homography transformation
        rotation *= resize
        translation -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            rotation = A.matmul(rotation)
            translation = A.matmul(translation) + b
        theta = rotate / 180 * np.pi
        A = torch.Tensor(
            [
                [np.cos(theta), np.sin(theta)],
                [-np.sin(theta), np.cos(theta)],
            ]
        )
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        rotation = A.matmul(rotation)
        translation = A.matmul(translation) + b

        return img, rotation, translation

    def __call__(self, data):
        imgs = data["img"]
        new_imgs = []
        transforms = []
        for img in imgs:
            resize, resize_dims, crop, flip, rotate = self.sample_augmentation(
                data
            )
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)
            new_img, rotation, translation = self.img_transform(
                img, post_rot, post_tran,
                resize=resize, resize_dims=resize_dims,
                crop=crop, flip=flip, rotate=rotate,
            )
            transform = torch.eye(4)
            transform[:2, :2] = rotation
            transform[:2, 3] = translation
            new_imgs.append(new_img)
            transforms.append(transform.numpy())
        data["img"] = new_imgs
        data["img_aug_matrix"] = transforms
        return data


class GlobalRotScaleTrans:
    """Global rotation, scaling, and translation augmentation."""

    def __init__(self, resize_lim, rot_lim, trans_lim, is_train):
        self.resize_lim = resize_lim
        self.rot_lim = rot_lim
        self.trans_lim = trans_lim
        self.is_train = is_train

    def __call__(self, data):
        transform = np.eye(4).astype(np.float32)

        if self.is_train:
            scale = np.random.uniform(*self.resize_lim)
            theta = np.random.uniform(*self.rot_lim)
            translation = np.array(
                [np.random.normal(0, self.trans_lim) for _ in range(3)]
            )
            rotation = np.eye(3)

            if "points" in data:
                data["points"].rotate(-theta)
                data["points"].translate(translation)
                data["points"].scale(scale)

            gt_boxes = data["gt_bboxes_3d"]
            rotation = rotation @ gt_boxes.rotate(theta).numpy()
            gt_boxes.translate(translation)
            gt_boxes.scale(scale)
            data["gt_bboxes_3d"] = gt_boxes

            transform[:3, :3] = rotation.T * scale
            transform[:3, 3] = translation * scale

        data["lidar_aug_matrix"] = transform
        return data


class RandomFlip3D:
    """Random horizontal and vertical flip in 3D."""

    def __call__(self, data):
        flip_horizontal = np.random.choice([0, 1])
        flip_vertical = np.random.choice([0, 1])

        rotation = np.eye(3)
        if flip_horizontal:
            rotation = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]) @ rotation
            if "points" in data:
                data["points"].flip("horizontal")
            if "gt_bboxes_3d" in data:
                data["gt_bboxes_3d"].flip("horizontal")
            if "gt_masks_bev" in data:
                data["gt_masks_bev"] = data["gt_masks_bev"][:, :, ::-1].copy()

        if flip_vertical:
            rotation = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]) @ rotation
            if "points" in data:
                data["points"].flip("vertical")
            if "gt_bboxes_3d" in data:
                data["gt_bboxes_3d"].flip("vertical")
            if "gt_masks_bev" in data:
                data["gt_masks_bev"] = data["gt_masks_bev"][:, ::-1, :].copy()

        data["lidar_aug_matrix"][:3, :] = (
            rotation @ data["lidar_aug_matrix"][:3, :]
        )
        return data


class GridMask:
    """Grid mask augmentation for images."""

    def __init__(
        self,
        use_h,
        use_w,
        max_epoch,
        rotate=1,
        offset=False,
        ratio=0.5,
        mode=0,
        prob=1.0,
        fixed_prob=False,
    ):
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.prob = prob
        self.epoch = None
        self.max_epoch = max_epoch
        self.fixed_prob = fixed_prob

    def set_epoch(self, epoch):
        self.epoch = epoch
        if not self.fixed_prob:
            self.set_prob(self.epoch, self.max_epoch)

    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * self.epoch / self.max_epoch

    def __call__(self, results):
        if np.random.rand() > self.prob:
            return results
        imgs = results["img"]
        h = imgs[0].shape[0]
        w = imgs[0].shape[1]
        self.d1 = 2
        self.d2 = min(h, w)
        hh = int(1.5 * h)
        ww = int(1.5 * w)
        d = np.random.randint(self.d1, self.d2)
        if self.ratio == 1:
            self.l = np.random.randint(1, d)
        else:
            self.l = min(max(int(d * self.ratio + 0.5), 1), d - 1)
        mask = np.ones((hh, ww), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        if self.use_h:
            for i in range(hh // d):
                s = d * i + st_h
                t = min(s + self.l, hh)
                mask[s:t, :] *= 0
        if self.use_w:
            for i in range(ww // d):
                s = d * i + st_w
                t = min(s + self.l, ww)
                mask[:, s:t] *= 0

        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[
            (hh - h) // 2 : (hh - h) // 2 + h,
            (ww - w) // 2 : (ww - w) // 2 + w,
        ]

        mask = mask.astype(np.float32)
        mask = mask[:, :, None]
        if self.mode == 1:
            mask = 1 - mask

        if self.offset:
            offset = torch.from_numpy(
                2 * (np.random.rand(h, w) - 0.5)
            ).float()
            offset = (1 - mask) * offset
            imgs = [x * mask + offset for x in imgs]
        else:
            imgs = [x * mask for x in imgs]

        results["img"] = imgs
        return results


class PointShuffle:
    """Shuffle point order."""

    def __call__(self, data):
        data["points"].shuffle()
        return data
