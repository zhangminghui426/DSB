"""dataset — NuScenesDataset (det/seg sample assembly + pipeline),
CBGSDataset (class-balanced resampling wrapper) and NuScenesOccDataset
(occupancy, BEVDet info protocol); build_dataset resolves config dicts.
"""

import logging
import os
import os.path as osp
import pickle

import numpy as np
import pyquaternion
from torch.utils.data import Dataset

from datasets.definitions import NAME_MAPPING, OBJECT_CLASSES
from datasets.geometry.boxes import LiDARInstance3DBoxes
from datasets.postprocess import evaluate as evaluate_results
from datasets.preprocess import build_pipeline, run_pipeline

logger = logging.getLogger(__name__)


class NuScenesDataset(Dataset):
    """Assembles per-sample info dicts (paths/poses/cameras/GT), runs the
    preprocessing pipeline, and evaluates model outputs via datasets.postprocess.

    Args:
        dataset_root / ann_file: data root and info pkl (from convert_nuscenes.py).
        pipeline: list of transform config dicts (see datasets.preprocess).
        object_classes / map_classes: detection / BEV-seg class names.
        modality: sensor modality dict (use_camera / use_lidar / ...).
        box_type_3d: "LiDAR" -> LiDARInstance3DBoxes (heads call it via metas).
        filter_empty_gt: drop train samples whose GT labels are all -1.
        test_mode: prepare_test_data path (no empty-GT retry).
        use_valid_flag: filter GT by valid_flag instead of num_lidar_pts > 0.
    """

    NameMapping = NAME_MAPPING
    CLASSES = OBJECT_CLASSES  # canonical order from datasets.definitions

    def __init__(
        self,
        dataset_root,
        ann_file,
        pipeline=None,
        object_classes=None,
        map_classes=None,
        load_interval=1,
        with_velocity=True,
        modality=None,
        box_type_3d="LiDAR",
        filter_empty_gt=True,
        test_mode=False,
        eval_version="detection_cvpr_2019",
        use_valid_flag=False,
    ):
        super().__init__()
        self.dataset_root = dataset_root
        self.ann_file = ann_file
        self.test_mode = test_mode
        self.modality = modality or dict(
            use_camera=False,
            use_lidar=True,
            use_radar=False,
            use_map=False,
            use_external=False,
        )
        self.filter_empty_gt = filter_empty_gt
        # resolve 'LiDAR' config string to the box class (heads call
        # metas['box_type_3d'](...) at test time); keep the mode string too
        if isinstance(box_type_3d, str):
            _box_types = {"LiDAR": LiDARInstance3DBoxes}
            if box_type_3d not in _box_types:
                raise ValueError(f"unsupported box_type_3d: {box_type_3d}")
            self.box_type_3d = _box_types[box_type_3d]
        else:
            self.box_type_3d = box_type_3d
        self.box_mode_3d = box_type_3d
        self.load_interval = load_interval
        self.use_valid_flag = use_valid_flag
        self.with_velocity = with_velocity
        self.eval_version = eval_version
        self.map_classes = map_classes

        if object_classes is not None:
            self.CLASSES = tuple(object_classes)
        self.cat2id = {name: i for i, name in enumerate(self.CLASSES)}

        self.data_infos = self.load_annotations(ann_file)
        self.pipeline = build_pipeline(pipeline) if pipeline is not None else None

        if not self.test_mode:
            self.flag = np.zeros(len(self), dtype=np.uint8)
        self.epoch = -1
        self._token_to_location = None  # lazy devkit fallback (sample -> location)

    def load_annotations(self, ann_file):
        """Load annotations from pkl file."""
        with open(ann_file, "rb") as f:
            data = pickle.load(f)
        data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))
        data_infos = data_infos[:: self.load_interval]
        self.metadata = data.get("metadata", {})
        self.version = self.metadata.get("version", "v1.0-mini")
        return data_infos

    def build_token_location_map(self):
        """Build token -> location mapping from NuScenes devkit."""
        if self._token_to_location is not None:
            return self._token_to_location

        try:
            from nuscenes.nuscenes import NuScenes

            nusc = NuScenes(
                version=self.version,
                dataroot=self.dataset_root,
                verbose=False,
            )
            token_to_loc = {}
            for scene in nusc.scene:
                # location lives on the LOG record, not the scene record
                log_rec = nusc.get("log", scene["log_token"])
                location = log_rec["location"]
                sample_token = scene["first_sample_token"]
                while sample_token != "":
                    token_to_loc[sample_token] = location
                    sample = nusc.get("sample", sample_token)
                    sample_token = sample.get("next", "")
            self._token_to_location = token_to_loc
        except Exception as e:
            logger.warning("failed to build token->location map via devkit: %s", e)
            self._token_to_location = {}

        return self._token_to_location

    def get_data_info(self, index):
        """Get data info for a given index."""
        info = self.data_infos[index]

        data = dict(
            token=info["token"],
            sample_idx=info["token"],
            lidar_path=info["lidar_path"],
            sweeps=info["sweeps"],
            timestamp=info["timestamp"],
            _dataset_root=self.dataset_root,
        )

        # Add location if available
        location = info.get("location", None)
        if location is None:
            # Try to get from devkit mapping
            token_loc_map = self.build_token_location_map()
            location = token_loc_map.get(info["token"], None)
        if location is not None:
            data["location"] = location

        # Add radar if available
        radar = info.get("radars", None)
        if radar is not None:
            data["radar"] = radar

        # Rebase lidar_path
        data["lidar_path"] = self.rebase_path(data["lidar_path"])

        # Rebase sweep paths
        for sweep in data["sweeps"]:
            sweep["data_path"] = self.rebase_path(sweep["data_path"])

        # ego to global transform
        ego2global = np.eye(4).astype(np.float64)
        ego2global[:3, :3] = pyquaternion.Quaternion(
            info["ego2global_rotation"]
        ).rotation_matrix
        ego2global[:3, 3] = info["ego2global_translation"]
        data["ego2global"] = ego2global

        # lidar to ego transform
        lidar2ego = np.eye(4).astype(np.float64)
        lidar2ego[:3, :3] = pyquaternion.Quaternion(
            info["lidar2ego_rotation"]
        ).rotation_matrix
        lidar2ego[:3, 3] = info["lidar2ego_translation"]
        data["lidar2ego"] = lidar2ego

        if self.modality.get("use_camera", False):
            data["image_paths"] = []
            data["lidar2camera"] = []
            data["lidar2image"] = []
            data["camera2ego"] = []
            data["camera_intrinsics"] = []
            data["camera2lidar"] = []

            for _, camera_info in info["cams"].items():
                cam_path = camera_info["data_path"]
                data["image_paths"].append(self.rebase_path(cam_path))

                # lidar to camera transform
                lidar2camera_r = np.linalg.inv(
                    camera_info["sensor2lidar_rotation"]
                )
                lidar2camera_t = (
                    camera_info["sensor2lidar_translation"] @ lidar2camera_r.T
                )
                lidar2camera_rt = np.eye(4).astype(np.float64)
                lidar2camera_rt[:3, :3] = lidar2camera_r.T
                lidar2camera_rt[3, :3] = -lidar2camera_t
                data["lidar2camera"].append(lidar2camera_rt.T)

                # camera intrinsics (raw pinhole K; together with the stored
                # lidar2camera — whose frame is (right, down, forward) in the
                # lidar (x=right, y=forward, z=up) convention — this yields the
                # standard z-depth projection u = cx + fx*xc/zc, validated
                # against side-camera FOV separation and official visualize.py)
                camera_intrinsics = np.eye(4).astype(np.float64)
                camera_intrinsics[:3, :3] = camera_info["cam_intrinsic"]
                data["camera_intrinsics"].append(camera_intrinsics)

                # lidar to image transform
                lidar2image = camera_intrinsics @ lidar2camera_rt.T
                data["lidar2image"].append(lidar2image)

                # camera to ego transform
                camera2ego = np.eye(4).astype(np.float64)
                camera2ego[:3, :3] = pyquaternion.Quaternion(
                    camera_info["sensor2ego_rotation"]
                ).rotation_matrix
                camera2ego[:3, 3] = camera_info["sensor2ego_translation"]
                data["camera2ego"].append(camera2ego)

                # camera to lidar transform
                camera2lidar = np.eye(4).astype(np.float64)
                camera2lidar[:3, :3] = camera_info["sensor2lidar_rotation"]
                camera2lidar[:3, 3] = camera_info["sensor2lidar_translation"]
                data["camera2lidar"].append(camera2lidar)

        # GT keys and box-type metadata go directly to top level
        data.update(self.get_ann_info(index))
        data["box_type_3d"] = self.box_type_3d
        data["box_mode_3d"] = self.box_mode_3d
        return data

    def get_ann_info(self, index):
        """Get annotation info for a given index."""
        info = self.data_infos[index]

        if self.use_valid_flag:
            mask = info["valid_flag"]
        else:
            mask = info["num_lidar_pts"] > 0

        gt_bboxes_3d = info["gt_boxes"][mask]
        gt_names_3d = info["gt_names"][mask]
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d, dtype=np.int64)

        if self.with_velocity:
            gt_velocity = info["gt_velocity"][mask]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate(
                [gt_bboxes_3d, gt_velocity], axis=-1
            )

        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0),
        )

        return dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
        )

    def get_cat_ids(self, idx):
        """Get category ids for a given index."""
        info = self.data_infos[idx]
        if self.use_valid_flag:
            mask = info["valid_flag"]
            gt_names = set(info["gt_names"][mask])
        else:
            gt_names = set(info["gt_names"])

        cat_ids = []
        for name in gt_names:
            if name in self.CLASSES:
                cat_ids.append(self.cat2id[name])
        return cat_ids

    def prepare_train_data(self, index):
        """get_data_info -> pipeline; None if the sample has no valid GT."""
        data = run_pipeline(self.pipeline, self.get_data_info(index))
        if data is None:
            return None
        if self.filter_empty_gt:
            gt_labels = data.get("gt_labels_3d", None)
            if gt_labels is not None and not (gt_labels != -1).any():
                return None
        return data

    def prepare_test_data(self, index):
        """get_data_info -> pipeline (no empty-GT filtering)."""
        return run_pipeline(self.pipeline, self.get_data_info(index))

    def rand_another(self, idx):
        """Randomly get another item with the same flag."""
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self.rand_another(idx)
                continue
            return data

    def __len__(self):
        return len(self.data_infos)

    def set_epoch(self, epoch):
        """Set epoch for transforms that need it."""
        self.epoch = epoch
        if self.pipeline is not None:
            for transform in self.pipeline:
                if hasattr(transform, "set_epoch"):
                    transform.set_epoch(epoch)

    def rebase_path(self, p):
        """Rebase a path to be relative to dataset_root."""
        if os.path.isabs(p):
            if os.path.exists(p):
                return p
            for marker in ("samples/", "sweeps/"):
                idx = p.find(marker)
                if idx >= 0:
                    rebased = os.path.join(self.dataset_root, p[idx:])
                    if os.path.exists(rebased):
                        return rebased
            return os.path.join(self.dataset_root, os.path.basename(p))
        else:
            full = os.path.join(self.dataset_root, p)
            if os.path.exists(full):
                return full
            return p

    def evaluate(self, results, metric="bbox", jsonfile_prefix=None, **kwargs):
        """Evaluate model outputs — see datasets.postprocess."""
        return evaluate_results(
            self, results, metric=metric, jsonfile_prefix=jsonfile_prefix, **kwargs
        )


class CBGSDataset(Dataset):
    """Class-balanced grouping & sampling: resample frame indices so every
    class contributes equally per epoch (oversample rare-class frames).

    Attribute access falls through to the wrapped dataset.
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self.CLASSES = dataset.CLASSES
        self.cat2id = {name: i for i, name in enumerate(self.CLASSES)}
        self.sample_indices = self.get_sample_indices()
        if hasattr(dataset, "flag"):
            self.flag = np.array(
                [dataset.flag[i] for i in self.sample_indices], dtype=np.uint8
            )

    def __getattr__(self, name):
        if name == "dataset":
            raise AttributeError(name)
        return getattr(self.dataset, name)

    def set_epoch(self, epoch):
        self.dataset.set_epoch(epoch)

    def get_sample_indices(self):
        """Per-class frame index lists -> balanced resample (with replacement)."""
        class_sample_idxs = {cat_id: [] for cat_id in self.cat2id.values()}
        for idx in range(len(self.dataset)):
            for cat_id in self.dataset.get_cat_ids(idx):
                class_sample_idxs[cat_id].append(idx)

        duplicated_samples = sum(len(v) for _, v in class_sample_idxs.items())
        class_distribution = {
            k: len(v) / max(duplicated_samples, 1)
            for k, v in class_sample_idxs.items()
        }

        sample_indices = []
        frac = 1.0 / len(self.CLASSES)
        ratios = [frac / max(v, 1e-10) for v in class_distribution.values()]
        for cls_inds, ratio in zip(list(class_sample_idxs.values()), ratios):
            if len(cls_inds) > 0:
                sample_indices += np.random.choice(
                    cls_inds, int(len(cls_inds) * ratio)
                ).tolist()
        return sample_indices

    def __getitem__(self, idx):
        return self.dataset[self.sample_indices[idx]]

    def __len__(self):
        return len(self.sample_indices)


def build_dataset(cfg):
    """Build a dataset from a config dict (NuScenesDataset | CBGSDataset |
    NuScenesOccDataset)."""
    cfg = dict(cfg)
    ds_type = cfg.pop("type")
    if ds_type == "CBGSDataset":
        return CBGSDataset(build_dataset(cfg.pop("dataset")))
    if ds_type == "NuScenesDataset":
        return NuScenesDataset(**cfg)
    if ds_type == "NuScenesOccDataset":
        return NuScenesOccDataset(**cfg)
    raise ValueError(f"Unknown dataset type: {ds_type}")


class NuScenesOccDataset(Dataset):
    """Args:
        ann_file: infos pkl path.
        pipeline: list of transform config dicts (see datasets.preprocess).
        is_train: train-mode augmentation in the transforms.
        load_interval: subsample stride over the timestamp-sorted infos.
        multi_adj_frame_id_cfg: (begin, end, step) selecting adjacent frames
            (4D temporal variants); None = single-frame.
        stereo: load one extra adjacent frame for stereo depth (4D+stereo).
    """

    def __init__(self, ann_file, pipeline=None, is_train=False, load_interval=1,
                 multi_adj_frame_id_cfg=None, stereo=False):
        self.ann_file = ann_file
        self.is_train = is_train
        self.load_interval = load_interval
        self.multi_adj_frame_id_cfg = multi_adj_frame_id_cfg
        self.stereo = stereo
        self.data_list = self.load_data_list()
        self.pipeline = build_pipeline(pipeline) if pipeline is not None else None

    def load_data_list(self):
        with open(self.ann_file, "rb") as f:
            data = pickle.load(f)
        infos = sorted(data["infos"], key=lambda e: e["timestamp"])
        infos = infos[:: self.load_interval]
        self.metadata = data.get("metadata", dict())
        self.version = self.metadata.get("version", None)
        return [self.parse_data_info(info) for info in infos]

    def parse_data_info(self, info):
        """Make repo-relative paths absolute (lidar / cams / occ GT)."""
        info["lidar_path"] = osp.abspath(osp.normpath(info["lidar_path"]))
        for cam_info in info["cams"].values():
            cam_info["data_path"] = osp.abspath(osp.normpath(cam_info["data_path"]))
        if "occ_path" in info:
            info["occ_path"] = osp.abspath(osp.normpath(info["occ_path"]))
        return info

    def get_data_info(self, index):
        """Assemble the per-sample dict consumed by the pipeline."""
        info = self.data_list[index]
        input_dict = dict(
            sample_idx=info["token"],
            pts_filename=info["lidar_path"],
            lidar_path=info["lidar_path"],
            sweeps=info.get("sweeps", []),
            timestamp=info["timestamp"] / 1e6,
            token=info["token"],
            scene_name=info.get("scene_name"),
            scene_token=info.get("scene_token"),
            occ_gt_path=info.get("occ_path"),
            lidar2ego_translation=info.get("lidar2ego_translation"),
            lidar2ego_rotation=info.get("lidar2ego_rotation"),
            ego2global_translation=info.get("ego2global_translation"),
            ego2global_rotation=info.get("ego2global_rotation"),
            curr=info,
        )
        if "ann_infos" in info:
            input_dict["ann_infos"] = info["ann_infos"]
        if self.multi_adj_frame_id_cfg is not None:
            input_dict["adjacent"] = self.get_adj_info(info, index)
        return input_dict

    def get_adj_info(self, info, index):
        """Adjacent (history) frame infos for 4D variants; falls back to the
        current frame when the neighbor crosses a scene boundary."""
        info_adj_list = []
        adj_id_list = list(range(*self.multi_adj_frame_id_cfg))
        if self.stereo:
            assert self.multi_adj_frame_id_cfg[0] == 1
            assert self.multi_adj_frame_id_cfg[2] == 1
            # stereo also needs the frame before the adjacent one
            adj_id_list.append(self.multi_adj_frame_id_cfg[1])
        for select_id in adj_id_list:
            select_id = max(index - select_id, 0)
            if self.data_list[select_id]["scene_token"] != info["scene_token"]:
                info_adj_list.append(info)
            else:
                info_adj_list.append(self.data_list[select_id])
        return info_adj_list

    def __getitem__(self, index):
        return run_pipeline(self.pipeline, self.get_data_info(index))

    def __len__(self):
        return len(self.data_list)

    def evaluate(self, results, metric=None, jsonfile_prefix=None,
                 save_dir=None, verbose=True, **kwargs):
        """Evaluate ordered predictions — see datasets.postprocess.

        metric/jsonfile_prefix accepted for call-site uniformity with the
        det/seg datasets (ignored here)."""
        return evaluate_results(self, results, save_dir=save_dir,
                              verbose=verbose)


