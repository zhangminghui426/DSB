"""Postprocessing — model outputs to evaluation metrics.

Detection: results -> nuScenes submission JSON -> devkit DetectionEval
(mAP/NDS under 'object/*'). Segmentation: BEV IoU @ 0.35..0.65 ('map/*').
Every entry point takes the NuScenesDataset instance as first argument.
"""

import json
import os
import os.path as osp
import tempfile

import numpy as np
import pyquaternion
import torch

from datasets.definitions import (DEFAULT_ATTR, ERR_NAME_MAPPING,
                                NUM_CLASSES, OCC_CLASS_NAMES)


def format_results(dataset, results, jsonfile_prefix=None):
    """Format results to JSON."""
    if jsonfile_prefix is None:
        tmp_dir = tempfile.TemporaryDirectory()
        jsonfile_prefix = os.path.join(tmp_dir.name, "results")
    else:
        tmp_dir = None

    result_files = format_bbox(dataset, results, jsonfile_prefix)
    return result_files, tmp_dir

def format_bbox(dataset, results, jsonfile_prefix=None):
    """Convert results to a nuScenes submission JSON (results_nusc.json)."""
    nusc_annos = {}
    mapped_class_names = dataset.CLASSES

    for sample_id, det in enumerate(results):
        annos = []
        boxes = output_to_nusc_box(det)
        sample_token = dataset.data_infos[sample_id]["token"]
        boxes = lidar_nusc_box_to_global(
            dataset, dataset.data_infos[sample_id], boxes, mapped_class_names
        )
        for i, box in enumerate(boxes):
            name = mapped_class_names[box.label]
            if np.sqrt(box.velocity[0] ** 2 + box.velocity[1] ** 2) > 0.2:
                if name in [
                    "car", "construction_vehicle", "bus", "truck", "trailer"
                ]:
                    attr = "vehicle.moving"
                elif name in ["bicycle", "motorcycle"]:
                    attr = "cycle.with_rider"
                else:
                    attr = DEFAULT_ATTR[name]
            else:
                if name in ["pedestrian"]:
                    attr = "pedestrian.standing"
                elif name in ["bus"]:
                    attr = "vehicle.stopped"
                else:
                    attr = DEFAULT_ATTR[name]

            nusc_anno = dict(
                sample_token=sample_token,
                translation=box.center.tolist(),
                size=box.wlh.tolist(),
                rotation=box.orientation.elements.tolist(),
                velocity=box.velocity[:2].tolist(),
                detection_name=name,
                detection_score=box.score,
                attribute_name=attr,
            )
            annos.append(nusc_anno)
        nusc_annos[sample_token] = annos

    nusc_submissions = {
        "meta": dataset.modality,
        "results": nusc_annos,
    }

    os.makedirs(jsonfile_prefix, exist_ok=True)
    res_path = os.path.join(jsonfile_prefix, "results_nusc.json")
    with open(res_path, "w") as f:
        json.dump(nusc_submissions, f)
    return res_path

def output_to_nusc_box(detection):
    """Convert detection output to NuScenesBox."""
    from nuscenes.utils.data_classes import Box as NuScenesBox

    box3d = detection["boxes_3d"]
    scores = detection["scores_3d"].numpy()
    labels = detection["labels_3d"].numpy()

    box_gravity_center = box3d.gravity_center.numpy()
    box_dims = box3d.dims.numpy()
    box_yaw = box3d.yaw.numpy()
    box_yaw = -box_yaw - np.pi / 2

    box_list = []
    for i in range(len(box3d)):
        quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
        velocity = (*box3d.tensor[i, 7:9], 0.0)
        box = NuScenesBox(
            box_gravity_center[i],
            box_dims[i],
            quat,
            label=labels[i],
            score=scores[i],
            velocity=velocity,
        )
        box_list.append(box)
    return box_list

def lidar_nusc_box_to_global(dataset, info, boxes, classes):
    """Convert boxes from lidar to global coordinates."""
    try:
        from nuscenes.eval.detection.config import config_factory
        eval_configs = config_factory(dataset.eval_version)
    except Exception:
        eval_configs = None

    box_list = []
    for box in boxes:
        box.rotate(pyquaternion.Quaternion(info["lidar2ego_rotation"]))
        box.translate(np.array(info["lidar2ego_translation"]))

        if eval_configs is not None:
            cls_range_map = eval_configs.class_range
            radius = np.linalg.norm(box.center[:2], 2)
            det_range = cls_range_map.get(classes[box.label], 100)
            if radius > det_range:
                continue

        box.rotate(pyquaternion.Quaternion(info["ego2global_rotation"]))
        box.translate(np.array(info["ego2global_translation"]))
        box_list.append(box)
    return box_list

def evaluate_single(dataset, result_path, metric="bbox"):
    """Evaluate using nuscenes detection eval."""
    from nuscenes import NuScenes
    from nuscenes.eval.detection.evaluate import DetectionEval
    from nuscenes.eval.detection.config import config_factory

    output_dir = os.path.dirname(result_path)
    nusc = NuScenes(
        version=dataset.version, dataroot=dataset.dataset_root, verbose=False
    )
    eval_set_map = {
        "v1.0-mini": "mini_val",
        "v1.0-trainval": "val",
    }
    eval_detection_configs = config_factory(dataset.eval_version)
    nusc_eval = DetectionEval(
        nusc,
        config=eval_detection_configs,
        result_path=result_path,
        eval_set=eval_set_map.get(dataset.version, "val"),
        output_dir=output_dir,
        verbose=False,
    )
    nusc_eval.main(render_curves=False)

    metrics_path = os.path.join(output_dir, "metrics_summary.json")
    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    detail = {}
    for name in dataset.CLASSES:
        for k, v in metrics["label_aps"][name].items():
            val = float(f"{v:.4f}")
            detail[f"object/{name}_ap_dist_{k}"] = val
        for k, v in metrics["label_tp_errors"][name].items():
            val = float(f"{v:.4f}")
            detail[f"object/{name}_{k}"] = val
    for k, v in metrics["tp_errors"].items():
        val = float(f"{v:.4f}")
        detail[f"object/{ERR_NAME_MAPPING[k]}"] = val

    detail["object/nds"] = metrics["nd_score"]
    detail["object/map"] = metrics["mean_ap"]
    return detail

def evaluate_map(dataset, results):
    """Evaluate BEV segmentation mAP."""
    thresholds = torch.tensor([0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65])
    num_classes = len(dataset.map_classes)
    num_thresholds = len(thresholds)

    tp = torch.zeros(num_classes, num_thresholds)
    fp = torch.zeros(num_classes, num_thresholds)
    fn = torch.zeros(num_classes, num_thresholds)

    for result in results:
        pred = result["masks_bev"]
        label = result["gt_masks_bev"]

        pred = pred.detach().reshape(num_classes, -1)
        label = label.detach().bool().reshape(num_classes, -1)

        pred = pred[:, :, None] >= thresholds
        label = label[:, :, None]

        tp += (pred & label).sum(dim=1)
        fp += (pred & ~label).sum(dim=1)
        fn += (~pred & label).sum(dim=1)

    ious = tp / (tp + fp + fn + 1e-7)

    metrics = {}
    for index, name in enumerate(dataset.map_classes):
        metrics[f"map/{name}/iou@max"] = ious[index].max().item()
        for threshold, iou in zip(thresholds, ious[index]):
            metrics[f"map/{name}/iou@{threshold.item():.2f}"] = iou.item()
    metrics["map/mean/iou@max"] = ious.max(dim=1).values.mean().item()
    return metrics

def evaluate(dataset, results, metric="bbox", jsonfile_prefix=None,
             save_dir=None, use_image_mask=True, verbose=True, **kwargs):
    """Dispatch on result keys: pred_occ -> occ mIoU, masks_bev -> BEV IoU,
    boxes_3d -> devkit det eval."""
    if results and "pred_occ" in results[0]:
        return evaluate_occ(dataset, results, save_dir=save_dir,
                            use_image_mask=use_image_mask, verbose=verbose)
    metrics = {}

    if results and "masks_bev" in results[0]:
        metrics.update(evaluate_map(dataset, results))

    if results and "boxes_3d" in results[0]:
        result_files, tmp_dir = format_results(dataset, results, jsonfile_prefix)
        if isinstance(result_files, str):
            metrics.update(evaluate_single(dataset, result_files))
        if tmp_dir is not None:
            tmp_dir.cleanup()

    return metrics


# ---- occupancy mIoU ----

class MetricMIoU:
    def __init__(self, num_classes=NUM_CLASSES, use_lidar_mask=False, use_image_mask=True):
        self.class_names = list(OCC_CLASS_NAMES)
        self.num_classes = num_classes
        self.use_lidar_mask = use_lidar_mask
        self.use_image_mask = use_image_mask
        self.hist = np.zeros((num_classes, num_classes))
        self.cnt = 0

    @staticmethod
    def hist_info(n_cl, pred, gt):
        """Confusion matrix over valid GT labels (excludes 255)."""
        assert pred.shape == gt.shape
        k = (gt >= 0) & (gt < n_cl)
        labeled = np.sum(k)
        correct = np.sum(pred[k] == gt[k])
        hist = np.bincount(
            n_cl * gt[k].astype(int) + pred[k].astype(int), minlength=n_cl ** 2
        ).reshape(n_cl, n_cl)
        return hist, correct, labeled

    def per_class_iu(self, hist):
        return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

    def add_batch(self, semantics_pred, semantics_gt, mask_lidar, mask_camera):
        self.cnt += 1
        if self.use_image_mask:
            semantics_gt = semantics_gt[mask_camera]
            semantics_pred = semantics_pred[mask_camera]
        elif self.use_lidar_mask:
            semantics_gt = semantics_gt[mask_lidar]
            semantics_pred = semantics_pred[mask_lidar]
        hist, _, _ = self.hist_info(
            self.num_classes, semantics_pred.flatten(), semantics_gt.flatten()
        )
        self.hist += hist

    def count_miou(self, verbose=True):
        miou = self.per_class_iu(self.hist)
        if verbose:
            print(f"===> per class IoU of {self.cnt} samples:")
            for i in range(self.num_classes - 1):
                print(f"===> {self.class_names[i]} - IoU = {round(miou[i] * 100, 2)}")
            print(
                f"===> mIoU of {self.cnt} samples: "
                f"{round(np.nanmean(miou[: self.num_classes - 1]) * 100, 2)}"
            )
        return miou



def evaluate_occ(dataset, results, save_dir=None, use_image_mask=True, verbose=True):
    """Evaluate ordered model outputs against the dataset's GT.

    Args:
        dataset: NuScenesOccDataset (sequential order == results order).
        results: list of {"pred_occ": uint8 (Dx, Dy, Dz) array/tensor}.
        save_dir: optional per-token pred.npz dump directory.

    Returns:
        {"mIoU": float, "<class>_IoU": float, ...}
    """
    metric = MetricMIoU(use_image_mask=use_image_mask)
    for i, res in enumerate(results):
        info = dataset.data_list[i]
        npz = np.load(osp.join(info["occ_path"], "labels.npz"))
        semantics = npz["semantics"]
        mask_lidar = npz["mask_lidar"].astype(bool)
        mask_camera = npz["mask_camera"].astype(bool)

        pred = res["pred_occ"]
        if not isinstance(pred, np.ndarray):
            pred = pred.numpy() if hasattr(pred, "numpy") else np.asarray(pred)
        metric.add_batch(pred.astype(np.uint8), semantics, mask_lidar, mask_camera)

        if save_dir is not None:
            scene = info.get("scene_name") or [
                t for t in info["occ_path"].split("/") if "scene-" in t
            ][0]
            out_dir = osp.join(save_dir, scene, info["token"])
            os.makedirs(out_dir, exist_ok=True)
            np.savez_compressed(
                osp.join(out_dir, "pred.npz"),
                pred=pred, gt=semantics, sample_token=info["token"],
            )

    miou = metric.count_miou(verbose=verbose)
    eval_res = dict(
        mIoU=round(float(np.nanmean(miou[: metric.num_classes - 1])) * 100, 2)
    )
    for i, name in enumerate(metric.class_names[: metric.num_classes - 1]):
        eval_res[f"{name}_IoU"] = round(float(miou[i]) * 100, 2)
    return eval_res
