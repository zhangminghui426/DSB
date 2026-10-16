"""Enrich base nuScenes infos pkls into the BEVDet/FlashOCC protocol.

Adds per-info: ann_infos (ego-frame GT boxes + labels via get_gt), scene_token,
scene_name and occ_path (./data/nuscenes/gts/<scene>/<token>) — the keys the
occupancy dataset/pipeline consume. The BASE infos pkls
({tag}_infos_{train,val}.pkl with cams/sweeps/poses) are produced by the
legacy mmdet3d nuscenes converter (kept in the original FlashOCC repo);
this tool covers the FlashOCC-specific enrichment step.

Usage (from repo root):
    python tools/create_data.py --dataroot data/nuscenes --version v1.0-mini \
        --tag bevdetv2-nuscenes
"""

import argparse
import os.path as osp
import pickle
import sys

sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), ".."))

import numpy as np
from nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion

map_name_from_general_to_detection = {
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.wheelchair": "ignore",
    "human.pedestrian.stroller": "ignore",
    "human.pedestrian.personal_mobility": "ignore",
    "human.pedestrian.police_officer": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "animal": "ignore",
    "vehicle.car": "car",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.bicycle": "bicycle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.truck": "truck",
    "vehicle.construction": "construction_vehicle",
    "vehicle.emergency.ambulance": "ignore",
    "vehicle.emergency.police": "ignore",
    "vehicle.trailer": "trailer",
    "movable_object.barrier": "barrier",
    "movable_object.trafficcone": "traffic_cone",
    "movable_object.pushable_pullable": "ignore",
    "movable_object.debris": "ignore",
    "static_object.bicycle_rack": "ignore",
}
classes = [
    "car", "truck", "construction_vehicle", "bus", "trailer", "barrier",
    "motorcycle", "bicycle", "pedestrian", "traffic_cone",
]


def get_gt(info):
    """Ego-frame GT boxes (x,y,z,dx,dy,dz,yaw,vx,vy) + labels from ann_infos."""
    ego2global_rotation = info["cams"]["CAM_FRONT"]["ego2global_rotation"]
    ego2global_translation = info["cams"]["CAM_FRONT"]["ego2global_translation"]
    trans = -np.array(ego2global_translation)
    rot = Quaternion(ego2global_rotation).inverse
    gt_boxes, gt_labels = [], []
    for ann_info in info["ann_infos"]:
        if (map_name_from_general_to_detection[ann_info["category_name"]] not in classes
                or ann_info["num_lidar_pts"] + ann_info["num_radar_pts"] <= 0):
            continue
        box = Box(
            ann_info["translation"], ann_info["size"], Quaternion(ann_info["rotation"]),
            velocity=ann_info["velocity"],
        )
        box.translate(trans)
        box.rotate(rot)
        gt_box = np.concatenate([
            np.array(box.center),
            np.array(box.wlh)[[1, 0, 2]],
            np.array([box.orientation.yaw_pitch_roll[0]]),
            np.array(box.velocity[:2]),
        ])
        gt_boxes.append(gt_box)
        gt_labels.append(
            classes.index(map_name_from_general_to_detection[ann_info["category_name"]])
        )
    return gt_boxes, gt_labels


def add_ann_adj_info(dataroot, version, extra_tag):
    """Rewrite {tag}_infos_{train,val}.pkl in place with the FlashOCC keys."""
    nuscenes = NuScenes(version, dataroot, verbose=True)
    for split in ("train", "val"):
        pkl_path = osp.join(dataroot, f"{extra_tag}_infos_{split}.pkl")
        with open(pkl_path, "rb") as f:
            dataset = pickle.load(f)
        for idx in range(len(dataset["infos"])):
            if idx % 10 == 0:
                print(f"{split}: {idx}/{len(dataset['infos'])}")
            info = dataset["infos"][idx]
            sample = nuscenes.get("sample", info["token"])
            ann_infos = []
            for ann in sample["anns"]:
                ann_info = nuscenes.get("sample_annotation", ann)
                velocity = nuscenes.box_velocity(ann_info["token"])
                if np.any(np.isnan(velocity)):
                    velocity = np.zeros(3)
                ann_info["velocity"] = velocity
                ann_infos.append(ann_info)
            dataset["infos"][idx]["ann_infos"] = ann_infos
            dataset["infos"][idx]["ann_infos"] = get_gt(dataset["infos"][idx])
            dataset["infos"][idx]["scene_token"] = sample["scene_token"]
            scene = nuscenes.get("scene", sample["scene_token"])
            dataset["infos"][idx]["scene_name"] = scene["name"]
            dataset["infos"][idx]["occ_path"] = osp.join(
                dataroot, "gts", scene["name"], info["token"]
            )
        with open(pkl_path, "wb") as fid:
            pickle.dump(dataset, fid)
        print("saved", pkl_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataroot", default="data/nuscenes")
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--tag", default="bevdetv2-nuscenes", help="infos pkl prefix")
    args = parser.parse_args()
    add_ann_adj_info(args.dataroot, args.version, args.tag)


if __name__ == "__main__":
    main()
