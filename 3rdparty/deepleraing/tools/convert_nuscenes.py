""" Generate nuScenes info pkls (train/val) with nuscenes-devkit.

Usage:
    python tools/convert_nuscenes.py \
        --dataroot data/nuscenes --version v1.0-mini --max_sweeps 10

Outputs {dataroot}/nuscenes_infos_{train,val}.pkl, schema
{'infos': [...], 'metadata': {'version': ...}} consumed by NuScenesDataset.
All data paths are stored RELATIVE to dataroot (portable across machines;
the dataset rebases them with its dataset_root).
"""

import argparse
import os
import os.path as osp
import pickle
import sys

sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), ".."))

import numpy as np
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
from tqdm import tqdm

from datasets.definitions import NAME_MAPPING as NameMapping

def relative_path(path: str, root_path: str) -> str:
    """ Store paths relative to the dataset root for portability."""
    path = str(path)
    root_abs = osp.abspath(root_path)
    path_abs = osp.abspath(path)
    if path_abs.startswith(root_abs):
        rel = osp.relpath(path_abs, root_abs)
        return rel
    return path


def create_nuscenes_infos(
    root_path,
    info_prefix="nuscenes",
    version="v1.0-trainval",
    max_sweeps=10,
):
    """ Create info file of nuscenes dataset.

    Args:
        root_path (str): Path of the data root (containing v1.0-* json dir,
            samples/, sweeps/, maps/).
        info_prefix (str): Prefix of the info file to be generated.
        version (str): Version of the data. Default: 'v1.0-trainval'.
        max_sweeps (int): Max number of lidar sweeps. Default: 10.
    """
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    from nuscenes.utils import splits

    available_vers = ["v1.0-trainval", "v1.0-test", "v1.0-mini"]
    assert version in available_vers
    if version == "v1.0-trainval":
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == "v1.0-test":
        train_scenes = splits.test
        val_scenes = []
    elif version == "v1.0-mini":
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    else:
        raise ValueError("unknown")

    # filter existing scenes.
    available_scenes = get_available_scenes(nusc)
    available_scene_names = [s["name"] for s in available_scenes]
    train_scenes = list(filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    train_scenes = set(
        [available_scenes[available_scene_names.index(s)]["token"] for s in train_scenes]
    )
    val_scenes = set(
        [available_scenes[available_scene_names.index(s)]["token"] for s in val_scenes]
    )

    test = "test" in version
    if test:
        print("test scene: {}".format(len(train_scenes)))
    else:
        print("train scene: {}, val scene: {}".format(len(train_scenes), len(val_scenes)))
    train_nusc_infos, val_nusc_infos = fill_trainval_infos(
        nusc, train_scenes, val_scenes, test, root_path, max_sweeps=max_sweeps
    )

    metadata = dict(version=version)
    if test:
        print("test sample: {}".format(len(train_nusc_infos)))
        data = dict(infos=train_nusc_infos, metadata=metadata)
        info_path = osp.join(root_path, "{}_infos_test.pkl".format(info_prefix))
        with open(info_path, "wb") as f:
            pickle.dump(data, f)
        print("saved", info_path)
    else:
        print("train sample: {}, val sample: {}".format(len(train_nusc_infos), len(val_nusc_infos)))
        data = dict(infos=train_nusc_infos, metadata=metadata)
        info_path = osp.join(root_path, "{}_infos_train.pkl".format(info_prefix))
        with open(info_path, "wb") as f:
            pickle.dump(data, f)
        print("saved", info_path)
        data = dict(infos=val_nusc_infos, metadata=metadata)
        info_val_path = osp.join(root_path, "{}_infos_val.pkl".format(info_prefix))
        with open(info_val_path, "wb") as f:
            pickle.dump(data, f)
        print("saved", info_val_path)


def get_available_scenes(nusc):
    """ Get available scenes from the input nuscenes class."""
    available_scenes = []
    print("total scene num: {}".format(len(nusc.scene)))
    for scene in nusc.scene:
        scene_token = scene["token"]
        scene_rec = nusc.get("scene", scene_token)
        sample_rec = nusc.get("sample", scene_rec["first_sample_token"])
        sd_rec = nusc.get("sample_data", sample_rec["data"]["LIDAR_TOP"])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec["token"])
            lidar_path = str(lidar_path)
            if os.getcwd() in lidar_path:
                lidar_path = lidar_path.split(f"{os.getcwd()}/")[-1]
            if not osp.isfile(lidar_path):
                scene_not_exist = True
                break
            else:
                break
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    print("exist scene num: {}".format(len(available_scenes)))
    return available_scenes


def fill_trainval_infos(
    nusc,
    train_scenes,
    val_scenes,
    test=False,
    root_path="",
    max_sweeps=10,
):
    """ Generate the train/val infos from the raw data."""
    train_nusc_infos = []
    val_nusc_infos = []
    token2idx = {}
    scene2location = {}

    for sample in tqdm(nusc.sample, desc="generate infos"):
        lidar_token = sample["data"]["LIDAR_TOP"]
        sd_rec = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        cs_record = nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
        pose_record = nusc.get("ego_pose", sd_rec["ego_pose_token"])
        lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)

        assert osp.isfile(lidar_path), f"lidar data missing: {lidar_path}"

        # location (from the scene's log record) — needed by the
        # LoadBEVSegmentation NuScenesMap backend
        scene_token = sample["scene_token"]
        if scene_token not in scene2location:
            log_rec = nusc.get("log", nusc.get("scene", scene_token)["log_token"])
            scene2location[scene_token] = log_rec["location"]

        info = {
            "lidar_path": relative_path(lidar_path, root_path),
            "token": sample["token"],
            "sweeps": [],
            "cams": dict(),
            "lidar2ego_translation": cs_record["translation"],
            "lidar2ego_rotation": cs_record["rotation"],
            "ego2global_translation": pose_record["translation"],
            "ego2global_rotation": pose_record["rotation"],
            "timestamp": sample["timestamp"],
            "prev_token": sample["prev"],
            "location": scene2location[scene_token],
        }

        l2e_r = info["lidar2ego_rotation"]
        l2e_t = info["lidar2ego_translation"]
        e2g_r = info["ego2global_rotation"]
        e2g_t = info["ego2global_translation"]
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        # obtain 6 image's information per frame
        camera_types = [
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_FRONT_LEFT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_BACK_RIGHT",
        ]
        for cam in camera_types:
            cam_token = sample["data"][cam]
            cam_path, _, cam_intrinsic = nusc.get_sample_data(cam_token)
            cam_info = obtain_sensor2top(
                nusc, cam_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, cam, root_path
            )
            cam_info.update(cam_intrinsic=cam_intrinsic)
            info["cams"].update({cam: cam_info})

        # obtain sweeps for a single key-frame
        sd_rec = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        sweeps = []
        while len(sweeps) < max_sweeps:
            if not sd_rec["prev"] == "":
                sweep = obtain_sensor2top(
                    nusc, sd_rec["prev"], l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, "lidar", root_path
                )
                sweeps.append(sweep)
                sd_rec = nusc.get("sample_data", sd_rec["prev"])
            else:
                break
        info["sweeps"] = sweeps

        # obtain annotation
        if not test:
            annotations = [nusc.get("sample_annotation", token) for token in sample["anns"]]
            locs = np.array([b.center for b in boxes]).reshape(-1, 3)
            dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
            rots = np.array([b.orientation.yaw_pitch_roll[0] for b in boxes]).reshape(-1, 1)
            velocity = np.array([nusc.box_velocity(token)[:2] for token in sample["anns"]])
            valid_flag = np.array(
                [(anno["num_lidar_pts"] + anno["num_radar_pts"]) > 0 for anno in annotations],
                dtype=bool,
            ).reshape(-1)
            # convert velo from global to lidar
            for i in range(len(boxes)):
                velo = np.array([*velocity[i], 0.0])
                velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                velocity[i] = velo[:2]

            names = [b.name for b in boxes]
            for i in range(len(names)):
                if names[i] in NameMapping:
                    names[i] = NameMapping[names[i]]
            names = np.array(names)
            # we need to convert rot to SECOND format.
            gt_boxes = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1)
            assert len(gt_boxes) == len(annotations), f"{len(gt_boxes)}, {len(annotations)}"
            info["gt_boxes"] = gt_boxes
            info["gt_names"] = names
            info["gt_velocity"] = velocity.reshape(-1, 2)
            info["num_lidar_pts"] = np.array([a["num_lidar_pts"] for a in annotations])
            info["num_radar_pts"] = np.array([a["num_radar_pts"] for a in annotations])
            info["valid_flag"] = valid_flag

        if sample["scene_token"] in train_scenes:
            train_nusc_infos.append(info)
            token2idx[info["token"]] = ("train", len(train_nusc_infos) - 1)
        else:
            val_nusc_infos.append(info)
            token2idx[info["token"]] = ("val", len(val_nusc_infos) - 1)

    for info in train_nusc_infos:
        prev_token = info["prev_token"]
        if prev_token == "":
            info["prev"] = -1
        else:
            prev_set, prev_idx = token2idx[prev_token]
            assert prev_set == "train"
            info["prev"] = prev_idx

    for info in val_nusc_infos:
        prev_token = info["prev_token"]
        if prev_token == "":
            info["prev"] = -1
        else:
            prev_set, prev_idx = token2idx[prev_token]
            assert prev_set == "val"
            info["prev"] = prev_idx

    return train_nusc_infos, val_nusc_infos


def obtain_sensor2top(
    nusc,
    sensor_token,
    l2e_t,
    l2e_r_mat,
    e2g_t,
    e2g_r_mat,
    sensor_type="lidar",
    root_path="",
):
    """ Obtain the info with RT matric from general sensor to Top LiDAR."""
    sd_rec = nusc.get("sample_data", sensor_token)
    cs_record = nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
    pose_record = nusc.get("ego_pose", sd_rec["ego_pose_token"])
    data_path = str(nusc.get_sample_data_path(sd_rec["token"]))
    if os.getcwd() in data_path:
        data_path = data_path.split(f"{os.getcwd()}/")[-1]
    sweep = {
        "data_path": relative_path(data_path, root_path),
        "type": sensor_type,
        "sample_data_token": sd_rec["token"],
        "sensor2ego_translation": cs_record["translation"],
        "sensor2ego_rotation": cs_record["rotation"],
        "ego2global_translation": pose_record["translation"],
        "ego2global_rotation": pose_record["rotation"],
        "timestamp": sd_rec["timestamp"],
    }
    l2e_r_s = sweep["sensor2ego_rotation"]
    l2e_t_s = sweep["sensor2ego_translation"]
    e2g_r_s = sweep["ego2global_rotation"]
    e2g_t_s = sweep["ego2global_translation"]

    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
    )
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
    )
    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T) + l2e_t @ np.linalg.inv(
        l2e_r_mat
    ).T
    sweep["sensor2lidar_rotation"] = R.T  # points @ R.T + T
    sweep["sensor2lidar_translation"] = T
    return sweep


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataroot", default="data/nuscenes", help="nuScenes data root")
    parser.add_argument("--version", default="v1.0-mini", help="v1.0-mini / v1.0-trainval / v1.0-test")
    parser.add_argument("--prefix", default="nuscenes", help="output pkl prefix")
    parser.add_argument("--max_sweeps", type=int, default=10)
    args = parser.parse_args()

    create_nuscenes_infos(
        root_path=args.dataroot,
        info_prefix=args.prefix,
        version=args.version,
        max_sweeps=args.max_sweeps,
    )


if __name__ == "__main__":
    main()
