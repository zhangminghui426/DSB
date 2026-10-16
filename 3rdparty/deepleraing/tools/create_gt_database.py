"""Generate the nuScenes GT database (object point crops) + dbinfos pkl for
the ObjectPaste (gt-sampling) augmentation.

Usage (from repo root, AFTER convert_nuscenes.py):
    python tools/create_gt_database.py --dataroot data/nuscenes --version v1.0-mini

Outputs:
    {dataroot}/nuscenes_gt_database/{token}_{name}_{i}.bin   (float32 N,5)
    {dataroot}/nuscenes_dbinfos_train.pkl
"""

import argparse
import os.path as osp
import pickle
import sys

sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), ".."))

import numpy as np
from tqdm import tqdm

from datasets import build_dataset
from datasets.definitions import OBJECT_CLASSES
from datasets.geometry import box_np_ops
from datasets.preprocess import run_pipeline


def create_groundtruth_database(
    data_path,
    info_prefix="nuscenes",
    used_classes=None,
    database_save_path=None,
    db_info_save_path=None,
):
    print("Create GT Database of NuScenesDataset")
    info_path = osp.join(data_path, f"{info_prefix}_infos_train.pkl")
    dataset_cfg = dict(
        type="NuScenesDataset",
        dataset_root=data_path,
        ann_file=info_path,
        use_valid_flag=True,
        object_classes=OBJECT_CLASSES,
        map_classes=None,
        modality=dict(use_lidar=True, use_camera=False, use_radar=False, use_map=False, use_external=False),
        test_mode=False,
        box_type_3d="LiDAR",
        pipeline=[
            dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=5, use_dim=5),
            dict(
                type="LoadPointsFromMultiSweeps",
                sweeps_num=10,
                use_dim=[0, 1, 2, 3, 4],
                pad_empty_sweeps=True,
                remove_close=True,
            ),
        ],
    )
    dataset = build_dataset(dataset_cfg)

    if database_save_path is None:
        database_save_path = osp.join(data_path, f"{info_prefix}_gt_database")
    if db_info_save_path is None:
        db_info_save_path = osp.join(data_path, f"{info_prefix}_dbinfos_train.pkl")
    import os

    os.makedirs(database_save_path, exist_ok=True)

    all_db_infos = dict()
    group_counter = 0
    for j in tqdm(range(len(dataset)), desc="gt database"):
        input_dict = dataset.get_data_info(j)
        example = run_pipeline(dataset.pipeline, input_dict)
        if example is None:
            continue
        image_idx = example["token"]
        points = example["points"].tensor.numpy()
        gt_boxes_3d = example["gt_bboxes_3d"].tensor.numpy()
        names = example["gt_names"]
        group_dict = dict()
        group_ids = np.arange(gt_boxes_3d.shape[0], dtype=np.int64)
        difficulty = np.zeros(gt_boxes_3d.shape[0], dtype=np.int32)

        num_obj = gt_boxes_3d.shape[0]
        if num_obj == 0:
            continue
        point_indices = box_np_ops.points_in_rbbox(points, gt_boxes_3d)

        for i in range(num_obj):
            filename = f"{image_idx}_{names[i]}_{i}.bin"
            abs_filepath = osp.join(database_save_path, filename)
            rel_filepath = osp.join(f"{info_prefix}_gt_database", filename)

            gt_points = points[point_indices[:, i]]
            gt_points[:, :3] -= gt_boxes_3d[i, :3]

            with open(abs_filepath, "w") as f:
                gt_points.tofile(f)

            if (used_classes is None) or names[i] in used_classes:
                db_info = {
                    "name": names[i],
                    "path": rel_filepath,
                    "image_idx": image_idx,
                    "gt_idx": i,
                    "box3d_lidar": gt_boxes_3d[i],
                    "num_points_in_gt": gt_points.shape[0],
                    "difficulty": difficulty[i],
                }
                local_group_id = group_ids[i]
                if local_group_id not in group_dict:
                    group_dict[local_group_id] = group_counter
                    group_counter += 1
                db_info["group_id"] = group_dict[local_group_id]
                if names[i] in all_db_infos:
                    all_db_infos[names[i]].append(db_info)
                else:
                    all_db_infos[names[i]] = [db_info]

    for k, v in all_db_infos.items():
        print(f"load {len(v)} {k} database infos")

    with open(db_info_save_path, "wb") as f:
        pickle.dump(all_db_infos, f)
    print("saved", db_info_save_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataroot", default="data/nuscenes")
    parser.add_argument("--prefix", default="nuscenes")
    parser.add_argument("--version", default="v1.0-mini", help="(informational)")
    args = parser.parse_args()
    create_groundtruth_database(args.dataroot, info_prefix=args.prefix)


if __name__ == "__main__":
    main()
