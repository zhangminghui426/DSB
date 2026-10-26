#!/usr/bin/env python
"""torch 参考对拍：训练接口（build_dataset + collate_fn）跑 det 全链 -> 3D 框。

example-data 由同一训练管线导出（sample-index 默认 0，与
deploy/tool/export_example_data.py 一致），保证与 C++（deploy/build/bevfusion）
输入约定完全相同，用于对比 C++ 输出的 3D 框。

用法（3rdparty/deeplearning 下）：
  python deploy/tool/torch_ref_boxes.py configs/det_convfuser.py ckpts/bevfusion-det-bf.pth [--sample-index 0]
"""
import argparse

import torch

from scripts.config import load_config
from models import build_model
from scripts.trainer import load_model_checkpoint, to_device
from datasets import build_dataset, collate_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("ckpt")
    ap.add_argument("--sample-index", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    model = build_model(cfg.model).cuda().eval()
    msg, _ = load_model_checkpoint(model, args.ckpt, strict=True)
    print("loaded %s missing=%d unexpected=%d" % (args.ckpt, len(msg.missing_keys), len(msg.unexpected_keys)))

    ds = build_dataset(cfg.data.val)
    sample = ds[args.sample_index]
    batch = to_device(collate_fn([sample]), "cuda")

    with torch.no_grad():
        out = model.forward(
            img=batch["img"], points=batch["points"], radar=batch.get("radar"),
            camera2ego=batch["camera2ego"], lidar2ego=batch["lidar2ego"],
            lidar2camera=batch["lidar2camera"], lidar2image=batch["lidar2image"],
            camera_intrinsics=batch["camera_intrinsics"],
            camera2lidar=batch["camera2lidar"],
            img_aug_matrix=batch["img_aug_matrix"],
            lidar_aug_matrix=batch["lidar_aug_matrix"],
            metas=batch["metas"],
        )

    b = out[0]["boxes_3d"].tensor.cpu().numpy()
    s = out[0]["scores_3d"].cpu().numpy()
    l = out[0]["labels_3d"].cpu().numpy()
    print(f"[torch] sample={args.sample_index} {len(b)} boxes")
    for i in range(len(b)):
        print("  box[%d] id=%d score=%.3f pos=(%.2f,%.2f,%.2f) size=(%.2f,%.2f,%.2f) rot=%.3f"
              % (i, int(l[i]), s[i], b[i, 0], b[i, 1], b[i, 2],
                 b[i, 3], b[i, 4], b[i, 5], b[i, 6]))


if __name__ == "__main__":
    main()
