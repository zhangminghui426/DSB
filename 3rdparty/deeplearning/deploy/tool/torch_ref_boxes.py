#!/usr/bin/env python
"""torch 参考框（训练接口全链），用于对比 C++ 输出。

用法（3rdparty/deeplearning 下）：
  python deploy/tool/torch_ref_boxes.py configs/det_convfuser.py ckpts/bevfusion-det-bf.pth
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
    load_model_checkpoint(model, args.ckpt, strict=True)

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
    print(f"[torch] sample={args.sample_index} {len(b)} boxes")


if __name__ == "__main__":
    main()
