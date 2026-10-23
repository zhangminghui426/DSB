#!/usr/bin/env python
"""torch 参考对拍：example-data -> torch det 3D 框（fp32）。

与 C++（deploy/cpp/build/bevfusion）输出的 3D 框数值对比，用于定位
"检测框位置不对"的环节（推理链 vs 可视化）。

用法（3rdparty/deeplearning 下）：
  python deploy/tools/torch_ref_boxes.py configs/det_convfuser.py ckpts/bevfusion-det-bf.pth outputs/example-data
"""
import argparse
import os

import numpy as np
import torch
import cv2

from scripts.config import load_config
from models import build_model
from scripts.trainer import load_model_checkpoint
from datasets.geometry.boxes import LiDARInstance3DBoxes

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255.0
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255.0
JPG = ["0-FRONT.jpg", "1-FRONT_RIGHT.jpg", "2-FRONT_LEFT.jpg",
       "3-BACK.jpg", "4-BACK_LEFT.jpg", "5-BACK_RIGHT.jpg"]
# 与 deploy/cpp/src/bevfusion/camera-normalization.cu 一致：
# resize 0.48 -> x center crop 32 / y bottom crop 176 -> 704x256
RESIZE_LIM = 0.48
OUT_W, OUT_H = 704, 256


def load_tensor(p):
    with open(p, "rb") as f:
        h = np.fromfile(f, dtype="<i4", count=3)
        ndim, dtypei = int(h[1]), int(h[2])
        dims = tuple(int(v) for v in np.fromfile(f, dtype="<i4", count=ndim))
        dt = {1: "<i4", 2: "<f2", 3: "<f4", 4: "<i8",
              5: "<u8", 6: "<u4", 7: "<i1", 8: "<u1"}[dtypei]
        return np.fromfile(f, dtype=dt).reshape(dims)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("ckpt")
    ap.add_argument("data")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model = build_model(cfg.model).cuda().eval()
    msg, _ = load_model_checkpoint(model, args.ckpt, strict=True)
    print("loaded %s missing=%d unexpected=%d" % (args.ckpt, len(msg.missing_keys), len(msg.unexpected_keys)))

    # ---- 图像：与 C++ normalizer 完全一致（resize + crop + ImageNet 归一化） ----
    imgs = []
    for name in JPG:
        im = cv2.imread(os.path.join(args.data, name))[:, :, ::-1].astype(np.float32)  # BGR->RGB, HWC
        h, w = im.shape[:2]
        rw, rh = int(w * RESIZE_LIM), int(h * RESIZE_LIM)
        im = cv2.resize(im, (rw, rh), interpolation=cv2.INTER_LINEAR)
        crop_x = (rw - OUT_W) // 2
        crop_y = rh - OUT_H  # bottom crop（与 camera-normalization 一致）
        im = im[crop_y:crop_y + OUT_H, crop_x:crop_x + OUT_W]
        assert im.shape[:2] == (OUT_H, OUT_W), im.shape
        im = (im - MEAN) / STD
        imgs.append(im.transpose(2, 0, 1))  # CHW
    img = torch.from_numpy(np.stack(imgs)[None]).float().cuda()  # [1,6,3,256,704]

    pts = load_tensor(os.path.join(args.data, "points.tensor"))
    points = [torch.from_numpy(pts).float().cuda()]  # list[B] [N,5]

    mats = {}
    for k in ["camera2lidar", "camera_intrinsics", "lidar2image", "img_aug_matrix"]:
        a = load_tensor(os.path.join(args.data, k + ".tensor"))
        mats[k] = torch.from_numpy(a).float().cuda()  # [6,4,4]
    lidar_aug = torch.eye(4, dtype=torch.float32).cuda().unsqueeze(0).repeat(6, 1, 1)

    metas = [{"box_type_3d": LiDARInstance3DBoxes}]

    with torch.no_grad():
        out = model.forward(
            img=img, points=points, radar=None,
            lidar2image=mats["lidar2image"],
            camera_intrinsics=mats["camera_intrinsics"],
            camera2lidar=mats["camera2lidar"],
            img_aug_matrix=mats["img_aug_matrix"],
            lidar_aug_matrix=lidar_aug,
            metas=metas,
        )

    b = out[0]["boxes_3d"].tensor.cpu().numpy()
    s = out[0]["scores_3d"].cpu().numpy()
    l = out[0]["labels_3d"].cpu().numpy()
    print(f"[torch] {len(b)} boxes")
    for i in range(len(b)):
        print("  box[%d] id=%d score=%.3f pos=(%.2f,%.2f,%.2f) size=(%.2f,%.2f,%.2f) rot=%.3f"
              % (i, int(l[i]), s[i], b[i, 0], b[i, 1], b[i, 2],
                 b[i, 3], b[i, 4], b[i, 5], b[i, 6]))


if __name__ == "__main__":
    main()
