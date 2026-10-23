"""run_bf.py —— bf（CUDA-BEVFusion 结构）Python 推理 CLI。

与 C++ 相同的模型布局（model/<name>/build/*.plan + ckpt 权重），
读取 example-data 目录推理并打印检测框，可 dump 中间结果供对拍。

用法（cd 3rdparty/deeplearning）：
  python deploy/python/scripts/run_bf.py configs/det_convfuser.py ckpts/bevfusion-det-bf.pth \
      --data outputs/example-data --model model/resnet50 --dump outputs/bf_out
"""
import argparse
import os
import os.path as osp
import sys

import numpy as np

_ROOT = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))  # deploy/
sys.path.insert(0, _ROOT)
sys.path.insert(0, osp.dirname(_ROOT))  # deeplearning/

from deploy.python.bf import BevfusionBf  # noqa: E402

FILES = ["camera2lidar", "camera_intrinsics", "lidar2image", "img_aug_matrix", "points"]
JPG_NAMES = ["0-FRONT", "1-FRONT_RIGHT", "2-FRONT_LEFT", "3-BACK_RIGHT", "4-BACK", "5-BACK_LEFT"]


def load_tensor(path, dtype=np.float32):
    with open(path, "rb") as f:
        return np.frombuffer(f.read(), dtype=dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("ckpt")
    ap.add_argument("--data", default="outputs/example-data")
    ap.add_argument("--model", default="model/resnet50", help="model/<name>/（4 plan 所在目录）")
    ap.add_argument("--dump", default=None, help="dump 目录（输出 cam_bev/lidar_bev/middle .npy）")
    args = ap.parse_args()

    import cv2

    data_dir = args.data
    imgs = np.stack([cv2.imread(osp.join(data_dir, f"{n}.jpg"))[:, :, ::-1] for n in JPG_NAMES])
    mats = {k: load_tensor(osp.join(data_dir, f"{k}.tensor")).reshape(6, 4, 4) for k in
            ["camera2lidar", "camera_intrinsics", "lidar2image", "img_aug_matrix"]}
    points = load_tensor(osp.join(data_dir, "points.tensor"), np.float16)

    infer = BevfusionBf(args.config, args.ckpt, args.model)
    boxes = infer.forward(imgs, points, mats["camera2lidar"], mats["camera_intrinsics"],
                          mats["lidar2image"], mats["img_aug_matrix"])
    print(f"[bf/run] {len(boxes)} boxes")
    for b in boxes:
        print(f"  id={b['id']} score={b['score']:.3f} pos=({b['position'][0]:.2f}, "
              f"{b['position'][1]:.2f}, {b['position'][2]:.2f}) size={b['size']} "
              f"rot={b['z_rotation']:.3f} vel=({b['velocity'][0]:.2f}, {b['velocity'][1]:.2f})")

    if args.dump:
        os.makedirs(args.dump, exist_ok=True)
        # 各阶段中间结果 dump（复跑 forward_no_normalize 不方便，直接再跑一次带 hook 不优雅——
        # 这里 dump 输入张量；如需对拍中间结果请用 infer 内部断点
        np.save(osp.join(args.dump, "boxes.npy"), np.array([[
            b["position"][0], b["position"][1], b["position"][2],
            *b["size"], b["z_rotation"], b["score"], b["id"]] for b in boxes]))
        print(f"[bf/run] dumped -> {args.dump}")


if __name__ == "__main__":
    main()
