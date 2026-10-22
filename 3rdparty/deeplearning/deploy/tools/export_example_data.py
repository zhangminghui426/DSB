"""导出 CUDA-BEVFusion 输入格式（example-data：.tensor + jpg）。

对应 deploy/cpp/src/main.cpp 的输入契约：
  camera2lidar.tensor      (6,4,4) float32   camera -> lidar 外参
  camera_intrinsics.tensor (6,4,4) float32   内参（4x4，下 3 行为旋转/平移）
  lidar2image.tensor       (6,4,4) float32   lidar -> 图像投影矩阵
  img_aug_matrix.tensor    (6,4,4) float32   图像增强矩阵
  points.tensor            (N,5)  float16   点云（x,y,z,intensity,time）
  0-FRONT.jpg ... 5-BACK_RIGHT.jpg           6 张原始相机图（1600x900）

用法：
  python deploy/tools/export_example_data.py configs/det_convfuser.py \
      ckpts/bevfusion-det-bf.pth --sample-index 0 --out-dir outputs/example-data

生成后：
  cd 3rdparty/deeplearning/deploy/cpp && . tool/environment.sh && \
  bash tool/run.sh（或 ./build/bevfusion <data-dir> resnet50int8 fp16）
"""

import argparse
import os
import os.path as osp
import shutil
import sys

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))  # deeplearning/
sys.path.insert(0, osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))  # 3rdparty/

import numpy as np

DTYPE_MAP = {"float32": 3, "float16": 2, "int32": 1, "int64": 4}
MAGIC = 0x33FF1101


def save_tensor(arr, path):
    """与 CUDA-BEVFusion common/tensor.hpp / tool/tensor.py 一致的二进制格式。"""
    arr = np.ascontiguousarray(arr)
    dtype = str(arr.dtype)
    assert dtype in DTYPE_MAP, f"unsupported dtype {dtype}"
    with open(path, "wb") as f:
        f.write(np.array([MAGIC, arr.ndim, DTYPE_MAP[dtype]], dtype=np.int32).tobytes())
        f.write(np.array(arr.shape, dtype=np.int32).tobytes())
        f.write(arr.tobytes())


CAM_NAMES = ["FRONT", "FRONT_RIGHT", "FRONT_LEFT", "BACK", "BACK_LEFT", "BACK_RIGHT"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config")
    ap.add_argument("checkpoint")
    ap.add_argument("--sample-index", type=int, default=0)
    ap.add_argument("--out-dir", default=osp.join("outputs", "example-data"))
    ap.add_argument("--options", nargs="*", default=[])
    args = ap.parse_args()

    from scripts.config import load_config
    from models import build_model
    from scripts.trainer import load_model_checkpoint, to_device
    from datasets import build_dataset, collate_fn

    cfg = load_config(args.config, options=args.options)
    model = build_model(cfg.model).cuda().eval()
    load_model_checkpoint(model, args.checkpoint, strict=True)

    ds = build_dataset(cfg.data.val)
    sample = ds[args.sample_index]
    batch = to_device(collate_fn([sample]), "cuda")
    metas = sample.get("metas") or batch["metas"]

    os.makedirs(args.out_dir, exist_ok=True)

    # 6 张原始相机图（1600x900）
    filenames = list(metas[0]["filename"]) if isinstance(metas, (list, tuple)) \
        else list(metas["filename"])
    for idx, path in enumerate(filenames):
        name = f"{idx}-{CAM_NAMES[idx]}.jpg"
        shutil.copyfile(path, osp.join(args.out_dir, name))
        print(f"[example] {name} <- {path}")

    # 相机/投影矩阵（去 batch 维，float32）
    mats = {
        "camera2lidar": "camera2lidar",
        "camera_intrinsics": "camera_intrinsics",
        "lidar2image": "lidar2image",
        "img_aug_matrix": "img_aug_matrix",
    }
    for out_name, key in mats.items():
        t = batch[key][0].detach().cpu().float().numpy()  # (6,4,4)
        save_tensor(t, osp.join(args.out_dir, f"{out_name}.tensor"))
        print(f"[example] {out_name}.tensor {t.shape} {t.dtype}")

    # 点云 (N,5) float16
    pts = batch["points"][0].detach().cpu().half().numpy()
    save_tensor(pts, osp.join(args.out_dir, "points.tensor"))
    print(f"[example] points.tensor {pts.shape} {pts.dtype} (N={pts.shape[0]})")

    print(f"[example] done -> {osp.join(args.out_dir)}")
    print("         cd 3rdparty/deeplearning/deploy/cpp && . tool/environment.sh")
    print("         ./build/bevfusion <上述目录> resnet50 fp16")


if __name__ == "__main__":
    main()
