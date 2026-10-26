"""torch 阶段：导出 C++ 推理输入（example-data）与 torch 参考框。

用法（3rdparty/deeplearning 下）：
  python deploy/scripts/example_data.py configs/det_convfuser.py ckpts/bevfusion-det-bf.pth \
      --out-dir outputs/example-data --sample-index 0

输出：<out-dir>/points.tensor、4 个矩阵 .tensor、6 张相机 jpg；
      同时跑一遍模型打印参考框数量（逐框 dump 用 model_diff.py --backend torch）。
"""

import argparse
import os
import os.path as osp
import shutil
import sys

_HERE = osp.dirname(osp.abspath(__file__))
_DL_ROOT = osp.dirname(osp.dirname(_HERE))  # 3rdparty/deeplearning
for _path in (_HERE, _DL_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from deploy.scripts.common import CAM_NAMES, save_tensor  # noqa: E402

MATRICES = ("camera2lidar", "camera_intrinsics", "lidar2image", "img_aug_matrix")


def main():
    import torch
    from datasets import build_dataset, collate_fn
    from models import build_model
    from scripts.config import load_config
    from scripts.trainer import load_model_checkpoint, to_device

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("checkpoint")
    ap.add_argument("--sample-index", type=int, default=0)
    ap.add_argument("--out-dir", default=osp.join("outputs", "example-data"))
    ap.add_argument("--options", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config, options=args.options)
    model = build_model(cfg.model).cuda().eval()
    load_model_checkpoint(model, args.checkpoint, strict=True)
    sample = build_dataset(cfg.data.val)[args.sample_index]
    batch = to_device(collate_fn([sample]), "cuda")
    metas = sample.get("metas") or batch["metas"]

    out = args.out_dir
    os.makedirs(out, exist_ok=True)

    filenames = list(metas[0]["filename"]) if isinstance(metas, (list, tuple)) else list(metas["filename"])
    for idx, path in enumerate(filenames):
        shutil.copyfile(path, osp.join(out, f"{idx}-{CAM_NAMES[idx]}.jpg"))

    for name in MATRICES:
        save_tensor(batch[name][0].detach().cpu().float().numpy(), osp.join(out, f"{name}.tensor"))
    points = batch["points"][0].detach().cpu().half().numpy()
    save_tensor(points, osp.join(out, "points.tensor"))

    with torch.no_grad():
        pred = model.forward(
            img=batch["img"], points=batch["points"], radar=batch.get("radar"),
            camera2ego=batch["camera2ego"], lidar2ego=batch["lidar2ego"],
            lidar2camera=batch["lidar2camera"], lidar2image=batch["lidar2image"],
            camera_intrinsics=batch["camera_intrinsics"], camera2lidar=batch["camera2lidar"],
            img_aug_matrix=batch["img_aug_matrix"], lidar_aug_matrix=batch["lidar_aug_matrix"],
            metas=batch["metas"],
        )
    boxes = pred[0]["boxes_3d"].tensor

    print(f"[torch] {out}: points {points.shape}, {len(boxes)} ref boxes")
    print("[torch] 逐框/逐模块 dump: python deploy/scripts/model_diff.py ... --backend torch")


if __name__ == "__main__":
    main()
