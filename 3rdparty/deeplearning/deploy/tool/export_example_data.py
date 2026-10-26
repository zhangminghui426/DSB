"""导出 C++ 推理输入（example-data：.tensor + 6 相机 jpg）。

用法（3rdparty/deeplearning 下）：
  python deploy/tool/export_example_data.py configs/det_convfuser.py \
      ckpts/bevfusion-det-bf.pth --out-dir outputs/example-data
之后：cd deploy && bash tool/run.sh
"""

import argparse
import os
import os.path as osp
import shutil
import sys

_HERE = osp.dirname(osp.abspath(__file__))
for _path in (_HERE, osp.dirname(osp.dirname(_HERE))):  # tool/, deeplearning/
    if _path not in sys.path:
        sys.path.insert(0, _path)

from common import save_tensor  # noqa: E402

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

    filenames = list(metas[0]["filename"]) if isinstance(metas, (list, tuple)) \
        else list(metas["filename"])
    for idx, path in enumerate(filenames):
        shutil.copyfile(path, osp.join(args.out_dir, f"{idx}-{CAM_NAMES[idx]}.jpg"))

    for out_name in ("camera2lidar", "camera_intrinsics", "lidar2image", "img_aug_matrix"):
        save_tensor(batch[out_name][0].detach().cpu().float().numpy(),
                    osp.join(args.out_dir, f"{out_name}.tensor"))

    pts = batch["points"][0].detach().cpu().half().numpy()
    save_tensor(pts, osp.join(args.out_dir, "points.tensor"))
    print(f"[example] {args.out_dir} (points {pts.shape})")


if __name__ == "__main__":
    main()
