"""渲染 torch / C++ 检测框到 6 相机 + BEV 视图。

用法（3rdparty/deeplearning 下）：
  python deploy/scripts/vis_torch.py --dump-dir outputs/model_diff \
      --data-dir outputs/example-data [--cpp-boxes cpp_boxes.txt]
输出：torch_vis.jpg、cpp_vis.jpg、compare_vis.jpg（--out-dir，默认 outputs/model_diff）。
"""

import argparse
import os
import os.path as osp
import sys

_HERE = osp.dirname(osp.abspath(__file__))
for _path in (_HERE, osp.join(_HERE, "..", "..")):  # scripts/, deeplearning/
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np  # noqa: E402
from deploy.scripts.common import load_tensor, parse_cpp_boxes  # noqa: E402
from PIL import Image

from scripts.render import DET_CLASSES, DET_COLORS, compose_det_frame, render_det_bev  # noqa: E402
from scripts.visualize import CAM_NAMES, shifted_corners  # noqa: E402

CAM_ORDER = [2, 0, 1, 4, 3, 5]  # 显示顺序 FL,F,FR / BL,B,BR
FILE_NAMES = ["FRONT", "FRONT_RIGHT", "FRONT_LEFT", "BACK", "BACK_LEFT", "BACK_RIGHT"]


def _filter(boxes, score_col, thr, max_boxes):
    """boxes [N,>=score_col+1]；按 score 过滤+排序截断，返回行索引。"""
    sc = boxes[:, score_col]
    keep = np.where(sc >= thr)[0]
    keep = keep[np.argsort(-sc[keep])[: max_boxes]]
    return keep


def _render(boxes, colors, labels, pts, l2i, data_dir, title, out, pc_range=54.0):
    corners, centers = shifted_corners(boxes[:, :9])
    bev = render_det_bev(pts, corners, centers, colors, labels, pc_range=pc_range)
    cam_imgs, cam_corners, cam_P, cam_colors, cam_labels = [], [], [], [], []
    for c in CAM_ORDER:
        cam_imgs.append(Image.open(osp.join(data_dir, f"{c}-{FILE_NAMES[c]}.jpg")).convert("RGB"))
        cam_corners.append(corners)
        cam_P.append(l2i[c])
        cam_colors.append(colors)
        cam_labels.append(labels)
    frame = compose_det_frame(
        cam_imgs, cam_corners, cam_P, cam_colors, cam_labels, bev, title,
        cam_names=CAM_NAMES)
    frame.save(out)
    print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump-dir", default="outputs/model_diff")
    ap.add_argument("--data-dir", default="outputs/example-data")
    ap.add_argument("--cpp-boxes", default=None)
    ap.add_argument("--score-thr", type=float, default=0.12)
    ap.add_argument("--max-boxes", type=int, default=100)
    ap.add_argument("--out-dir", default="outputs/model_diff")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    pts = load_tensor(osp.join(args.data_dir, "points.tensor"))
    l2i = load_tensor(osp.join(args.data_dir, "lidar2image.tensor"))
    data_dir = args.data_dir

    # torch 框（boxes.npy: x,y,z,w,l,h,rot,vx,vy,score,label）
    tb = np.load(osp.join(args.dump_dir, "torch", "boxes.npy"))
    ti = _filter(tb, 9, args.score_thr, args.max_boxes)
    if len(ti):
        tbc = tb[ti]
        tcolors = [DET_COLORS[int(tbc[i, 10]) % len(DET_COLORS)] for i in range(len(tbc))]
        tlabels = [f"{DET_CLASSES[int(tbc[i, 10])]} {tbc[i, 9]:.2f}" for i in range(len(tbc))]
        _render(tbc, tcolors, tlabels, pts, l2i, data_dir,
                f"torch | {len(tbc)} boxes (score>={args.score_thr})",
                osp.join(args.out_dir, "torch_vis.jpg"))
    else:
        print("torch boxes 为空（阈值内无框）")

    # C++ 框（cpp_boxes.txt: x,y,z,w,l,h,rot,score,id）
    if args.cpp_boxes and osp.exists(args.cpp_boxes):
        cb = parse_cpp_boxes(args.cpp_boxes)
        ci = _filter(cb, 7, args.score_thr, args.max_boxes)
        if len(ci):
            cbc = cb[ci]
            ccolors = [DET_COLORS[int(cbc[i, 8]) % len(DET_COLORS)] for i in range(len(cbc))]
            clabels = [f"cpp {DET_CLASSES[int(cbc[i, 8])]} {cbc[i, 7]:.2f}" for i in range(len(cbc))]
            _render(cbc, ccolors, clabels, pts, l2i, data_dir,
                    f"cpp | {len(cbc)} boxes (score>={args.score_thr})",
                    osp.join(args.out_dir, "cpp_vis.jpg"))

            # 叠加图：torch 类色 + C++ 洋红
            if len(ti):
                allb = np.concatenate([tbc[:, :9], cbc[:, :9]], axis=0)
                allc = tcolors + ["#ff00ff" for _ in range(len(cbc))]
                alll = tlabels + [f"cpp {DET_CLASSES[int(cbc[i, 8])]} {cbc[i, 7]:.2f}"
                                  for i in range(len(cbc))]
                _render(allb, allc, alll, pts, l2i, data_dir,
                        f"compare torch({len(tbc)}) vs cpp({len(cbc)})",
                        osp.join(args.out_dir, "compare_vis.jpg"))


if __name__ == "__main__":
    main()
