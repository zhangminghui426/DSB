"""渲染 torch 推理框到 6 相机 + BEV 雷达视图（与 C++ cuda-bevfusion.jpg 同数据、同风格）。

数据源：outputs/example-data（与 C++ deploy 完全同数据：6 相机 jpg + points + lidar2image）。
框源：model_diff 的 torch/trt 后端 dump（boxes.npy），可叠加 C++ 打印框（--cpp-boxes）。

用法：
  python deploy/tools/vis_torch.py --dump-dir outputs/model_diff --data-dir outputs/example-data \\
      --cpp-boxes outputs/model_diff/cpp_boxes.txt --score-thr 0.12

输出（--out-dir，默认 outputs/model_diff）：
  torch_vis.jpg      torch 推理框（类色）
  cpp_vis.jpg        C++ 框（类色，需 --cpp-boxes）
  compare_vis.jpg    叠加：torch 类色 + C++ 洋红
"""

import argparse
import os
import os.path as osp
import sys

sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), "..", ".."))  # deeplearning/

import numpy as np
from PIL import Image

from scripts.render import DET_CLASSES, DET_COLORS, compose_det_frame, render_det_bev
from scripts.visualize import CAM_NAMES, shifted_corners

CAM_ORDER = [2, 0, 1, 4, 3, 5]  # 显示顺序 FL,F,FR / BL,B,BR
FILE_NAMES = ["FRONT", "FRONT_RIGHT", "FRONT_LEFT", "BACK", "BACK_LEFT", "BACK_RIGHT"]


def load_tensor(p):
    with open(p, "rb") as f:
        h = np.fromfile(f, dtype="<i4", count=3)
        d = tuple(np.fromfile(f, dtype="<i4", count=h[1]))
        dt = {2: np.float16, 3: np.float32}.get(h[2], np.float32)
        return np.fromfile(f, dtype=dt).reshape(d)


def _parse_cpp_boxes(path):
    import re
    boxes = []
    for line in open(path):
        m = re.search(
            r"id=(\d+) score=([-\d.]+) pos=\(([-\d.]+),([-\d.]+),([-\d.]+)\) "
            r"size=\(([-\d.]+),([-\d.]+),([-\d.]+)\) rot=([-\d.]+)", line)
        if m:
            g = m.groups()
            boxes.append([float(g[2]), float(g[3]), float(g[4]),
                          float(g[5]), float(g[6]), float(g[7]),
                          float(g[8]), float(g[1]), float(g[0])])
    return np.array(boxes, dtype=np.float32)


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
        cb = _parse_cpp_boxes(args.cpp_boxes)
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
                allc = tcolors + [[255, 0, 255] for _ in range(len(cbc))]
                alll = tlabels + [f"cpp {DET_CLASSES[int(cbc[i, 8])]} {cbc[i, 7]:.2f}"
                                  for i in range(len(cbc))]
                _render(allb, allc, alll, pts, l2i, data_dir,
                        f"compare torch({len(tbc)}) vs cpp({len(cbc)})",
                        osp.join(args.out_dir, "compare_vis.jpg"))


if __name__ == "__main__":
    main()
