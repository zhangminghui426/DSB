"""可视化：检测框叠加图与延时/精度对比报告。"""

import json
import os.path as osp

import numpy as np

from deploy.scripts.accuracy import AccuracyAnalyzer
from deploy.scripts.common import CAM_NAMES, GRAPHS, STAGE_TENSORS, load_tensor, parse_cpp_boxes

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e1"
COLORS = {"torch": "#2a78d6", "onnx": "#eb6834", "trt": "#1baf7a"}


class Visualizer:

    def __init__(self, dump_dir, data_dir):
        self.dump_dir = dump_dir
        self.data_dir = data_dir

    # ---------------------------------------------------------- boxes
    @staticmethod
    def _visible(boxes, thr=0.12):
        """boxes.npy 第 9 列为 score 时按阈值过滤。"""
        return boxes[boxes[:, 9] >= thr] if boxes.shape[1] >= 11 else boxes

    def boxes(self, cpp_boxes_txt=None):
        """torch vs onnx / trt 框叠加图（BEV + 6 相机）。"""
        cpp_sets = ()
        if cpp_boxes_txt and osp.exists(cpp_boxes_txt):
            cpp_sets = ((parse_cpp_boxes(cpp_boxes_txt), "blue", "cpp"),)
        for sub, extra in [("onnx", ()), ("trt", cpp_sets)]:
            subs = [("torch", "limegreen", "torch"), (sub, "red", sub)]
            self._plot_boxes(osp.join(self.dump_dir, f"{sub}_vis.jpg"), subs, extra)

    def _plot_boxes(self, out, subs, extra_sets=()):
        import cv2
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        sets = list(extra_sets)
        for sub, color, label in subs:
            path = osp.join(self.dump_dir, sub, "boxes.npy")
            if osp.exists(path):
                sets.append((self._visible(np.load(path)), color, label))

        l2i = load_tensor(osp.join(self.data_dir, "lidar2image.tensor"))
        pts = load_tensor(osp.join(self.data_dir, "points.tensor"))

        fig, axes = plt.subplots(3, 3, figsize=(26, 15))
        ax = axes[0][0]
        if pts.ndim >= 2 and pts.shape[1] >= 3:
            sel = pts[::5, :3]
            ax.scatter(sel[:, 0], -sel[:, 1], s=0.3, c="lightgray", alpha=0.4)
        for arr, color, _ in sets:
            for b in arr:
                x, y, _, w, l, _, rot = b[:7]
                c, s = np.cos(rot), np.sin(rot)
                rect = np.array([[-w / 2, -l / 2], [w / 2, -l / 2], [w / 2, l / 2],
                                 [-w / 2, l / 2], [-w / 2, -l / 2]]) @ np.array([[c, -s], [s, c]]).T
                rect += np.array([x, y])
                ax.plot(rect[:, 0], -rect[:, 1], color=color, lw=1.2)
        ax.set_xlim(-54, 54)
        ax.set_ylim(-54, 54)
        ax.set_aspect("equal")
        ax.set_title("BEV " + " / ".join(f"{label}({color})" for _, color, label in sets))
        ax.grid(True, alpha=0.3)

        cam_axes = [axes[0][1], axes[0][2], axes[1][0], axes[1][1], axes[1][2], axes[2][0]]
        for c, axc in enumerate(cam_axes):
            img = cv2.imread(osp.join(self.data_dir, f"{c}-{CAM_NAMES[c]}.jpg"))
            if img is None:
                continue
            height, width = img.shape[:2]
            axc.imshow(img[:, :, ::-1])
            for arr, color, _ in sets:
                for b in arr:
                    x, y, z, w, l, h, rot = b[:7]  # z 为框中心（nuScenes/预测同约定）
                    cc, ss = np.cos(rot), np.sin(rot)
                    base = np.array([[w / 2, l / 2], [w / 2, -l / 2],
                                     [-w / 2, -l / 2], [-w / 2, l / 2]]) @ np.array([[cc, -ss], [ss, cc]]).T
                    base += np.array([x, y])
                    corners = np.concatenate([np.c_[base, np.full(4, z - h / 2)],
                                              np.c_[base, np.full(4, z + h / 2)]], axis=0)
                    uv = np.c_[corners, np.ones(8)] @ l2i[c].T
                    if np.any(uv[:, 2] <= 0.1):  # 相机后方，跳过
                        continue
                    u, v = uv[:, 0] / uv[:, 2], uv[:, 1] / uv[:, 2]
                    for i, j in [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
                                 (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]:
                        axc.plot([u[i], u[j]], [v[i], v[j]], color=color, lw=1)
            axc.set_xlim(0, width)
            axc.set_ylim(height, 0)
            axc.set_title(CAM_NAMES[c])
            axc.axis("off")

        axes[2][1].axis("off")
        axl = axes[2][2]
        axl.axis("off")
        for arr, color, label in sets:
            axl.plot([], [], color=color, lw=2, label=f"{label} ({len(arr)})")
        axl.legend(loc="center", fontsize=14)

        plt.savefig(out, dpi=100, bbox_inches="tight")
        print(f"[vis] {out}")

    # ---------------------------------------------------------- report
    @staticmethod
    def _style(ax):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#d5d4cf")
        ax.tick_params(colors=INK2, labelsize=9)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)

    def report(self, bench=None):
        """延时 + 精度对比图 -> report.png。"""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if bench is None:
            path = osp.join(self.dump_dir, "benchmark.json")
            bench = json.load(open(path)) if osp.exists(path) else None
        analyzer = AccuracyAnalyzer(self.dump_dir)
        acc = {sub: analyzer.stage_stats(sub) for sub in ("onnx", "trt")}
        boxes = analyzer.box_stats()

        fig = plt.figure(figsize=(13, 9), facecolor=SURFACE, constrained_layout=True)
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.10])
        fig.suptitle("BEVFusion deployment parity: latency and accuracy",
                     color=INK, fontsize=13, x=0.01, ha="left")

        # 1. per-module latency
        ax = fig.add_subplot(gs[0, 0])
        self._style(ax)
        if bench:
            y = np.arange(len(GRAPHS))
            h = 0.26
            for i, b in enumerate(("torch", "onnx", "trt")):
                vals = [bench["backends"].get(b, {}).get(g, 0) for g in GRAPHS]
                ax.barh(y + (i - 1) * h, vals, height=h * 0.92,
                        color=COLORS[b], label=b, zorder=3)
            ax.set_yticks(y, GRAPHS, color=INK, fontsize=9)
            ax.invert_yaxis()
            ax.set_xlabel("latency (ms)", color=INK2, fontsize=9)
            ax.legend(frameon=False, fontsize=8.5, loc="lower right")
            ax.set_title("Per-module latency", color=INK, fontsize=11, loc="left")
        else:
            ax.text(0.5, 0.5, "benchmark.json missing\n(run --backend bench)",
                    ha="center", va="center", color=INK2, transform=ax.transAxes)
            ax.set_axis_off()

        # 2. end-to-end latency
        ax = fig.add_subplot(gs[0, 1])
        self._style(ax)
        ax.grid(axis="x", visible=False)
        ax.grid(axis="y", color=GRID, lw=0.8)
        if bench:
            backends = ("torch", "onnx", "trt")
            vals = [bench["e2e"].get(b, 0) for b in backends]
            bars = ax.bar(backends, vals, color=[COLORS[b] for b in backends],
                          width=0.55, zorder=3)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.1f} ms",
                        ha="center", va="bottom", color=INK, fontsize=9)
            ax.set_ylim(0, max(vals) * 1.25)
            ax.set_ylabel("latency (ms)", color=INK2, fontsize=9)
            ax.set_title("End-to-end (4 modules, 1 frame)", color=INK, fontsize=11, loc="left")
            ax.text(0.0, -0.22, f"iters={bench['iters']}, warmup={bench['warmup']}; "
                                "torch/onnx fp32, trt fp16",
                    transform=ax.transAxes, color=INK2, fontsize=8)
        else:
            ax.set_axis_off()

        # 3/4. accuracy vs torch
        for idx, metric in enumerate(("mean_abs", "max_abs")):
            ax = fig.add_subplot(gs[1, idx])
            self._style(ax)
            stages = list(STAGE_TENSORS)
            y = np.arange(len(stages))
            h = 0.34
            for i, sub in enumerate(("onnx", "trt")):
                vals = [max(acc[sub].get(s, {}).get(metric, 0), 1e-9) for s in stages]
                ax.barh(y + (i - 0.5) * h, vals, height=h * 0.92,
                        color=COLORS[sub], label=sub, zorder=3)
            ax.set_xscale("log")
            ax.set_yticks(y, stages, color=INK, fontsize=9)
            ax.invert_yaxis()
            ax.set_xlabel(f"{metric.replace('_', ' ')} (log)", color=INK2, fontsize=9)
            ax.legend(frameon=False, fontsize=8.5, loc="upper right")
            ax.set_title(("Mean absolute error vs torch" if metric == "mean_abs"
                          else "Max absolute error vs torch"),
                         color=INK, fontsize=11, loc="left")

        # footer: boxes + dumps
        axf = fig.add_subplot(gs[2, :])
        axf.axis("off")
        lines = []
        if boxes:
            parts = []
            for sub in ("torch", "onnx", "trt"):
                if sub in boxes:
                    b = boxes[sub]
                    extra = f", mean nn {b['mean_m'] * 100:.1f}cm" if "mean_m" in b else ""
                    parts.append(f"{sub} {b['count']}{extra}")
            lines.append("boxes (score>=0.12): " + " | ".join(parts))
        lines.append(f"dump: {self.dump_dir}")
        axf.text(0.0, 0.5, "    ".join(lines), color=INK2, fontsize=8,
                 va="center", transform=axf.transAxes)

        out = osp.join(self.dump_dir, "report.png")
        fig.savefig(out, dpi=140, facecolor=SURFACE)
        plt.close(fig)
        print(f"[report] {out}")
