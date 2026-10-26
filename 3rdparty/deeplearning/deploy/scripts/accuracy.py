"""精度分析：各阶段张量与检测框对拍（参考端 torch）。"""

import os.path as osp

import numpy as np

from deploy.scripts.common import STAGE_TENSORS, load_dump, load_tensor

TENSORS = ["camera_feature", "camera_depth_weights", "cam_bev", "middle",
           "score", "rot", "dim", "reg", "height", "vel"]


class AccuracyAnalyzer:

    def __init__(self, dump_dir):
        self.dump_dir = dump_dir

    def _pair(self, name, ref, cur):
        pa = osp.join(self.dump_dir, ref, name + ".npy")
        pb = osp.join(self.dump_dir, cur, name + ".npy")
        if not (osp.exists(pa) and osp.exists(pb)):
            return None
        a, b = np.load(pa).astype(np.float32), np.load(pb).astype(np.float32)
        return (a, b) if a.shape == b.shape else None

    def stage_stats(self, backend, ref="torch"):
        """每个部署阶段 vs 参考端的 max/mean abs。"""
        stats = {}
        for stage, names in STAGE_TENSORS.items():
            max_abs = 0.0
            total, count = 0.0, 0
            for name in names:
                pair = self._pair(name, ref, backend)
                if pair is None:
                    continue
                d = np.abs(pair[0] - pair[1])
                max_abs = max(max_abs, float(d.max()))
                total += float(d.sum())
                count += d.size
            if count:
                stats[stage] = {"max_abs": max_abs, "mean_abs": total / count}
        return stats

    def print_table(self):
        """打印逐张量精度对比表。"""
        for ref, cur, title in [("torch", "onnx", "torch vs onnx"),
                                ("torch", "trt", "torch vs trt")]:
            print(f"== {title} ==")
            for t in TENSORS:
                pair = self._pair(t, ref, cur)
                if pair is None:
                    continue
                d = np.abs(pair[0] - pair[1])
                print(f"  {t}: max_abs={d.max():.4g} mean_abs={d.mean():.4g}")

    def print_cpp(self, cpp_dir):
        """torch vs C++ 全链（C++ 由 BEVFUSION_DUMP_DIR 落盘）。"""
        print("== torch vs C++ 全链 ==")
        for t in ["lidar_bev", "cam_bev", "middle"]:
            try:
                a = load_dump(t, self.dump_dir, "torch").astype(np.float32)
                b = load_tensor(osp.join(cpp_dir, f"trt_{t}")).astype(np.float32)
                d = np.abs(a - b)
                print(f"  {t}: max_abs={d.max():.4g} mean_abs={d.mean():.4g}")
            except FileNotFoundError:
                print(f"  {t}: C++ dump missing (BEVFUSION_DUMP_DIR)")

    def box_stats(self):
        """各后端框数量 + 到 torch 最近框距离（score>=0.12）。"""
        def load(sub):
            path = osp.join(self.dump_dir, sub, "boxes.npy")
            if not osp.exists(path):
                return None
            b = np.load(path)
            if b.shape[1] >= 11:
                return b[:, :3], b[:, 9]
            return b[:, :3], np.load(osp.join(self.dump_dir, sub, "scores.npy"))

        ref = load("torch")
        if ref is None:
            return {}
        rp, rs = ref
        rp = rp[rs > 0.12]
        out = {"torch": {"count": int(len(rp))}}
        for sub in ("onnx", "trt"):
            cur = load(sub)
            if cur is None:
                continue
            p, s = cur
            p = p[s > 0.12]
            if len(p) == 0 or len(rp) == 0:
                out[sub] = {"count": int(len(p))}
                continue
            d = np.linalg.norm(p[:, None, :] - rp[None, :, :], axis=2).min(axis=1)
            out[sub] = {"count": int(len(p)), "mean_m": float(d.mean()),
                        "max_m": float(d.max())}
        return out
