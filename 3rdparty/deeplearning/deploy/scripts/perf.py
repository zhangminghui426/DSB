"""性能分析：torch / onnx / trt 逐模块延时（ms）。"""

import json
import os.path as osp
import time

import numpy as np
import torch

from deploy.scripts.backends import bind_trt, trt_plan_path, trt_session
from deploy.scripts.common import FEEDS, GRAPHS, load_dump
from deploy.scripts.export import (
    CamBackboneBf, DownsampleBf, FuserDecoder, HeadBboxBf)


class PerfBenchmark:
    """4 个部署模块的 warmup + 多次迭代延时，结果写 benchmark.json。"""

    def __init__(self, onnx_dir, engine_dir, dump_dir, iters=20, warmup=5, head_plan=None):
        self.onnx_dir = onnx_dir
        self.engine_dir = engine_dir
        self.dump_dir = dump_dir
        self.iters = iters
        self.warmup = warmup
        self.head_plan = head_plan

    @staticmethod
    def _time_cuda(fn, warmup, iters):
        with torch.no_grad():
            for _ in range(warmup):
                fn()
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                fn()
            end.record()
            torch.cuda.synchronize()
        return start.elapsed_time(end) / iters

    def torch(self, model):
        tensors = {k: torch.from_numpy(load_dump(k, self.dump_dir, "torch")).cuda()
                   for k in ("img", "depth", "bev360", "cam_bev", "lidar_bev", "middle")}
        cam, down = CamBackboneBf(model), DownsampleBf(model)
        fuser, head = FuserDecoder(model), HeadBboxBf(model)
        calls = {
            "camera.backbone": lambda: cam(tensors["img"], tensors["depth"]),
            "camera.vtransform": lambda: down(tensors["bev360"]),
            "fuser": lambda: fuser(tensors["cam_bev"], tensors["lidar_bev"]),
            "head.bbox": lambda: head(tensors["middle"]),
        }
        return {g: self._time_cuda(calls[g], self.warmup, self.iters) for g in GRAPHS}

    def onnx(self):
        import onnxruntime as ort

        result = {}
        for graph in GRAPHS:
            path = osp.join(self.onnx_dir, graph + ".onnx")
            if not osp.exists(path):
                continue
            sess = ort.InferenceSession(path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            feed = {k: np.ascontiguousarray(load_dump(v, self.dump_dir).astype(np.float32))
                    for k, v in FEEDS[graph].items()}
            for _ in range(self.warmup):
                sess.run(None, feed)
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(self.iters):
                sess.run(None, feed)
            torch.cuda.synchronize()
            result[graph] = (time.perf_counter() - start) * 1000.0 / self.iters
        return result

    def trt(self):
        stream = torch.cuda.current_stream().cuda_stream
        result = {}
        for graph in GRAPHS:
            path = trt_plan_path(self.engine_dir, graph, self.head_plan)
            if not osp.exists(path):
                continue
            ctx, in_names, out_names = trt_session(self.engine_dir, path)
            bind_trt(ctx, in_names, out_names, graph, self.dump_dir)
            run = lambda: ctx.execute_async_v3(stream)
            result[graph] = self._time_cuda(run, self.warmup, self.iters)
        return result

    def run(self, model):
        data = {"iters": self.iters, "warmup": self.warmup, "graphs": GRAPHS, "backends": {}}
        data["backends"]["torch"] = self.torch(model)
        data["backends"]["onnx"] = self.onnx()
        data["backends"]["trt"] = self.trt()
        data["e2e"] = {k: sum(v.values()) for k, v in data["backends"].items() if v}

        with open(osp.join(self.dump_dir, "benchmark.json"), "w") as f:
            json.dump(data, f, indent=2)

        print(f"\n[bench] ms (iters={self.iters}, warmup={self.warmup})")
        print(f"{'module':<18}{'torch':>10}{'onnx':>10}{'trt':>10}")
        for g in GRAPHS:
            row = [data["backends"][b].get(g) for b in ("torch", "onnx", "trt")]
            print(f"{g:<18}" + "".join(f"{v:>10.2f}" if v else f"{'-':>10}" for v in row))
        print(f"{'end2end':<18}" + "".join(
            f"{data['e2e'].get(b, 0):>10.2f}" for b in ("torch", "onnx", "trt")))
        return data
