"""bf TensorRT runtime：加载 4 个 .plan（CUDA-BEVFusion 结构）并执行。

与 deploy/cpp/src/bevfusion/ 一致：engine 由 build_trt_engine.sh 构建
（--fp16 + IO fp16:chw），输入输出均为 CUDA half 张量。
"""
import os

import numpy as np
import tensorrt as trt
import torch

# 绑定名（与 main.cpp create_core 一致）
ENGINE_NAMES = ["camera.backbone", "camera.vtransform", "fuser", "head.bbox"]

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class TrtEngine:
    """单个 .plan 的加载与执行（零拷贝：输入/输出直接是 torch CUDA 张量）。"""

    def __init__(self, plan_path: str):
        self.plan_path = plan_path
        with open(plan_path, "rb") as f:
            runtime = trt.Runtime(TRT_LOGGER)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        assert self.engine is not None, f"failed to deserialize: {plan_path}"
        self.context = self.engine.create_execution_context()

        self.inputs = {}
        self.outputs = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            shape = self.engine.get_tensor_shape(name)  # 固定形状
            dtype = self.engine.get_tensor_dtype(name)
            torch_dtype = torch.float16 if dtype == trt.float16 else torch.float32
            if mode == trt.TensorIOMode.INPUT:
                self.inputs[name] = dict(shape=tuple(shape), dtype=torch_dtype)
            else:
                self.outputs[name] = dict(shape=tuple(shape), dtype=torch_dtype)
        print(f"[bf/trt] loaded {os.path.basename(plan_path)}  "
              f"in={ {k: v['shape'] for k, v in self.inputs.items()} }  "
              f"out={ {k: v['shape'] for k, v in self.outputs.items()} }")

    def run(self, feeds: dict) -> dict:
        """feeds: {输入绑定名: torch CUDA 张量} -> {输出绑定名: torch CUDA 张量}。"""
        buffers = {}
        for name, info in self.inputs.items():
            t = feeds[name]
            assert t.is_cuda and t.dtype == info["dtype"], f"{name}: need {info['dtype']} cuda got {t.dtype}"
            buffers[name] = t
        for name, info in self.outputs.items():
            buffers[name] = torch.empty(info["shape"], dtype=info["dtype"], device="cuda")

        stream = torch.cuda.current_stream().cuda_stream
        for name, t in buffers.items():
            self.context.set_tensor_address(name, t.data_ptr())
        # TRT 10.x 用 execute_async_v3（配合 set_tensor_address），旧版回退 v2
        if hasattr(self.context, "execute_async_v3"):
            ok = self.context.execute_async_v3(stream)
        else:
            ok = self.context.execute_async_v2(stream)
        assert ok, f"engine execute failed: {self.plan_path}"
        return {name: buffers[name] for name in self.outputs}


def load_engines(model_dir: str) -> dict:
    """加载 model/<name>/build/ 下 4 个 .plan，返回 {绑定名: TrtEngine}。"""
    engines = {}
    for name in ENGINE_NAMES:
        path = os.path.join(model_dir, name, "build", f"{name}.plan")
        path2 = os.path.join(model_dir, "build", f"{name}.plan")
        p = path if os.path.exists(path) else path2
        assert os.path.exists(p), f"missing engine: {p}（先执行 deploy/build_trt_engine.sh）"
        engines[name] = TrtEngine(p)
    return engines
