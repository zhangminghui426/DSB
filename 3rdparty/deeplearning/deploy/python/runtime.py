"""TensorRT runtime: engine deserialization and execution.

The runtime half of the deployment split. Engines are built by
deploy.builder (trtexec CLI); this module loads and runs them with I/O
buffers allocated through torch (no pycuda dependency), static shapes from
the exported graphs (deploy.graphs).
"""

import os.path as osp

import torch

from deploy.python.graphs import GRAPH_IO


class TrtEngine:
    """Minimal TRT runtime wrapper; I/O buffers allocated with torch."""

    def __init__(self, engine_path, stream=None):
        import tensorrt as trt

        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        assert self.engine is not None, f"failed to deserialize: {engine_path}"
        self.context = self.engine.create_execution_context()
        self.stream = stream or torch.cuda.current_stream()

    def run(self, inputs):
        """inputs: {name: torch.Tensor} -> outputs: {name: torch.Tensor}."""
        import tensorrt as trt

        outputs = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                t = inputs[name].contiguous()
                self.context.set_tensor_address(name, t.data_ptr())
            else:
                shape = tuple(self.context.get_tensor_shape(name))
                t = torch.empty(shape, dtype=torch.float32, device="cuda")
                self.context.set_tensor_address(name, t.data_ptr())
                outputs[name] = t
        ok = self.context.execute_async_v3(self.stream.cuda_stream)
        assert ok, "engine execute failed"
        self.stream.synchronize()
        return outputs


def load_engines(task, onnx_dir, engine_dir, build=False, fp16=False,
                 autocast_fp16=False):
    """Load (optionally build) every engine for a task; returns {name: engine}."""
    if build:
        from deploy.python.builder import EngineBuilder
        from deploy.python.config import DeployConfig

        EngineBuilder().build_all(
            DeployConfig(task=task, onnx_dir=onnx_dir, engine_dir=engine_dir,
                         fp16=fp16, autocast_fp16=autocast_fp16))

    engines = {}
    for name in GRAPH_IO[task]:
        engine_path = osp.join(engine_dir, f"{name}.engine")
        if not osp.exists(engine_path):
            raise FileNotFoundError(
                f"missing {engine_path} — run deploy/python/builder.py first "
                f"(or pass --build)")
        engines[name] = TrtEngine(engine_path)
        print(f"[trt] loaded {engine_path}")
    return engines
