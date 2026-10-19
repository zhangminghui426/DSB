"""TensorRT build/runtime helpers shared by deploy/trt_build.py and
deploy/trt_infer.py.

Strong-typing migration (NVIDIA trt-strong-typing-migration):
  - TRT >= 10.12 (incl. 11.x): networks are created strongly typed
    (STRONGLY_TYPED flag; accepted-but-ignored on 11.x) and precision-hint
    flags (BuilderFlag.FP16/INT8) are REMOVED — precision must live in the
    ONNX graph (explicit Cast / QDQ nodes). An FP32 ONNX yields an FP32
    engine; for a mixed-FP16 engine run ModelOpt AutoCast first (or pass
    --autocast-fp16 to deploy/trt_build.py / deploy/trt_infer.py).
  - TRT < 10.12 (e.g. docker target 10.9.0.34): strong typing is unavailable,
    so fall back to explicit-batch weak typing where BuilderFlag.FP16 still
    works. Version is probed at build time; no source edits needed per env.
"""

import os
import os.path as osp
import shutil
import subprocess
import sys
import tempfile

import torch

DET_HEAD_KEYS = [
    "dense_heatmap",
    "heatmap",
    "query_heatmap_score",
    "center",
    "height",
    "dim",
    "rot",
    "vel",
]


def graph_io(task):
    """Graph name -> (input names, output names); mirrors scripts/export.py."""
    if task == "det":
        return {
            "camera.backbone": (["img"], ["feats"]),
            "camera.vtransform": (["feats", "depth"], ["depthnet"]),
            "fuser": (["cam_bev", "lidar_bev"], ["decoder_feat"]),
            "head.bbox": (["decoder_feat"], DET_HEAD_KEYS),
        }
    return {
        "camera.backbone": (["img"], ["feats"]),
        "camera.vtransform": (["feats"], ["depthnet"]),
        "fuser": (["cam_bev", "lidar_bev"], ["decoder_feat"]),
        "head.map": (["decoder_feat"], ["masks"]),
    }


def _network_flags(trt):
    """Return (create_network flags, typing mode 'strong' | 'weak')."""
    try:
        major, minor = (int(x) for x in str(trt.__version__).split(".")[:2])
    except (ValueError, AttributeError):
        major, minor = 0, 0
    strong = getattr(trt.NetworkDefinitionCreationFlag, "STRONGLY_TYPED", None)
    if strong is not None and (major, minor) >= (10, 12):
        return 1 << int(strong), "strong"
    # TRT 8-10.11: explicit batch is the opt-in form of weak typing.
    explicit = getattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH", None)
    if explicit is not None:
        return 1 << int(explicit), "weak"
    return 0, "weak"


def _autocast_onnx_fp16(onnx_path, out_path):
    """ModelOpt AutoCast: FP32 ONNX -> mixed-FP16 ONNX (explicit Cast nodes).

    Returns True on success, False if nvidia-modelopt is not installed.
    """
    try:
        import modelopt  # noqa: F401  (import check only)
    except ImportError:
        return False
    cmd = [
        sys.executable, "-m", "modelopt.onnx.autocast",
        "--onnx_path", onnx_path,
        "--output_path", out_path,
        "--low_precision_type", "fp16",
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return True


def build_engine(onnx_path, engine_path, fp16=False, autocast_fp16=False,
                 workspace=2 << 30):
    """ONNX -> TensorRT engine (static shapes).

    fp16 is honored differently by the two typing modes:
      weak  (TRT < 10.12): BuilderFlag.FP16 (probed; absent -> FP32 fallback).
      strong (TRT >= 10.12): the flag no longer exists; with autocast_fp16 the
      ONNX is AutoCast'd to mixed FP16 first, otherwise the ONNX's own
      precision is built as-is (FP32 ONNX -> FP32 engine).
    """
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags, mode = _network_flags(trt)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)

    src = onnx_path
    tmp_dir = None
    if fp16 and mode == "strong":
        if autocast_fp16:
            tmp_dir = tempfile.mkdtemp(prefix="trt_autocast_")
            conv = osp.join(tmp_dir, osp.basename(onnx_path).replace(".onnx", ".fp16.onnx"))
            if _autocast_onnx_fp16(onnx_path, conv):
                src = conv
                print(f"[trt] AutoCast FP16 ONNX -> {conv}")
            else:
                print("[trt] WARNING: nvidia-modelopt not installed; cannot "
                      "AutoCast to FP16. `pip install nvidia-modelopt` or drop "
                      "--fp16 to build FP32 from the FP32 ONNX.")
        else:
            print("[trt] strong typing: precision comes from the ONNX graph "
                  "(FP32 ONNX -> FP32 engine). Use --autocast-fp16 for a "
                  "mixed-FP16 engine (requires nvidia-modelopt).")

    with open(src, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"[trt] onnx error {i}: {parser.get_error(i)}")
            raise RuntimeError(f"ONNX parse failed: {src}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace)
    if fp16 and mode == "weak":
        # Weak typing only: probe both spellings; if absent, FP32 fallback.
        flag = next(
            (getattr(trt.BuilderFlag, n, None) for n in ("FP16", "HALF")
             if getattr(trt.BuilderFlag, n, None) is not None),
            None,
        )
        if flag is not None:
            config.set_flag(flag)
        else:
            print(f"[trt] WARNING: BuilderFlag FP16/HALF unavailable in "
                  f"tensorrt {trt.__version__}; building FP32 engine")

    serialized = builder.build_serialized_network(network, config)
    assert serialized is not None, f"engine build failed: {onnx_path}"
    os.makedirs(osp.dirname(engine_path), exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized)
    print(f"[trt] built {engine_path} (fp16={fp16}, typing={mode})")
    if tmp_dir:
        shutil.rmtree(tmp_dir, ignore_errors=True)


class TRTEngine:
    """Minimal TRT runtime wrapper; I/O buffers allocated with torch
    (no pycuda dependency). Static shapes from the exported graphs."""

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
    engines = {}
    for name, (in_names, out_names) in graph_io(task).items():
        onnx_path = osp.join(onnx_dir, f"{name}.onnx")
        engine_path = osp.join(engine_dir, f"{name}.engine")
        if build or not osp.exists(engine_path):
            assert osp.exists(onnx_path), (
                f"missing {onnx_path} — run scripts/export.py first"
            )
            build_engine(onnx_path, engine_path, fp16=fp16,
                         autocast_fp16=autocast_fp16)
        engines[name] = TRTEngine(engine_path)
        print(f"[trt] loaded {engine_path}")
    return engines
