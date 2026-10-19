"""TensorRT build/runtime helpers shared by deploy/trt_build.py and
deploy/trt_infer.py.

ONNX -> engine conversion runs the trtexec CLI only (no Python builder API):
  trtexec --onnx=<onnx> --saveEngine=<engine> [--fp16]

Strong-typing migration (NVIDIA trt-strong-typing-migration):
  - TRT >= 10.12 (incl. 11.x): networks are strongly typed by default;
    precision-hint flags (BuilderFlag.FP16/INT8) are REMOVED — precision must
    live in the ONNX graph (explicit Cast / QDQ nodes). An FP32 ONNX yields an
    FP32 engine; for a mixed-FP16 engine run ModelOpt AutoCast first (or pass
    --autocast-fp16 to deploy/trt_build.py / deploy/trt_infer.py).
  - TRT < 10.12 (e.g. docker target 10.9.0.34): weak typing; trtexec --fp16
    still applies FP16.
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


def _find_trtexec():
    """trtexec lookup: $TRTEXEC env, then PATH. None when unavailable."""
    exe = os.environ.get("TRTEXEC")
    if exe:
        return exe
    return shutil.which("trtexec")


_TRTEXEC_STRONG_CACHE = {}


def _trtexec_supports_strongly_typed(exe):
    """Probe `trtexec --help` for --stronglyTyped (TRT >= 10.12).

    On any probe failure we conservatively assume strong typing: adding
    --fp16 to an 11.x trtexec fails outright, whereas omitting it on 10.x
    only yields an FP32 engine (never a hard error).
    """
    if exe in _TRTEXEC_STRONG_CACHE:
        return _TRTEXEC_STRONG_CACHE[exe]
    try:
        proc = subprocess.run([exe, "--help"], capture_output=True,
                              text=True, timeout=60)
        text = (proc.stdout or "") + (proc.stderr or "")
        strong = "stronglyTyped" in text
    except (OSError, subprocess.TimeoutExpired):
        strong = True
    _TRTEXEC_STRONG_CACHE[exe] = strong
    return strong


def _build_engine_trtexec(exe, onnx_path, engine_path, fp16, autocast_fp16):
    """ONNX -> engine via trtexec CLI (the standard conversion path)."""
    os.makedirs(osp.dirname(osp.abspath(engine_path)), exist_ok=True)
    strong = _trtexec_supports_strongly_typed(exe)
    src = onnx_path
    tmp_dir = None
    if fp16 and strong:
        if autocast_fp16:
            tmp_dir = tempfile.mkdtemp(prefix="trt_autocast_")
            conv = osp.join(tmp_dir, osp.basename(onnx_path)
                            .replace(".onnx", ".fp16.onnx"))
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

    cmd = [exe, f"--onnx={osp.abspath(src)}",
           f"--saveEngine={osp.abspath(engine_path)}"]
    if fp16 and not strong:
        cmd.append("--fp16")
    print("[trt] " + " ".join(cmd))
    # Run in a scratch dir so trtexec's extra output files never pollute CWD.
    workdir = tempfile.mkdtemp(prefix="trt_engine_")
    try:
        proc = subprocess.run(cmd, cwd=workdir, text=True,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    if proc.returncode != 0:
        tail = "\n".join(proc.stdout.splitlines()[-40:])
        raise RuntimeError(f"trtexec failed for {src}\n--- tail ---\n{tail}")
    print(f"[trt] built {engine_path} (trtexec, fp16={fp16}, strong={strong})")


def build_engine(onnx_path, engine_path, fp16=False, autocast_fp16=False):
    """ONNX -> TensorRT engine (static shapes) via the trtexec CLI only.

    trtexec is resolved from $TRTEXEC or PATH; a missing binary is a hard
    error (set TRTEXEC=/path/to/bin/trtexec, e.g. a TensorRT tar package).
    fp16 semantics by trtexec generation:
      - TRT >= 10.12 strong typing: --fp16 is removed; precision comes from
        the ONNX (--autocast-fp16 ModelOpt-AutoCasts first, needs
        nvidia-modelopt; otherwise FP32 ONNX -> FP32 engine).
      - TRT < 10.12 (10.9 docker target): --fp16 applies weak-typing FP16.
    """
    exe = _find_trtexec()
    if exe is None:
        raise RuntimeError(
            "trtexec not found (set TRTEXEC=/path/to/bin/trtexec, e.g. the "
            "TensorRT tar package's bin/trtexec)"
        )
    _build_engine_trtexec(exe, onnx_path, engine_path, fp16, autocast_fp16)


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
