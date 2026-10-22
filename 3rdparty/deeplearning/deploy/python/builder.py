"""ONNX -> TensorRT engine builder.

Runs the trtexec CLI only — no Python builder API. The trtexec binary is
resolved from $TRTEXEC (env) or PATH; a missing binary is a hard error.

Usage (run from the model repo root 3rdparty/deeplearning):
  python deploy/python/builder.py --task det --onnx_dir outputs/onnx \
      --engine_dir outputs/trt [--graphs a,b] [--fp16] [--autocast-fp16]
"""

import argparse
import os
import os.path as osp
import shlex
import shutil
import subprocess
import sys
import tempfile

from deploy.python.config import DeployConfig
from deploy.python.graphs import GRAPH_IO


class EngineBuilder:
    """trtexec-based ONNX -> engine conversion (static shapes)."""

    def __init__(self, exe=None):
        self.exe = exe or self.find_trtexec()

    # ------------------------------------------------------------- discovery
    @staticmethod
    def find_trtexec():
        """trtexec lookup: $TRTEXEC env, then PATH. None when unavailable."""
        exe = os.environ.get("TRTEXEC")
        if exe:
            return exe
        return shutil.which("trtexec")

    # --------------------------------------------------------------- helpers
    def autocast_fp16(self, onnx_path, out_path):
        """ModelOpt AutoCast: FP32 ONNX -> mixed-FP16 ONNX (explicit Casts).

        Returns True on success, False if nvidia-modelopt is not installed.
        """
        try:
            import modelopt  # noqa: F401  (import check only)
        except ImportError:
            return False
        cmd = [sys.executable, "-m", "modelopt.onnx.autocast",
               "--onnx_path", onnx_path, "--output_path", out_path,
               "--low_precision_type", "fp16"]
        subprocess.run(cmd, check=True, capture_output=True)
        return True

    def _supports_fp16_flag(self):
        """Probe trtexec --help for the legacy --fp16 flag.

        TRT 10.x trtexec still accepts --fp16; TRT 11 removed it (strong
        typing, precision comes from the ONNX graph). Probe failure falls
        back to NOT adding --fp16 (safe: FP32 build, never a hard error).
        """
        if self.exe is None:
            return False
        try:
            proc = subprocess.run([self.exe, "--help"], capture_output=True,
                                  text=True, timeout=60)
            text = (proc.stdout or "") + (proc.stderr or "")
            return "--fp16" in text
        except (OSError, subprocess.TimeoutExpired):
            return False

    # ---------------------------------------------------------------- build
    def build(self, onnx_path, engine_path, fp16=False, autocast_fp16=False):
        """Convert one ONNX graph to an engine via the trtexec shell cmd."""
        if self.exe is None:
            raise RuntimeError(
                "trtexec not found (set TRTEXEC=/path/to/bin/trtexec, e.g. "
                "the TensorRT tar package's bin/trtexec)")

        os.makedirs(osp.dirname(osp.abspath(engine_path)), exist_ok=True)
        legacy_fp16 = self._supports_fp16_flag()
        src = onnx_path
        tmp_dir = None
        if fp16 and not legacy_fp16:
            if autocast_fp16:
                tmp_dir = tempfile.mkdtemp(prefix="trt_autocast_")
                conv = osp.join(tmp_dir, osp.basename(onnx_path)
                                .replace(".onnx", ".fp16.onnx"))
                if self.autocast_fp16(onnx_path, conv):
                    src = conv
                    print(f"[trt] AutoCast FP16 ONNX -> {conv}")
                else:
                    print("[trt] WARNING: nvidia-modelopt not installed; "
                          "cannot AutoCast to FP16. `pip install "
                          "nvidia-modelopt` or drop --fp16 to build FP32 "
                          "from the FP32 ONNX.")
            else:
                print("[trt] strong typing: precision comes from the ONNX "
                      "graph (FP32 ONNX -> FP32 engine). Use "
                      "--autocast-fp16 for a mixed-FP16 engine "
                      "(requires nvidia-modelopt).")

        # Invoke trtexec as a shell command line (the standard form).
        cmd = (f"{shlex.quote(self.exe)} "
               f"--onnx={shlex.quote(osp.abspath(src))} "
               f"--saveEngine={shlex.quote(osp.abspath(engine_path))}")
        if fp16 and legacy_fp16:
            cmd += " --fp16"
        print("[trt] " + cmd)
        # Scratch dir so trtexec's extra output files never pollute CWD.
        workdir = tempfile.mkdtemp(prefix="trt_engine_")
        try:
            proc = subprocess.run(cmd, cwd=workdir, shell=True, text=True,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
        if proc.returncode != 0:
            tail = "\n".join(proc.stdout.splitlines()[-40:])
            raise RuntimeError(f"trtexec failed for {src}\n--- tail ---\n{tail}")
        print(f"[trt] built {engine_path} (trtexec, fp16={fp16}, "
              f"legacy_fp16={legacy_fp16})")

    def build_graph(self, name, onnx_dir, engine_dir, fp16=False,
                    autocast_fp16=False):
        onnx_path = osp.join(onnx_dir, f"{name}.onnx")
        assert osp.exists(onnx_path), (
            f"missing {onnx_path} — run deploy/export.py first")
        self.build(onnx_path, osp.join(engine_dir, f"{name}.engine"),
                   fp16=fp16, autocast_fp16=autocast_fp16)

    def build_all(self, cfg):
        """Build every graph of cfg.task (or a comma-separated cfg.graphs)."""
        names = (cfg.graphs.split(",") if cfg.graphs
                 else list(GRAPH_IO[cfg.task]))
        for name in names:
            self.build_graph(name, cfg.onnx_dir, cfg.engine_dir,
                             fp16=cfg.fp16,
                             autocast_fp16=cfg.autocast_fp16)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="det", choices=["det", "seg"])
    ap.add_argument("--onnx_dir", default=osp.join("outputs", "onnx"))
    ap.add_argument("--engine_dir", default=osp.join("outputs", "trt"))
    ap.add_argument("--graphs", default=None,
                    help="comma-separated graph names; default: all for --task")
    ap.add_argument("--fp16", action="store_true", help="build with FP16")
    ap.add_argument("--autocast-fp16", action="store_true",
                    help="strong typing: ModelOpt-AutoCast the ONNX to mixed "
                         "FP16 before building (requires nvidia-modelopt)")
    args = ap.parse_args()

    cfg = DeployConfig(task=args.task, onnx_dir=args.onnx_dir,
                       engine_dir=args.engine_dir, fp16=args.fp16,
                       autocast_fp16=args.autocast_fp16,
                       graphs=args.graphs or "")
    EngineBuilder().build_all(cfg)


if __name__ == "__main__":
    main()
