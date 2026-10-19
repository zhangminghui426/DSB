"""Build TensorRT engines from the exported ONNX graphs.

Usage:
  python deploy/trt_build.py --task det --onnx_dir outputs/onnx \
      --engine_dir outputs/trt [--fp16] [--autocast-fp16]

ONNX -> engine conversion uses the trtexec CLI only (lookup: $TRTEXEC env,
then PATH — same convention as launch/perf_trt.sh). A missing trtexec is a
hard error: set TRTEXEC=/path/to/bin/trtexec (e.g. a TensorRT tar package).

Notes:
  - Strong typing (TRT >= 10.12 / 11.x): precision comes from the ONNX; an
    FP32 ONNX builds an FP32 engine. Use --autocast-fp16 (needs
    nvidia-modelopt) for a mixed-FP16 engine.
  - Weak typing (TRT 10.9 docker target): --fp16 sets BuilderFlag.FP16.
"""

import argparse
import os.path as osp

from deploy.trt_common import build_engine, graph_io


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

    graphs = args.graphs.split(",") if args.graphs else list(graph_io(args.task))
    for name in graphs:
        onnx_path = osp.join(args.onnx_dir, f"{name}.onnx")
        assert osp.exists(onnx_path), (
            f"missing {onnx_path} — run scripts/export.py first"
        )
        engine_path = osp.join(args.engine_dir, f"{name}.engine")
        build_engine(onnx_path, engine_path, fp16=args.fp16,
                     autocast_fp16=args.autocast_fp16)


if __name__ == "__main__":
    main()
