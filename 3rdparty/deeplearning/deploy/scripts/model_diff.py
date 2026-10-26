"""torch / onnxruntime / TensorRT 三端对比入口：推理、性能、精度、可视化。

用法（3rdparty/deeplearning 下）：
  python deploy/scripts/model_diff.py configs/det_convfuser.py ckpts/bevfusion-det-bf.pth \
      --backend all --dump-dir outputs/model_diff

  --backend torch    训练接口全链，落盘各模块张量/框
  --backend onnx     onnxruntime 跑 4 张图（输入用 torch dump）
  --backend trt      TensorRT 跑 4 个 plan
  --backend bench    逐模块延时 -> benchmark.json
  --backend diff     精度对比表
  --backend trt-full torch vs C++ 全链（需 BEVFUSION_DUMP_DIR）
  --backend vis      框叠加图
  --backend report   延时/精度报告 -> report.png
"""

import argparse
import os.path as osp
import sys

_HERE = osp.dirname(osp.abspath(__file__))
_DL_ROOT = osp.dirname(osp.dirname(_HERE))  # 3rdparty/deeplearning
for _path in (_DL_ROOT, _HERE):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from deploy.scripts.accuracy import AccuracyAnalyzer  # noqa: E402
from deploy.scripts.backends import (  # noqa: E402
    OnnxBackend, TorchBackend, TrtBackend, load_model_only)
from deploy.scripts.perf import PerfBenchmark  # noqa: E402
from deploy.scripts.vis import Visualizer  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?")
    ap.add_argument("ckpt", nargs="?")
    ap.add_argument("--backend", default="all",
                    choices=["torch", "onnx", "trt", "bench", "diff",
                             "trt-full", "vis", "report", "all"])
    ap.add_argument("--onnx-dir", default="outputs/onnx")
    ap.add_argument("--engine-dir", default="deploy/model/resnet50/build")
    ap.add_argument("--dump-dir", default="outputs/model_diff")
    ap.add_argument("--sample-index", type=int, default=0)
    ap.add_argument("--data-dir", default="outputs/example-data")
    ap.add_argument("--cpp-boxes", default=None)
    ap.add_argument("--head-plan", default=None)
    ap.add_argument("--cpp-dir", default="outputs/model_diff/cpp")
    ap.add_argument("--bench-iters", type=int, default=20)
    ap.add_argument("--bench-warmup", type=int, default=5)
    args = ap.parse_args()

    need_model = args.backend in ("torch", "onnx", "trt", "bench", "all")
    if need_model:
        assert args.config and args.ckpt, f"{args.backend} 需要 config + ckpt"

    model = None
    if args.backend in ("torch", "all"):
        model = TorchBackend(args.config, args.ckpt, args.dump_dir,
                             args.sample_index, args.cpp_dir).run()
    elif need_model:
        model = load_model_only(args.config, args.ckpt)

    if args.backend in ("onnx", "all"):
        OnnxBackend(args.onnx_dir, args.dump_dir).run(model)
    if args.backend in ("trt", "all"):
        TrtBackend(args.engine_dir, args.dump_dir, args.head_plan).run(model)

    if args.backend in ("bench", "all"):
        PerfBenchmark(args.onnx_dir, args.engine_dir, args.dump_dir,
                      args.bench_iters, args.bench_warmup, args.head_plan).run(model)

    analyzer = AccuracyAnalyzer(args.dump_dir)
    if args.backend in ("diff", "all"):
        analyzer.print_table()
    if args.backend in ("trt-full", "all"):
        analyzer.print_cpp(args.cpp_dir)

    visualizer = Visualizer(args.dump_dir, args.data_dir)
    if args.backend in ("vis", "all"):
        visualizer.boxes(args.cpp_boxes)
    if args.backend in ("report", "all"):
        visualizer.report()


if __name__ == "__main__":
    main()
