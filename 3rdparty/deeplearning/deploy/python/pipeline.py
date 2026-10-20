"""TRT 推理编排入口：Preprocessor -> Inferencer -> Postprocessor。

导出 ONNX 图（deploy.python.export）以 TRT engine 运行（deploy.python.runtime）；
自定义 CUDA 算子留在 torch/CUDA 侧（几何 + bev_pool + 降采样、
lidar 体素化 + 稀疏编码、检测解码 + NMS）。

链路（det）：
  img/points（val 样本）
  -> [Preprocessor] 深度投影 + 体素化
  -> [Inferencer] camera.backbone -> camera.vtransform -> bev_pool ->
     fuser -> head.bbox
  -> [Postprocessor] 解码 + NMS -> boxes / scores / labels

Usage (run from the model repo root 3rdparty/deeplearning):
  python deploy/python/pipeline.py CONFIG CKPT --task det --engine_dir outputs/trt \
      [--build] [--fp16] [--verify] [--sample-index N] [--dump out.npz]
"""

import argparse
import os
import os.path as osp
import sys

# deeplearning 根（上 3 级）：models/datasets/scripts 与 deploy.python 包都在其下
_DL_ROOT = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.insert(0, _DL_ROOT)

import numpy as np
import torch

from datasets import build_dataset, collate_fn
from models import build_model
from scripts.config import load_config
from scripts.trainer import load_model_checkpoint, to_device

from deploy.python.inference import Inferencer
from deploy.python.postprocess import Postprocessor
from deploy.python.preprocess import Preprocessor
from deploy.python.runtime import load_engines


class TrtPipeline:
    """三层编排：预处理 -> 推理 -> 后处理（+ verify / dump）。"""

    def __init__(self, preprocessor, inferencer, postprocessor, task):
        self.preprocessor = preprocessor
        self.inferencer = inferencer
        self.postprocessor = postprocessor
        self.task = task

    @torch.no_grad()
    def run(self, batch, dump="", verify=False):
        """对单个 collated batch（B=1）执行完整推理链路并返回结果。"""
        pre = self.preprocessor(batch)
        preds, decoder_feat = self.inferencer.infer(batch, pre)
        print(f"[trt] decoder_feat {tuple(decoder_feat.shape)}")

        if self.task == "det":
            boxes, scores, labels = self.postprocessor.decode(
                preds, batch["metas"]
            )
            print(f"[trt] decoded {len(scores)} boxes "
                  f"(top scores: {scores[:5].tolist()})")
            print(f"[trt] boxes[:5]:\n{boxes.tensor[:5].cpu().numpy()}")
            out = {
                "boxes": boxes.tensor.cpu().numpy(),
                "scores": scores.cpu().numpy(),
                "labels": labels.cpu().numpy(),
            }
        else:
            print(f"[trt] masks {tuple(preds.shape)} (sigmoid in-graph)")
            out = {"masks": preds.cpu().numpy()}

        if dump:
            os.makedirs(osp.dirname(osp.abspath(dump)), exist_ok=True)
            np.savez(dump, **out)
            print(f"[trt] dumped -> {dump}")

        if verify:
            ok = self.inferencer.verify(batch, decoder_feat)
            assert ok, "TRT/torch parity failed"
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config")
    ap.add_argument("checkpoint")
    ap.add_argument("--task", default="det", choices=["det", "seg"])
    ap.add_argument("--onnx_dir", default=osp.join("outputs", "onnx"))
    ap.add_argument("--engine_dir", default=osp.join("outputs", "trt"))
    ap.add_argument("--build", action="store_true",
                    help="build engines from onnx before running")
    ap.add_argument("--fp16", action="store_true", help="build with FP16")
    ap.add_argument("--autocast-fp16", action="store_true",
                    help="strong typing: ModelOpt-AutoCast ONNX to mixed FP16 "
                         "before building (requires nvidia-modelopt)")
    ap.add_argument("--sample-index", type=int, default=0)
    ap.add_argument("--verify", action="store_true",
                    help="compare decoder_feat against the torch model")
    ap.add_argument("--dump", default=None,
                    help="save decoded output to npz "
                         "(det: boxes/scores/labels)")
    ap.add_argument("--options", nargs="*", default=[])
    args = ap.parse_args()

    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    cfg = load_config(args.config, options=args.options)
    model = build_model(cfg.model).cuda().eval()
    load_model_checkpoint(model, args.checkpoint, strict=True)

    engines = load_engines(args.task, args.onnx_dir, args.engine_dir,
                           build=args.build, fp16=args.fp16,
                           autocast_fp16=args.autocast_fp16)

    ds = build_dataset(cfg.data.val)
    batch = to_device(collate_fn([ds[args.sample_index]]), "cuda")

    TrtPipeline(
        Preprocessor(model),
        Inferencer(model, engines, args.task),
        Postprocessor(model),
        args.task,
    ).run(batch, dump=args.dump or "", verify=args.verify)


if __name__ == "__main__":
    main()
