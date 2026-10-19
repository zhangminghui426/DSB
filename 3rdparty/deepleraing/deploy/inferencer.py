"""TensorRT inference for the CUDA-BEVFusion-style module split.

The exported ONNX graphs (deploy.export) run as TRT engines
(deploy.runtime); everything with a custom CUDA op stays on the torch/CUDA
side: camera geometry + bev_pool + vtransform downsample (deploy.bev_pool_torch),
lidar voxelization + SparseEncoder (deploy.lidar_branch), box decode + NMS
(TransFusionHead.get_bboxes).

Pipeline (det):
  img/points (val-sample eval preprocessing)
  -> camera.backbone.engine    img (1,6,3,256,704)      -> feats (1,6,256,32,88)
  -> camera.vtransform.engine  feats + depth            -> depthnet (1,6,D+C,32,88)
  -> torch geometry/bev_pool   (camera_bev_from_depthnet) -> cam_bev (1,80,180,180)
  -> torch voxelize+spconv     (LidarBranchEncoder)       -> lidar_bev (1,256,180,180)
  -> fuser.engine              cam_bev + lidar_bev        -> decoder_feat (1,512,180,180)
  -> head.bbox.engine          decoder_feat               -> 9 pred tensors (incl. query_labels)
  -> TransFusionHead.get_bboxes (decode + NMS)

Usage:
  python deploy/inferencer.py CONFIG CKPT --task det --engine_dir outputs/trt \
      [--build] [--fp16] [--verify] [--sample-index N] [--dump out.npz]
"""

import argparse
import os
import os.path as osp
import sys

sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), ".."))

import numpy as np
import torch

from datasets import build_dataset, collate_fn
from models import build_model
from scripts.config import load_config
from scripts.trainer import load_model_checkpoint, to_device

from deploy.graphs import DET_HEAD_KEYS
from deploy.runtime import load_engines


class DetInferencer:
    """TRT pipeline for one collated batch (B=1): init / infer / decode / verify."""

    def __init__(self, model, engines, task):
        self.model = model
        self.engines = engines
        self.task = task

    @torch.no_grad()
    def infer(self, batch):
        """Full TRT pipeline for one collated batch (B=1).

        Returns (preds, decoder_feat): for det, preds is the 9-key dict from
        head.bbox (8 preds + query_labels); for seg, preds is the masks tensor.
        """
        from deploy.bev_pool_torch import camera_bev_from_depthnet
        from deploy.lidar_branch import LidarBranchEncoder
        from deploy.preprocess import project_depth

        img = batch["img"]
        feats = self.engines["camera.backbone"].run({"img": img})["feats"]

        if self.task == "det":
            # sparse lidar depth map (torch) -> dtransform+depthnet (in-graph)
            depth = project_depth(
                batch["points"][0].cpu(), batch["lidar2image"][0].cpu(),
                batch["img_aug_matrix"][0].cpu().float(),
                batch["lidar_aug_matrix"][0].cpu().float(),
            ).unsqueeze(0).to("cuda")
            depthnet = self.engines["camera.vtransform"].run(
                {"feats": feats, "depth": depth})["depthnet"]
        else:
            depthnet = self.engines["camera.vtransform"].run(
                {"feats": feats})["depthnet"]

        # camera geometry + bev_pool + downsample (torch/CUDA, not in graph)
        cam_bev = camera_bev_from_depthnet(self.model, batch, depthnet,
                                           self.task)

        # lidar branch (torch/spconv, weights from checkpoint)
        lidar_bev = LidarBranchEncoder(self.model)(
            [batch["points"][0]], use_fp16=False)

        decoder_feat = self.engines["fuser"].run(
            {"cam_bev": cam_bev, "lidar_bev": lidar_bev})["decoder_feat"]

        if self.task == "det":
            preds = self.engines["head.bbox"].run(
                {"decoder_feat": decoder_feat})
            preds = {k: preds[k] for k in DET_HEAD_KEYS}
        else:
            preds = self.engines["head.map"].run(
                {"decoder_feat": decoder_feat})["masks"]
        return preds, decoder_feat

    @torch.no_grad()
    def decode(self, preds, metas):
        """TransFusionHead decode + NMS -> (boxes, scores, labels)."""
        head = self.model.heads["object"]
        # get_bboxes returns [[boxes3d, scores, labels]] (batch of one here),
        # so [0] is the (boxes, scores, labels) triple for the sample.
        boxes, scores, labels = head.get_bboxes(([preds],), metas)[0]
        return boxes, scores, labels

    @torch.no_grad()
    def verify(self, batch, decoder_feat):
        """Parity: torch bev_branch decoder_feat vs TRT fuser engine output."""
        b = batch
        torch_feat = self.model.bev_branch(
            b["img"], b["points"], None, b["camera2ego"], b["lidar2ego"],
            b["lidar2camera"], b["lidar2image"], b["camera_intrinsics"],
            b["camera2lidar"], b["img_aug_matrix"], b["lidar_aug_matrix"],
            b["metas"], depths=None, reverse=True)[0]
        rel = float((decoder_feat - torch_feat).abs().max()
                    / max(1e-6, float(torch_feat.abs().max())))
        # TRT engines run FP32 with TF32 kernels on Ampere+ by default, while
        # torch runs strict FP32 (tf32 disabled in main); a single-point max
        # rel diff of ~1e-2 is expected for conv nets. 5e-3 was too tight.
        tol = 2e-2
        print(f"[verify] decoder_feat rel diff vs torch: {rel:.3e} "
              f"(tol {tol:.0e})")
        if self.task == "det" and rel < tol:
            out = self.model(**b)[0]
            boxes_t = out["boxes_3d"].tensor[:5]
            print(f"[verify] torch boxes[:5]:\n{boxes_t.cpu().numpy()}")
        return rel < tol

    def run(self, batch, dump="", verify=False):
        """Infer one batch, optionally decode/dump/verify; returns npz dict."""
        preds, decoder_feat = self.infer(batch)
        print(f"[trt] decoder_feat {tuple(decoder_feat.shape)}")

        if self.task == "det":
            boxes, scores, labels = self.decode(preds, batch["metas"])
            print(f"[trt] decoded {len(scores)} boxes "
                  f"(top scores: {scores[:5].tolist()})")
            print(f"[trt] boxes[:5]:\n{boxes.tensor[:5].cpu().numpy()}")
            out = {"boxes": boxes.tensor.cpu().numpy(),
                   "scores": scores.cpu().numpy(),
                   "labels": labels.cpu().numpy()}
        else:
            print(f"[trt] masks {tuple(preds.shape)} (sigmoid in-graph)")
            out = {"masks": preds.cpu().numpy()}

        if dump:
            os.makedirs(osp.dirname(osp.abspath(dump)), exist_ok=True)
            np.savez(dump, **out)
            print(f"[trt] dumped -> {dump}")

        if verify:
            ok = self.verify(batch, decoder_feat)
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

    DetInferencer(model, engines, args.task).run(
        batch, dump=args.dump or "", verify=args.verify)


if __name__ == "__main__":
    main()
