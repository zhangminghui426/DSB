"""TensorRT inference for the CUDA-BEVFusion-style module split.

The 4 exported ONNX graphs (scripts/export.py) run as TRT engines; everything
with a custom CUDA op stays on the torch/CUDA side:
  camera geometry + bev_pool + vtransform downsample (deploy.bev_pool_torch),
  lidar voxelization + SparseEncoder (torch/spconv, weights from the
  checkpoint), box decode + NMS (TransFusionHead.get_bboxes).

Pipeline (det):
  img/points from a val-dataset sample (eval-protocol preprocessing)
  -> camera.backbone.engine   img (1,6,3,256,704)        -> feats (1,6,256,32,88)
  -> camera.vtransform.engine feats + depth              -> depthnet (1,6,D+C,32,88)
  -> torch geometry/bev_pool  (camera_bev_from_depthnet) -> cam_bev (1,80,180,180)
  -> torch voxelize+spconv    (LidarBranchEncoder)       -> lidar_bev (1,256,180,180)
  -> fuser.engine             cam_bev + lidar_bev        -> decoder_feat (1,512,180,180)
  -> head.bbox.engine         decoder_feat               -> 8 pred tensors
  -> TransFusionHead.get_bboxes (decode + NMS)

Usage:
  # 1) export ONNX first (on the GPU box):
  python scripts/export.py CONFIG CKPT --task det --verify
  # 2) build engines (cached) + run one val sample:
  python deploy/trt_infer.py CONFIG CKPT --engine_dir outputs/trt --build
  #    fp16 engines + parity check against the torch model:
  python deploy/trt_infer.py CONFIG CKPT --engine_dir outputs/trt --build --fp16 --verify
  # or via launch scripts:
  launch/export.sh det ckpts/bevfusion-det-bf.pth
  launch/infer_trt.sh det ckpts/bevfusion-det-bf.pth
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

from deploy.trt_common import DET_HEAD_KEYS, graph_io, load_engines


@torch.no_grad()
def run_trt(model, engines, batch, task, stream=None):
    """Full TRT pipeline for one collated batch (B=1)."""
    from deploy.bev_pool_torch import camera_bev_from_depthnet
    from deploy.lidar_branch import LidarBranchEncoder
    from deploy.preprocess import project_depth

    img = batch["img"]
    feats = engines["camera.backbone"].run({"img": img})["feats"]

    if task == "det":
        # sparse lidar depth map (torch) -> dtransform+depthnet (in-graph)
        depth = project_depth(
            batch["points"][0].cpu(),
            batch["lidar2image"][0].cpu(),
            batch["img_aug_matrix"][0].cpu().float(),
            batch["lidar_aug_matrix"][0].cpu().float(),
        ).unsqueeze(0).to("cuda")
        depthnet = engines["camera.vtransform"].run(
            {"feats": feats, "depth": depth}
        )["depthnet"]
    else:
        depthnet = engines["camera.vtransform"].run({"feats": feats})["depthnet"]

    # camera geometry + bev_pool + downsample (torch/CUDA, not in graph)
    cam_bev = camera_bev_from_depthnet(model, batch, depthnet, task)

    # lidar branch (torch/spconv, weights from checkpoint)
    lidar_bev = LidarBranchEncoder(model)([batch["points"][0]], use_fp16=False)

    decoder_feat = engines["fuser"].run(
        {"cam_bev": cam_bev, "lidar_bev": lidar_bev}
    )["decoder_feat"]

    if task == "det":
        preds = engines["head.bbox"].run({"decoder_feat": decoder_feat})
        pred_dict = {k: preds[k] for k in DET_HEAD_KEYS}
        return pred_dict, decoder_feat
    masks = engines["head.map"].run({"decoder_feat": decoder_feat})["masks"]
    return masks, decoder_feat


@torch.no_grad()
def decode_bboxes(model, pred_dict, metas):
    """TransFusionHead decode + NMS -> (boxes, scores, labels)."""
    head = model.heads["object"]
    boxes, scores, labels = head.get_bboxes(([pred_dict],), metas)[0][0]
    return boxes, scores, labels


@torch.no_grad()
def verify_torch(model, batch, decoder_feat, preds, task):
    """Parity check: torch bev_branch decoder_feat vs TRT fuser engine output."""
    b = batch
    torch_feat = model.bev_branch(
        b["img"], b["points"], None, b["camera2ego"], b["lidar2ego"],
        b["lidar2camera"], b["lidar2image"], b["camera_intrinsics"],
        b["camera2lidar"], b["img_aug_matrix"], b["lidar_aug_matrix"],
        b["metas"], depths=None, reverse=True,
    )[0]
    rel = float(
        (decoder_feat - torch_feat).abs().max()
        / max(1e-6, float(torch_feat.abs().max()))
    )
    print(f"[verify] decoder_feat rel diff vs torch: {rel:.3e}")
    if task == "det" and rel < 5e-3:
        # full torch model decode for a second sanity signal
        out = model(**b)[0]
        boxes_t = out["boxes_3d"].tensor[:5]
        print(f"[verify] torch boxes[:5]:\n{boxes_t.cpu().numpy()}")
    return rel < 5e-3


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config")
    ap.add_argument("checkpoint")
    ap.add_argument("--task", default="det", choices=["det", "seg"])
    ap.add_argument("--onnx_dir", default=osp.join("outputs", "onnx"))
    ap.add_argument("--engine_dir", default=osp.join("outputs", "trt"))
    ap.add_argument("--build", action="store_true", help="build engines from onnx")
    ap.add_argument("--fp16", action="store_true", help="build with FP16")
    ap.add_argument("--autocast-fp16", action="store_true",
                    help="strong typing: ModelOpt-AutoCast ONNX to mixed FP16 "
                         "before building (requires nvidia-modelopt)")
    ap.add_argument("--sample-index", type=int, default=0)
    ap.add_argument("--verify", action="store_true",
                    help="compare decoder_feat against the torch model")
    ap.add_argument("--dump", default=None,
                    help="save decoded output to npz (det: boxes/scores/labels)")
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
    sample = ds[args.sample_index]
    batch = to_device(collate_fn([sample]), "cuda")

    preds, decoder_feat = run_trt(model, engines, batch, args.task)
    print(f"[trt] decoder_feat {tuple(decoder_feat.shape)}")

    if args.task == "det":
        boxes, scores, labels = decode_bboxes(model, preds, batch["metas"])
        print(f"[trt] decoded {len(scores)} boxes "
              f"(top scores: {scores[:5].tolist()})")
        print(f"[trt] boxes[:5]:\n{boxes.tensor[:5].cpu().numpy()}")
        out = {"boxes": boxes.tensor.cpu().numpy(),
               "scores": scores.cpu().numpy(),
               "labels": labels.cpu().numpy()}
    else:
        print(f"[trt] masks {tuple(preds.shape)} (sigmoid in-graph)")
        out = {"masks": preds.cpu().numpy()}

    if args.dump:
        os.makedirs(osp.dirname(osp.abspath(args.dump)), exist_ok=True)
        np.savez(args.dump, **out)
        print(f"[trt] dumped -> {args.dump}")

    if args.verify:
        ok = verify_torch(model, batch, decoder_feat, preds, args.task)
        assert ok, "TRT/torch parity failed"


if __name__ == "__main__":
    main()
