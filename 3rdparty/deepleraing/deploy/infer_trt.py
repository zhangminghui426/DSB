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
  python deploy/infer_trt.py CONFIG CKPT --engine_dir outputs/trt --build
  #    fp16 engines + parity check against the torch model:
  python deploy/infer_trt.py CONFIG CKPT --engine_dir outputs/trt --build --fp16 --verify
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


def build_engine(onnx_path, engine_path, fp16=False, workspace=2 << 30):
    """ONNX -> TensorRT engine (static shapes)."""
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    # TRT 10 removed NetworkDefinitionCreationFlag.EXPLICIT_BATCH (explicit
    # batch is the default since TRT 8); guard for older versions.
    explicit = getattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH", None)
    if explicit is None:
        network = builder.create_network()
    else:
        network = builder.create_network(1 << int(explicit))
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"[trt] onnx error {i}: {parser.get_error(i)}")
            raise RuntimeError(f"ONNX parse failed: {onnx_path}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace)
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    serialized = builder.build_serialized_network(network, config)
    assert serialized is not None, f"engine build failed: {onnx_path}"
    os.makedirs(osp.dirname(engine_path), exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized)
    print(f"[trt] built {engine_path} (fp16={fp16})")


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


def load_engines(task, onnx_dir, engine_dir, build=False, fp16=False):
    engines = {}
    for name, (in_names, out_names) in graph_io(task).items():
        onnx_path = osp.join(onnx_dir, f"{name}.onnx")
        engine_path = osp.join(engine_dir, f"{name}.engine")
        if build or not osp.exists(engine_path):
            assert osp.exists(onnx_path), (
                f"missing {onnx_path} — run scripts/export.py first"
            )
            build_engine(onnx_path, engine_path, fp16=fp16)
        engines[name] = TRTEngine(engine_path)
        print(f"[trt] loaded {engine_path}")
    return engines


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
                           build=args.build, fp16=args.fp16)

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
