"""Test/eval entry point — torchrun-based (det/seg/occ metrics dispatch).

Single GPU:
    python scripts/test.py CONFIG CHECKPOINT
Multi GPU:
    torchrun --nproc_per_node=N scripts/test.py CONFIG CHECKPOINT

CHECKPOINT is typically the EMA artifact (epoch_N_ema.pth or the official
release) — {'epoch', 'state_dict', 'updates'} and full training checkpoints
are both accepted.
"""

import argparse
import json
import logging
import os.path as osp
import pickle
import sys

sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), ".."))

import torch

from datasets import build_dataset, build_dataloader
from models import build_model
from scripts.config import load_config
from scripts.tester import run_inference
from scripts.trainer import init_dist, is_main_process, load_model_checkpoint


def parse_args():
    p = argparse.ArgumentParser(description="test")
    p.add_argument("config")
    p.add_argument("checkpoint")
    p.add_argument("--eval", default=None, choices=["bbox", "map"],
                   help="det/seg metric (default: infer from output keys)")
    p.add_argument("--jsonfile_prefix", default=None, help="prefix for results_nusc.json")
    p.add_argument("--save_dir", default=None, help="dump per-token pred.npz here (occ)")
    p.add_argument("--no-aavt", action="store_true",
                   help="do NOT set align_after_view_transfromation for 4D models")
    p.add_argument("--out", default=None, help="pickle raw outputs to file")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--options", nargs="*", default=[])
    return p.parse_args()


def main():
    args = parse_args()
    rank, local_rank, world_size = init_dist()
    distributed = world_size > 1
    logging.basicConfig(
        level=logging.INFO if is_main_process() else logging.WARNING,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger(__name__)

    cfg = load_config(args.config, options=args.options)
    torch.manual_seed(cfg.get("seed", 0))

    # official test protocol: 4D-stereo occ models align adjacent BEV feats
    occ_cfg = (cfg.model.get("encoders") or {}).get("occ") or {}
    if occ_cfg.get("type") == "OccEncoderStereo4D" and not args.no_aavt:
        occ_cfg["align_after_view_transfromation"] = True
    occ_only = bool(occ_cfg) and cfg.model.get("decoder") is None

    dataset = build_dataset(cfg.data.test)
    loader = build_dataloader(
        dataset, args.batch_size, cfg.data.get("workers_per_gpu", 4), distributed
    )

    model = build_model(cfg.model).to("cuda")
    msg, _ = load_model_checkpoint(model, args.checkpoint, strict=True)
    if is_main_process():
        log.info("loaded %s: missing=%d unexpected=%d", args.checkpoint,
                 len(msg.missing_keys), len(msg.unexpected_keys))

    outputs = run_inference(model, loader, "cuda", use_amp=not occ_only)

    if args.out and is_main_process():
        with open(args.out, "wb") as f:
            pickle.dump(outputs, f)
        log.info("raw outputs saved to %s", args.out)

    if is_main_process():
        metric = args.eval
        if metric is None and outputs:
            metric = "map" if "masks_bev" in outputs[0] else "bbox"
        kwargs = {}
        if args.jsonfile_prefix:
            kwargs["jsonfile_prefix"] = args.jsonfile_prefix
        results = dataset.evaluate(outputs, metric=metric,
                                   save_dir=args.save_dir, **kwargs)
        print(json.dumps({k: (float(v) if isinstance(v, (int, float)) else v)
                          for k, v in results.items()}, indent=2))


if __name__ == "__main__":
    main()
