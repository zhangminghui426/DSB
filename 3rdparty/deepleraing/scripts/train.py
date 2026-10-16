"""Training entry point — torchrun-based (det/seg/multitask/occ recipes).

Single GPU:
    python scripts/train.py CONFIG [overrides...]
Multi GPU:
    torchrun --nproc_per_node=N scripts/train.py CONFIG [overrides...]

Overrides use dot notation, e.g.:
    --options max_epochs=2 data.samples_per_gpu=1
"""

import argparse
import logging
import os
import os.path as osp
import sys

sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), ".."))

import torch

from datasets import build_dataset, build_dataloader
from models import build_model
from scripts.config import load_config
from scripts.trainer import Trainer, init_dist, is_main_process, set_random_seed

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="train")
    p.add_argument("config", help="train config .py path")
    p.add_argument("--work_dir", default=None)
    p.add_argument("--load_from", default=None, help="pretrained checkpoint (weights only)")
    p.add_argument("--resume_from", default=None, help="full training checkpoint")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None, help="override data.samples_per_gpu")
    p.add_argument("--epochs", type=int, default=None, help="override max_epochs")
    p.add_argument("--no_amp", action="store_true", help="disable fp16")
    p.add_argument("--no_eval", action="store_true", help="skip in-training evaluation")
    p.add_argument("--debug_iters", type=int, default=None, help="stop each epoch after N iterations (smoke test)")
    p.add_argument("--options", nargs="*", default=[], help="dot-notation config overrides")
    return p.parse_args()


def build_loaders(cfg, distributed, seed):
    train_dataset = build_dataset(cfg.data.train)
    val_dataset = build_dataset(cfg.data.val)
    bs = cfg.data.get("samples_per_gpu", 1)
    nw = cfg.data.get("workers_per_gpu", 4)
    train_loader = build_dataloader(
        train_dataset, bs, nw, distributed, shuffle=True, seed=seed, drop_last=True
    )
    val_loader = build_dataloader(val_dataset, 1, max(1, nw // 2), distributed)
    return train_dataset, val_dataset, train_loader, val_loader


def main():
    args = parse_args()

    rank, local_rank, world_size = init_dist()
    distributed = world_size > 1

    logging.basicConfig(
        level=logging.INFO if is_main_process() else logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    cfg = load_config(args.config, options=args.options)
    seed = args.seed if args.seed is not None else cfg.get("seed", 0)
    deterministic = cfg.get("deterministic", False)
    set_random_seed(seed, deterministic=deterministic, rank=rank)
    torch.backends.cudnn.benchmark = cfg.get("cudnn_benchmark", False)

    if args.batch_size is not None:
        cfg.data.samples_per_gpu = args.batch_size
    if args.epochs is not None:
        cfg.max_epochs = args.epochs
    if args.no_amp:
        cfg.fp16 = None
    if args.no_eval:
        cfg.evaluation = None

    work_dir = args.work_dir or osp.join(
        "outputs", osp.splitext(osp.basename(args.config))[0]
    )
    if is_main_process():
        os.makedirs(work_dir, exist_ok=True)
        cfg.dump(osp.join(work_dir, "configs.yaml"))

    train_dataset, val_dataset, train_loader, val_loader = build_loaders(cfg, distributed, seed)
    if is_main_process():
        logger.info(
            "train samples: %d (CBGS len %d), val samples: %d, batch/gpu: %d",
            len(train_dataset.dataset) if hasattr(train_dataset, "dataset") else len(train_dataset),
            len(train_dataset),
            len(val_dataset),
            cfg.data.get("samples_per_gpu", 1),
        )

    model = build_model(cfg.model)

    trainer = Trainer(
        model=model,
        cfg=cfg,
        train_loader=train_loader,
        val_loader=None if args.no_eval else val_loader,
        val_dataset=None if args.no_eval else val_dataset,
        work_dir=work_dir,
        load_from=args.load_from or cfg.get("load_from", None),
        resume_from=args.resume_from or cfg.get("resume_from", None),
        debug_iters=args.debug_iters,
    )
    trainer.train()


if __name__ == "__main__":
    main()
