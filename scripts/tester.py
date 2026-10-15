"""Inference engine: run a model over a dataloader and (optionally) evaluate.

Replaces mmcv single_gpu_test / multi_gpu_test.
"""

import torch
import torch.distributed as dist

from scripts.trainer import is_distributed, to_device


@torch.no_grad()
def run_inference(model, loader, device="cuda", use_amp=True, progress=True):
    """Forward the whole loader in eval mode; returns the FULL ordered
    result list on every rank (gathered when distributed).

    Callers pick the protocol: det/seg evaluate under fp16 autocast
    (use_amp=True); the occ recipe evaluates in fp32 (use_amp=False) to match
    the protocol the official metrics were produced with."""
    was_training = model.training
    model.eval()

    rank_outputs = []
    iterator = loader
    if progress:
        try:
            from tqdm import tqdm

            if (not is_distributed()) or dist.get_rank() == 0:
                iterator = tqdm(loader, desc="inference", total=len(loader))
        except ImportError:
            pass

    for data in iterator:
        data = to_device(data, device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            outputs = model(**data)
        rank_outputs.extend(outputs)

    if is_distributed():
        world = dist.get_world_size()
        gathered = [None for _ in range(world)]
        dist.all_gather_object(gathered, rank_outputs)
        # interleave back into sampler order: DistributedSampler splits
        # indices as rank r -> samples r, r+world, r+2*world, ...
        merged = [None] * (max(len(g) for g in gathered) * world)
        for r, part in enumerate(gathered):
            for i, item in enumerate(part):
                idx = i * world + r
                if idx < len(merged):
                    merged[idx] = item
        n = len(loader.dataset)
        results = [m for m in merged[:n] if m is not None]
    else:
        results = rank_outputs

    if was_training:
        model.train()
    return results
