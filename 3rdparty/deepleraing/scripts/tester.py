"""Inference loop used by test.py and render.py — no metric logic."""

import logging

import torch

logger = logging.getLogger(__name__)


@torch.no_grad()
def run_inference(model, loader, device="cuda", use_amp=True):
    """Run model over loader; return list of per-batch output dicts.

    Pure inference — no evaluation, no saving. Callers (test.py / render.py)
    own metric and serialization.
    """
    model.eval()
    model.to(device)
    outputs = []

    amp_dtype = torch.float16 if use_amp else torch.bfloat16
    for i, data in enumerate(loader):
        # Move tensors to device; keep non-tensor metadata as-is
        inputs = {}
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(device)
            else:
                inputs[k] = v

        with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            result = model(**inputs, return_loss=False)

        # result is a list of dicts (per-sample) or a single dict
        if isinstance(result, list):
            outputs.extend(result)
        else:
            outputs.append(result)

        if (i + 1) % 50 == 0:
            logger.info("inference: %d/%d batches done", i + 1, len(loader))

    return outputs
