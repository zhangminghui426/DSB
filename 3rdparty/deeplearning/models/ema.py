"""MEGVII-style Model EMA — the official FlashOCC checkpoints ARE EMA weights.

Exact semantics of the original MEGVIIEMAHook/ModelEMA:
- decay ramp: d(x) = decay * (1 - exp(-x / tau)), decay=0.999, tau=2000
- EMA covers parameters AND floating-point buffers (BN running stats);
  integer buffers are left untouched
- updates counter starts at init_updates (10560 in the official recipe)
- artifact: epoch_N_ema.pth = {'epoch', 'state_dict', 'updates'}
"""

import math
from copy import deepcopy

import torch

__all__ = ["ModelEMA"]


class ModelEMA:
    def __init__(self, model, decay=0.9990, tau=2000, updates=0):
        """model: UNWRAPPED nn.Module (no DDP)."""
        self.ema = deepcopy(model).eval()
        self.updates = updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = self.decay(self.updates)
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v *= d
                v += (1.0 - d) * msd[k].detach().to(v.device)

    def state_dict(self):
        return dict(state_dict=self.ema.state_dict(), updates=self.updates)

    def load_state_dict(self, state):
        self.ema.load_state_dict(state["state_dict"])
        self.updates = state["updates"]

    def save_checkpoint(self, work_dir, epoch):
        """Write the official-format epoch_{epoch+1}_ema.pth; returns the path."""
        import os.path as osp

        path = osp.join(work_dir, f"epoch_{epoch + 1}_ema.pth")
        torch.save(
            dict(epoch=epoch, state_dict=self.ema.state_dict(), updates=self.updates),
            path,
        )
        return path
