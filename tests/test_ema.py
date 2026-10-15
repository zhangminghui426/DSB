"""ModelEMA semantics tests: decay ramp, float-buffer coverage, save format.

Run from repo root:
    python tests/test_ema.py
"""

import math
import os
import sys
import tempfile

import os.path as osp; sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), ".."))

import torch
from torch import nn

from models.ema import ModelEMA


def test_decay_ramp():
    m = nn.Conv2d(2, 2, 1)
    ema = ModelEMA(m, decay=0.999, tau=2000, updates=0)
    for x in (1, 100, 2000, 52776):
        expect = 0.999 * (1 - math.exp(-x / 2000))
        assert abs(ema.decay(x) - expect) < 1e-12, x
    print("[OK] decay ramp d(x) = decay * (1 - exp(-x/tau))")


def test_update_math():
    torch.manual_seed(0)
    m = nn.BatchNorm2d(3)  # float buffers (running stats) + int buffer (num_batches_tracked)
    ema = ModelEMA(m, decay=0.5, tau=2000, updates=0)
    with torch.no_grad():
        m.weight.fill_(2.0)
        m.running_mean.fill_(7.0)
        m.running_var.fill_(9.0)
        m.num_batches_tracked.fill_(5)
        ema.ema.weight.fill_(10.0)
        ema.ema.running_mean.fill_(10.0)
        ema.ema.running_var.fill_(10.0)
        ema.ema.num_batches_tracked.fill_(99)
    ema.update(m)
    d = ema.decay(1)
    assert ema.updates == 1
    # params AND float buffers follow v = d*v + (1-d)*model
    assert torch.allclose(ema.ema.weight, torch.full_like(ema.ema.weight, d * 10 + (1 - d) * 2))
    assert torch.allclose(ema.ema.running_mean, torch.full_like(ema.ema.running_mean, d * 10 + (1 - d) * 7))
    assert torch.allclose(ema.ema.running_var, torch.full_like(ema.ema.running_var, d * 10 + (1 - d) * 9))
    # int buffers untouched
    assert int(ema.ema.num_batches_tracked) == 99
    print("[OK] EMA update: params + float buffers, int buffers skipped")


def test_save_format():
    m = nn.Conv2d(2, 2, 1)
    ema = ModelEMA(m, updates=10560)
    with tempfile.TemporaryDirectory() as d:
        path = ema.save_checkpoint(d, epoch=0)
        assert osp.basename(path) == "epoch_1_ema.pth"
        ck = torch.load(path, map_location="cpu", weights_only=False)
        assert sorted(ck.keys()) == ["epoch", "state_dict", "updates"]
        assert ck["updates"] == 10560 and ck["epoch"] == 0
        assert set(ck["state_dict"].keys()) == set(m.state_dict().keys())
        # round-trip through load_state_dict
        ema2 = ModelEMA(nn.Conv2d(2, 2, 1), updates=0)
        ema2.load_state_dict(ck)
        assert ema2.updates == 10560
    print("[OK] official artifact format {epoch, state_dict, updates}")


def main():
    test_decay_ramp()
    test_update_math()
    test_save_format()
    print("ALL EMA TESTS PASSED")


if __name__ == "__main__":
    main()
