"""Remap official BEVFusion checkpoints (trained with the vendored spconv
1.x) to the bf package (pip spconv 2.x).

Only change needed: 5-dim sparse conv weights under `encoders.lidar.backbone`
are stored as (K, D, H, Cin, Cout) by spconv 1.x, while spconv 2.x expects
(Cout, K, D, H, Cin). All other keys pass through unchanged (bf keeps the
original parameter tree: encoders/fuser/decoder/heads).

Usage:
    python tools/remap_pretrained.py ckpts/bevfusion-det.pth \
        ckpts/bevfusion-det-bf.pth
    python tools/remap_pretrained.py ckpts/bevfusion-seg.pth \
        ckpts/bevfusion-seg-bf.pth
"""

import argparse
import sys

import torch

LIDAR_PREFIX = "encoders.lidar.backbone."


def remap_state_dict(state: dict) -> dict:
    out = {}
    n_permuted = 0
    for k, v in state.items():
        if k.startswith(LIDAR_PREFIX) and v.ndim == 5:
            # spconv 1.x (K,D,H,Cin,Cout) -> spconv 2.x (Cout,K,D,H,Cin)
            v = v.permute(4, 0, 1, 2, 3).contiguous()
            n_permuted += 1
        out[k] = v
    print(f"remapped: {len(out)} keys, {n_permuted} sparse conv weights permuted")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", help="official checkpoint (bevfusion-det.pth / bevfusion-seg.pth)")
    parser.add_argument("dst", help="output path for the bf-compatible checkpoint")
    args = parser.parse_args()

    ckpt = torch.load(args.src, map_location="cpu")
    meta = {k: v for k, v in ckpt.items() if k != "state_dict"} if isinstance(ckpt, dict) else {}
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    new_state = remap_state_dict(state)
    out = dict(meta)
    out["state_dict"] = new_state
    torch.save(out, args.dst)
    print(f"saved -> {args.dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
