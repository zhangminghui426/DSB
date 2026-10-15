"""Remap an official FlashOCC checkpoint (top-level attribute naming) onto the
unified BevPerception parameter tree:

    img_backbone / img_neck / img_view_transformer / img_bev_encoder_* /
    pre_process_net        -> encoders.occ.<same>
    occ_head               -> heads.occ.<same>

Same precedent as tools/remap_pretrained.py (spconv layout) and the
bevfusion-det.pth -> -bf.pth remaps: one-off artifact, strict load afterwards.

Usage:
    python tools/remap_occ.py ckpts/flashocc-r50-256x704.pth \
        ckpts/flashocc-r50-256x704-bff.pth
"""

import argparse

import torch


def remap_key(key):
    if key.startswith("occ_head."):
        return "heads.occ." + key[len("occ_head."):]
    return "encoders.occ." + key


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src")
    p.add_argument("dst")
    args = p.parse_args()

    ckpt = torch.load(args.src, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    out = {remap_key(k): v for k, v in state.items()}
    assert len(out) == len(state), "remap is not a bijection"
    torch.save(
        {
            "epoch": ckpt.get("epoch") if isinstance(ckpt, dict) else None,
            "state_dict": out,
            "updates": ckpt.get("updates") if isinstance(ckpt, dict) else None,
        },
        args.dst,
    )
    print(f"remapped {len(out)} keys -> {args.dst}")


if __name__ == "__main__":
    main()
