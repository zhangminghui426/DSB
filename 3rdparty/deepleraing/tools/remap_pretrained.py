"""Remap pretrained checkpoint keys to match current model architecture.

Useful when porting weights from mmdet3d / other codebases.

Usage:
    python tools/remap_pretrained.py INPUT_PTH OUTPUT_PTH
"""

import argparse

import torch

# Default key mappings (prefix replacements)
DEFAULT_RENAMES = [
    # mmdet3d -> our naming
    ("backbone.", "encoders.camera.backbone."),
    ("neck.", "encoders.camera.neck."),
    ("pts_bbox_head.", "heads.object."),
    ("pts_mapparse_head.", "heads.map."),
    # bevfusion specific
    ("img_bev_encoder.", "encoders.camera."),
    ("lidar_bev_encoder.", "encoders.lidar."),
]


def remap_state_dict(state_dict, renames=None):
    """Rename state_dict keys."""
    if renames is None:
        renames = DEFAULT_RENAMES

    new_dict = {}
    for key, val in state_dict.items():
        new_key = key
        for old_prefix, new_prefix in renames:
            if new_key.startswith(old_prefix):
                new_key = new_prefix + new_key[len(old_prefix):]
                break
        new_dict[new_key] = val
    return new_dict


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--strict", action="store_true", help="error on unmapped keys")
    args = ap.parse_args()

    ckpt = torch.load(args.input, map_location="cpu")
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    new_state = remap_state_dict(state_dict)

    if args.strict:
        old_keys = set(state_dict.keys())
        new_keys = set(new_state.keys())
        unmapped = old_keys - new_keys
        if unmapped:
            print(f"Warning: {len(unmapped)} keys not remapped:")
            for k in list(unmapped)[:10]:
                print(f"  {k}")

    torch.save({"state_dict": new_state}, args.output)
    print(f"Remapped {len(state_dict)} keys -> {len(new_state)}, saved to {args.output}")


if __name__ == "__main__":
    main()
