"""Remap occupancy labels to a subset of classes.

Usage:
    python tools/remap_occ.py INPUT_NPZ OUTPUT_NPZ
"""

import argparse
import os

import numpy as np

# Default remap: merge rare classes into nearest valid class
# Occ3D-nuScenes 16 -> 11 class mapping
DEFAULT_REMAP = {
    0: 0,   # noise -> noise
    1: 0,   # ignore -> noise
    2: 0,   # barrier -> noise (or 1 if barrier class kept)
    3: 1,   # bicycle -> 1 (bike)
    4: 2,   # bus -> 2 (vehicle)
    5: 3,   # car -> 3 (vehicle)
    6: 4,   # construction -> 4 (other)
    7: 5,   # motorcycle -> 5 (bike)
    8: 6,   # pedestrian -> 6 (ped)
    9: 7,   # traffic cone -> 7 (cone)
    10: 8,  # trailer -> 8 (vehicle)
    11: 9,  # truck -> 9 (vehicle)
    12: 10, # driveable surface -> 10 (driveable)
    13: 11, # other flat -> 11 (flat)
    14: 12, # sidewalk -> 12 (walkway)
    15: 13, # terrain -> 13 (terrain)
    16: 14, # manmade -> 14 (manmade)
    17: 15, # vegetation -> 15 (vegetation)
}


def remap_labels(labels, remap_dict=None):
    """Remap label ids using a lookup table."""
    if remap_dict is None:
        remap_dict = DEFAULT_REMAP
    out = np.zeros_like(labels)
    for src, dst in remap_dict.items():
        out[labels == src] = dst
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input")
    ap.add_argument("output")
    args = ap.parse_args()

    data = np.load(args.input)
    semantics = data["semantics"]
    remapped = remap_labels(semantics)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.savez_compressed(
        args.output,
        semantics=remapped,
        mask_lidar=data["mask_lidar"],
        mask_camera=data["mask_camera"],
    )
    print(f"Remapped {semantics.shape} -> {remapped.shape}, saved to {args.output}")


if __name__ == "__main__":
    main()
