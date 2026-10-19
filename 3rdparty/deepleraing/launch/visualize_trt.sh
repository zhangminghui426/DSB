#!/usr/bin/env bash
# usage: launch/visualize_trt.sh [checkpoint] [extra args...]
# Renders TRT inference results to outputs/vis/:
#   sample{N}_det.jpg      BEV + 6 camera views with projected 3D boxes
#   sample{N}_bev_rot.gif  360-deg BEV rotation (24 frames)
# Reuse a previous dump with: --npz outputs/trt/sample5.npz
set -e
cd "$(dirname "$0")/.."
source launch/env.sh
CFG=configs/det_convfuser.py
CKPT=ckpts/bevfusion-det-bf.pth
if [ $# -gt 0 ] && [ -f "${1:-}" ]; then CKPT="$1"; shift; fi
"$PY" deploy/visualize.py "$CFG" "$CKPT" \
    --engine_dir outputs/trt --out outputs/vis "$@"
