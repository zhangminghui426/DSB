#!/usr/bin/env bash
# usage: deploy/launch/visualize_trt.sh [checkpoint] [extra args...]
# Renders TRT inference results to outputs/vis/:
#   sample{N}_det.jpg            BEV + 6 camera views with projected 3D boxes
#   sample{N}-{M}_seq.gif        temporal sequence gif (--frames N)
# Reuse a previous dump with: --npz outputs/trt/sample5.npz
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"            # 3rdparty/deepleraing/deploy/launch
DL="$(cd "$HERE/../.." && pwd)"                  # 3rdparty/deepleraing（模型库根）
cd "$DL"
source "$HERE/env.sh"
CFG=configs/det_convfuser.py
CKPT=ckpts/bevfusion-det-bf.pth
if [ $# -gt 0 ] && [ -f "${1:-}" ]; then CKPT="$1"; shift; fi
"$PY" "$DL/deploy/python/visualize.py" "$CFG" "$CKPT" \
    --engine_dir outputs/trt --out outputs/vis "$@"
