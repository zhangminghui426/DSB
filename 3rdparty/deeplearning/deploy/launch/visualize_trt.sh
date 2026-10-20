#!/usr/bin/env bash
# usage: deploy/launch/visualize_trt.sh [extra args...]
# Renders TRT inference results to outputs/vis/:
#   sample{N}_det.jpg            BEV + 6 camera views with projected 3D boxes
#   sample{N}-{M}_seq.gif        temporal sequence gif (--frames N)
# TRT 推理直接用 engine（不需要 checkpoint）；也可 --npz 复用已 dump 结果。
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"            # 3rdparty/deeplearning/deploy/launch
DL="$(cd "$HERE/../.." && pwd)"                  # 3rdparty/deeplearning（模型库根）
cd "$DL"
source "$HERE/env.sh"
CFG=configs/det_convfuser.py
"$PY" "$DL/deploy/python/scripts/visualize.py" "$CFG" \
    --engine_dir outputs/trt --out outputs/vis "$@"
