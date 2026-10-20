#!/usr/bin/env bash
# usage: deploy/launch/export.sh <det|seg> [checkpoint] [extra args...]
# Exports the 4 module graphs to ONNX, then builds the TRT engines
# (trtexec CLI) in one step. Pass --onnx-only for ONNX only.
# Runs with the model repo (3rdparty/deeplearning) as CWD so
# configs/ ckpts/ outputs/ relative paths keep working.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"            # 3rdparty/deeplearning/deploy/launch
DL="$(cd "$HERE/../.." && pwd)"                  # 3rdparty/deeplearning（模型库根）
cd "$DL"
source "$HERE/env.sh"
TASK="${1:?task: det|seg}"; shift || true
case "$TASK" in
  det) CFG=configs/det_convfuser.py;            CKPT=ckpts/bevfusion-det-bf.pth ;;
  seg) CFG=configs/seg_fusion_bev256d2_lss.py;  CKPT=ckpts/bevfusion-seg-bf.pth ;;
  *) echo "unknown task: $TASK" >&2; exit 2 ;;
esac
if [ $# -gt 0 ] && [ -f "${1:-}" ]; then CKPT="$1"; shift; fi
[ -f "$CKPT" ] || { echo "checkpoint missing: $CKPT" >&2; exit 2; }
"$PY" "$DL/deploy/python/export.py" "$CFG" "$CKPT" --task "$TASK" "$@"
