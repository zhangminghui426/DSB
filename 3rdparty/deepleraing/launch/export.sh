#!/usr/bin/env bash
# usage: launch/export.sh <det|seg> [checkpoint] [extra args...]
# Exports the 4 module graphs to ONNX, then builds the TRT engines
# (trtexec CLI) in one step. Pass --onnx-only for ONNX only.
set -e
cd "$(dirname "$0")/.."
source launch/env.sh
TASK="${1:?task: det|seg}"; shift || true
case "$TASK" in
  det) CFG=configs/det_convfuser.py;            CKPT=ckpts/bevfusion-det-bf.pth ;;
  seg) CFG=configs/seg_fusion_bev256d2_lss.py;  CKPT=ckpts/bevfusion-seg-bf.pth ;;
  *) echo "unknown task: $TASK" >&2; exit 2 ;;
esac
if [ $# -gt 0 ] && [ -f "${1:-}" ]; then CKPT="$1"; shift; fi
[ -f "$CKPT" ] || { echo "checkpoint missing: $CKPT" >&2; exit 2; }
"$PY" deploy/export.py "$CFG" "$CKPT" --task "$TASK" "$@"
