#!/usr/bin/env bash
# usage: launch/infer_trt.sh <det|seg> [checkpoint] [extra args...]
# Builds (--build, optional --fp16) + runs TRT engines on a val sample.
# Requires export.sh to have produced outputs/onnx/*.onnx first.
# ONNX -> engine uses trtexec (see deploy/builder.py): set
# TRTEXEC=/path/to/bin/trtexec when the conda env is pip-only (no trtexec).
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
"$PY" deploy/inferencer.py "$CFG" "$CKPT" --task "$TASK" "$@"
