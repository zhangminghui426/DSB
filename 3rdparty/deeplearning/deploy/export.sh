#!/usr/bin/env bash
# usage: deploy/export.sh <det|seg> [checkpoint] [extra args...]
# onnx 阶段导出：5 张分模块图 -> ONNX（含 SCN）。
# TRT engine 用 deploy/export_trt.sh；推理输入用 deploy/scripts/example_data.py。
# Runs with the model repo as CWD so relative paths keep working.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"            # 3rdparty/deeplearning/deploy
DL="$(cd "$HERE/.." && pwd)"                     # 3rdparty/deeplearning（模型库根）
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
"$PY" "$DL/deploy/scripts/export.py" "$CFG" "$CKPT" "$@"
