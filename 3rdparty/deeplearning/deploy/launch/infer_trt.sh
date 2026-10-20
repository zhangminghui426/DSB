#!/usr/bin/env bash
# usage: deploy/launch/infer_trt.sh <det|seg> [extra args...]
# Runs the TRT engines on a val sample (deploy/python/scripts/pipeline.py)。
# TRT 推理直接用 --engine_dir 下的 engine，不需要 .pth；
# 与 torch 参考对照时加 --verify --checkpoint <pth>。
# Requires deploy/launch/export.sh to have produced outputs/onnx + outputs/trt.
# ONNX -> engine uses trtexec (see deploy/python/builder.py): set
# TRTEXEC=/path/to/bin/trtexec when the conda env is pip-only (no trtexec).
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"            # 3rdparty/deeplearning/deploy/launch
DL="$(cd "$HERE/../.." && pwd)"                  # 3rdparty/deeplearning（模型库根）
cd "$DL"
source "$HERE/env.sh"
TASK="${1:?task: det|seg}"; shift || true
case "$TASK" in
  det) CFG=configs/det_convfuser.py ;;
  seg) CFG=configs/seg_fusion_bev256d2_lss.py ;;
  *) echo "unknown task: $TASK" >&2; exit 2 ;;
esac
[ -f "$CFG" ] || { echo "config missing: $CFG" >&2; exit 2; }
"$PY" "$DL/deploy/python/scripts/pipeline.py" "$CFG" --task "$TASK" "$@"
