#!/usr/bin/env bash
# usage: deploy/export.sh <det|seg> [checkpoint] [extra args...]
# Python 与 C++ 推理共用的前置导出：4 张分模块图 -> ONNX -> TRT engine
# （trtexec CLI）一步完成。C++ 侧输入文件另用 deploy/tool/export_cpp_input.py。
# 可选 --onnx-only（只导出 ONNX）。Runs with the model repo as CWD so
# configs/ ckpts/ outputs/ relative paths keep working.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"            # 3rdparty/deeplearning/deploy
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
"$PY" "$DL/deploy/export.py" "$CFG" "$CKPT" --task "$TASK" "$@"
