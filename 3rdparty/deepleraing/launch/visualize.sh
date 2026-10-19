#!/usr/bin/env bash
# usage: launch/visualize.sh <det|seg|occ> [checkpoint] [extra args...]
set -e
cd "$(dirname "$0")/.."
source launch/env.sh
TASK="${1:?task: det|seg|occ}"; shift || true
case "$TASK" in
  det) CFG=configs/det_convfuser.py;            CKPT=ckpts/bevfusion-det-bf.pth ;;
  seg) CFG=configs/seg_fusion_bev256d2_lss.py;  CKPT=ckpts/bevfusion-seg-bf.pth ;;
  occ) CFG=configs/flashocc_r50.py;             CKPT=ckpts/flashocc-r50-256x704-bff.pth ;;
  *) echo "unknown task: $TASK" >&2; exit 2 ;;
esac
if [ $# -gt 0 ] && [ -f "${1:-}" ]; then CKPT="$1"; shift; fi
"$PY" scripts/visualize.py "$CFG" "$CKPT" --task "$TASK" --out outputs/vis "$@"
