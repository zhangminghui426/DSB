#!/usr/bin/env bash
# usage: launch/train.sh <det|seg|multitask|occ|m0|stereo|stbase> [extra args...]
set -e
cd "$(dirname "$0")/.."
source launch/env.sh
TASK="${1:?task: det|seg|multitask|occ|m0|stereo|stbase}"; shift || true
case "$TASK" in
  det)       CFG=configs/det_convfuser.py;                        CKPT=ckpts/bevfusion-det-bf.pth ;;
  seg)       CFG=configs/seg_fusion_bev256d2_lss.py;              CKPT=ckpts/bevfusion-seg-bf.pth ;;
  multitask) CFG=configs/multitask_fusion.py;                     CKPT= ;;
  occ)       CFG=configs/flashocc_r50.py;                         CKPT=ckpts/flashocc-r50-256x704-bff.pth ;;
  m0)        CFG=configs/flashocc_r50_m0.py;                      CKPT= ;;
  stereo)    CFG=configs/flashocc_r50_4d_stereo.py;               CKPT= ;;
  stbase)    CFG=configs/flashocc_stbase_4d_stereo_512x1408.py;   CKPT= ;;
  *) echo "unknown task: $TASK" >&2; exit 2 ;;
esac
ARGS=()
if [ -n "$CKPT" ] && [ -f "$CKPT" ]; then ARGS+=(--load_from "$CKPT"); fi
if [ "${NGPU:-1}" -gt 1 ]; then
  "$TORCHRUN" --nproc_per_node="$NGPU" scripts/train.py "$CFG" "${ARGS[@]}" "$@"
else
  "$PY" scripts/train.py "$CFG" "${ARGS[@]}" "$@"
fi
