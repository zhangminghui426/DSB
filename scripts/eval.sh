#!/usr/bin/env bash
# usage: scripts/eval.sh <det|seg|multitask|occ|m0|stereo|stbase> [checkpoint] [extra args...]
set -e
cd "$(dirname "$0")/.."
source scripts/env.sh
TASK="${1:?task: det|seg|multitask|occ|m0|stereo|stbase}"; shift || true
case "$TASK" in
  det)       CFG=configs/det_convfuser.py;                      CKPT=ckpts/bevfusion-det-bf.pth;      METRIC=bbox ;;
  seg)       CFG=configs/seg_fusion_bev256d2_lss.py;            CKPT=ckpts/bevfusion-seg-bf.pth;      METRIC=map ;;
  multitask) CFG=configs/multitask_fusion.py;                   CKPT=;                                METRIC= ;;
  occ)       CFG=configs/flashocc_r50.py;                       CKPT=ckpts/flashocc-r50-256x704-bff.pth; METRIC= ;;
  m0)        CFG=configs/flashocc_r50_m0.py;                    CKPT=;                                METRIC= ;;
  stereo)    CFG=configs/flashocc_r50_4d_stereo.py;             CKPT=;                                METRIC= ;;
  stbase)    CFG=configs/flashocc_stbase_4d_stereo_512x1408.py; CKPT=;                                METRIC= ;;
  *) echo "unknown task: $TASK" >&2; exit 2 ;;
esac
if [ $# -gt 0 ] && [ -f "${1:-}" ]; then CKPT="$1"; shift; fi
[ -f "$CKPT" ] || { echo "checkpoint missing: $CKPT" >&2; exit 2; }
EVAL=(); [ -n "$METRIC" ] && EVAL=(--eval "$METRIC")
"$PY" scripts/test.py "$CFG" "$CKPT" "${EVAL[@]}" --jsonfile_prefix "outputs/eval_${TASK}" "$@"
