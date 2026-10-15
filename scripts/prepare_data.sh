#!/usr/bin/env bash
# usage: scripts/prepare_data.sh [--with-gt-db] [--version v1.0-mini] [--dataroot data/nuscenes]
# Creates the data symlink + det/seg infos pkls; the occ infos enrichment
# (ann_infos/scene/occ_path) is tools/create_data.py on top of the bevdetv2 pkls.
set -e
cd "$(dirname "$0")/.."
source scripts/env.sh
WITH_GT_DB=0; VERSION=v1.0-mini; DATAROOT=data/nuscenes
while [ $# -gt 0 ]; do
  case "$1" in
    --with-gt-db) WITH_GT_DB=1 ;;
    --version) VERSION="$2"; shift ;;
    --dataroot) DATAROOT="$2"; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
  shift
done
[ -e "$DATAROOT" ] || ln -sfn ../../data/nuscenes "$DATAROOT"
"$PY" tools/convert_nuscenes.py --dataroot "$DATAROOT" --version "$VERSION"
if [ "$WITH_GT_DB" = 1 ]; then
  "$PY" tools/create_gt_database.py --dataroot "$DATAROOT" --version "$VERSION"
fi
