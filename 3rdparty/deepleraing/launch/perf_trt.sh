#!/usr/bin/env bash
# usage: launch/perf_trt.sh <det|seg>
# Collects per-engine layer/profile dumps with trtexec (--dumpProfile
# --dumpLayerInfo) and normalizes them to
#   outputs/perf/layers_<graph>.json  +  profile_<graph>.json
# so the NVIDIA trt-perf-analysis tooling (tools/trt_perf/) can build an
# interactive HTML performance report. Requires built engines (outputs/trt).
#
# trtexec lookup order: $TRTEXEC env var, then PATH, then common install
# locations (TensorRT-10.9 tar bin/). pip-only tensorrt wheels have no
# trtexec binary — point TRTEXEC at the tar package's bin/trtexec.
set -e
cd "$(dirname "$0")/.."
source launch/env.sh
TASK="${1:?task: det|seg}"
case "$TASK" in
  det) GRAPHS="camera.backbone camera.vtransform fuser head.bbox" ;;
  seg) GRAPHS="camera.backbone camera.vtransform fuser head.map" ;;
  *) echo "unknown task: $TASK" >&2; exit 2 ;;
esac

ENGINE_DIR=outputs/trt
PERF_DIR=outputs/perf
mkdir -p "$PERF_DIR"

TRTEXEC=${TRTEXEC:-}
if [ -z "$TRTEXEC" ]; then
  for c in trtexec /usr/local/TensorRT/bin/trtexec \
           /opt/TensorRT/bin/trtexec /usr/src/tensorrt/bin/trtexec; do
    if command -v "$c" >/dev/null 2>&1; then TRTEXEC="$c"; break; fi
  done
fi
[ -n "$TRTEXEC" ] || { echo "trtexec not found. Set TRTEXEC=/path/to/trtexec \
(e.g. the TensorRT-10.9 tar package's bin/trtexec)." >&2; exit 2; }
echo "[perf] trtexec: $TRTEXEC"

for g in $GRAPHS; do
  engine="$ENGINE_DIR/$g.engine"
  [ -f "$engine" ] || { echo "missing $engine — build engines first \
(launch/infer_trt.sh ... --build)" >&2; exit 2; }
  work="$PERF_DIR/_work_$g"
  rm -rf "$work"; mkdir -p "$work"
  ( cd "$work" && "$TRTEXEC" --loadEngine="$OLDPWD/$engine" \
      --dumpProfile --dumpLayerInfo \
      --warmUp=500 --iterations=50 --noDataTransfers >/dev/null )
  layer=$(ls "$work"/layers*.json 2>/dev/null | head -1)
  prof=$(ls "$work"/profile*.json 2>/dev/null | head -1)
  [ -n "$layer" ] && mv "$layer" "$PERF_DIR/layers_$g.json"
  [ -n "$prof" ] && mv "$prof" "$PERF_DIR/profile_$g.json"
  rm -rf "$work"
  echo "[perf] dumped $g -> layers_$g.json / profile_$g.json"
done

echo
echo "[perf] done. Generate the report with tools/trt_perf/run.sh:"
echo "  bash ../../tools/trt_perf/run.sh"
