#!/usr/bin/env bash
# usage: deploy/launch/perf_trt.sh <det|seg>
# Collects per-engine layer/profile dumps with trtexec (--dumpProfile
# --dumpLayerInfo) and normalizes them to
#   outputs/perf/layers_<graph>.json  +  profile_<graph>.json
# so the NVIDIA trt-perf-analysis tooling (deploy/tools/trt_perf/) can build
# an interactive HTML performance report. Requires built engines (outputs/trt).
#
# trtexec lookup order: $TRTEXEC env var, then PATH, then common install
# locations (TensorRT-10.9 tar bin/). pip-only tensorrt wheels have no
# trtexec binary — point TRTEXEC at the tar package's bin/trtexec.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"            # 3rdparty/deeplearning/deploy/launch
DL="$(cd "$HERE/../.." && pwd)"                  # 3rdparty/deeplearning（模型库根）
cd "$DL"
source "$HERE/env.sh"
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
($DL/deploy/launch/infer_trt.sh ... --build)" >&2; exit 2; }
  work="$PERF_DIR/_work_$g"
  rm -rf "$work"; mkdir -p "$work"
  # --dumpProfile/--dumpLayerInfo 只打印日志；落盘需 --exportProfile/
  # --exportLayerInfo（TRT 10/11 通用，文件名由 =<file> 显式指定）
  ( cd "$work" && "$TRTEXEC" --loadEngine="$OLDPWD/$engine" \
      --exportLayerInfo=layers.json --exportProfile=profile.json \
      --warmUp=500 --iterations=50 --noDataTransfers >/dev/null )
  [ -f "$work/layers.json" ] && mv "$work/layers.json" "$PERF_DIR/layers_$g.json"
  [ -f "$work/profile.json" ] && mv "$work/profile.json" "$PERF_DIR/profile_$g.json"
  rm -rf "$work"
  [ -f "$PERF_DIR/layers_$g.json" ] || { echo \
    "error: layers_$g.json not produced (trtexec --exportLayerInfo failed)" >&2; exit 2; }
  [ -f "$PERF_DIR/profile_$g.json" ] || { echo \
    "error: profile_$g.json not produced (trtexec --exportProfile failed)" >&2; exit 2; }
  echo "[perf] dumped $g -> layers_$g.json / profile_$g.json"
done

echo
echo "[perf] done. Generate the report with deploy/tools/trt_perf/run.sh:"
echo "  bash $DL/deploy/tools/trt_perf/run.sh"
