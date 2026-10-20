#!/usr/bin/env bash
# usage: bash deploy/tools/trt_perf/run.sh [PERF_DIR]
#   PERF_DIR  default: 3rdparty/deeplearning/outputs/perf
#   MODEL_NAME env (default bevfusion-det) passed to the report.
# Wraps the NVIDIA trt-perf-analysis pipeline:
#   analyze_trt_perf.py -> validate_analyze_data.py --require-passed
#   -> package_report.py (interactive HTML report)
# Requires layers_*.json / profile_*.json pairs in PERF_DIR
# (collect them with deploy/launch/perf_trt.sh).
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
# 本目录位于 3rdparty/deeplearning/deploy/tools/trt_perf/，上 5 级为仓库根
REPO_ROOT="$(cd "$HERE/../../../../.." && pwd)"
# Resolve PERF_DIR against the repo root so the script works from any CWD.
PERF_DIR="${1:-$REPO_ROOT/3rdparty/deeplearning/outputs/perf}"
case "$PERF_DIR" in
  /*) ;;
  *) PERF_DIR="$REPO_ROOT/$PERF_DIR" ;;
esac
MODEL_NAME="${MODEL_NAME:-bevfusion-det}"
PY="${PY:-python3}"
STAGE="$PERF_DIR/analyze-data.json"
REPORT="$PERF_DIR/report"

[ -d "$PERF_DIR" ] || { echo "perf dir missing: $PERF_DIR" >&2; exit 2; }
ls "$PERF_DIR"/layers_*.json >/dev/null 2>&1 || { echo \
  "no layers_*.json in $PERF_DIR — collect with deploy/launch/perf_trt.sh" >&2; exit 2; }

# Large bundled assets (app.js / netron-assets.js) are not committed to this
# repo; package_report.py's template validation requires them, so fetch the
# exact NVIDIA originals from the official TensorRT repo when missing.
NETRON_RAW="https://raw.githubusercontent.com/NVIDIA/TensorRT/main/.agents/skills/trt-perf-analysis/report_template"
for f in app.js netron-assets.js; do
  if [ ! -f "$HERE/report_template/$f" ]; then
    echo "[perf] fetching $f from NVIDIA/TensorRT (raw)..."
    curl -fsSL "$NETRON_RAW/$f" -o "$HERE/report_template/$f" \
      || { echo "error: failed to download $f" >&2; exit 2; }
  fi
done

echo "[perf] analyzing $PERF_DIR"
"$PY" "$HERE/scripts/analyze_trt_perf.py" "$PERF_DIR" \
  --model-name "$MODEL_NAME" --output "$STAGE"

echo "[perf] validating"
"$PY" "$HERE/scripts/validate_analyze_data.py" "$STAGE" --require-passed

echo "[perf] packaging report"
rm -rf "$REPORT"
"$PY" "$HERE/scripts/package_report.py" \
  --analyze-data "$STAGE" --report-dir "$REPORT"

echo
echo "[perf] report ready: $REPORT"
echo "  serve: python -m http.server 8765 --bind 127.0.0.1 --directory $REPORT"
echo "  open:  http://127.0.0.1:8765/"
