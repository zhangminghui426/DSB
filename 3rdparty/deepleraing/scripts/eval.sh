#!/bin/bash
# Evaluation script for DSB models

set -e

SCRIPT_DIR="$(cd $(dirname $0) && pwd)"
PROJECT_ROOT="$(cd $SCRIPT_DIR/.. && pwd)"

CONFIG="${1:-configs/bevfusion.yaml}"
CKPT="${2:-work_dirs/latest.pth}"

echo "Evaluating with config: $CONFIG"
python scripts/test.py --config $CONFIG --checkpoint $CKPT
