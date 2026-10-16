#!/bin/bash
# Visualization script for DSB models

set -e

SCRIPT_DIR="$(cd $(dirname $0) && pwd)"
PROJECT_ROOT="$(cd $SCRIPT_DIR/.. && pwd)"

CONFIG="${1:-configs/bevfusion.yaml}"
CKPT="${2:-work_dirs/latest.pth}"

echo "Visualizing with config: $CONFIG"
python scripts/visualize.py --config $CONFIG --checkpoint $CKPT
