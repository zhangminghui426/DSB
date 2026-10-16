#!/bin/bash
# Training script for DSB models

set -e

SCRIPT_DIR="$(cd $(dirname $0) && pwd)"
PROJECT_ROOT="$(cd $SCRIPT_DIR/.. && pwd)"

CONFIG="${1:-configs/bevfusion.yaml}"
GPUS="${2:-1}"

echo "Training with config: $CONFIG"
python scripts/train.py --config $CONFIG --gpus $GPUS
