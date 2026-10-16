#!/bin/bash
# Environment setup for DSB (bevfusion_flashocc)

# Add project root to PYTHONPATH
export PYTHONPATH="$(cd $(dirname $0)/.. && pwd):$PYTHONPATH"

# Default dataset path
export DATA_ROOT="${DATA_ROOT:-/data/nuscenes}"

# CUDA settings
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "DSB environment configured."
echo "  PYTHONPATH=$PYTHONPATH"
echo "  DATA_ROOT=$DATA_ROOT"
