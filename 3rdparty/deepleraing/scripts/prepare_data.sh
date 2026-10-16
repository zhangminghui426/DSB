#!/bin/bash
# Prepare nuScenes data (info files, GT database, etc.)

set -e

SCRIPT_DIR="$(cd $(dirname $0) && pwd)"
PROJECT_ROOT="$(cd $SCRIPT_DIR/.. && pwd)"

echo "Preparing nuScenes data..."
python -m datasets.tools.create_data --root $DATA_ROOT
echo "Data preparation complete."
