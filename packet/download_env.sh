#!/bin/bash
# Download all installation packages for DSB Docker build
# Run this script before `docker build`

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Downloading CUDA 12.8.0 (runfile) ==="
wget -c https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_570.86.10_linux.run \
    -O cuda_12.8.0_570.86.10_linux.run

echo "=== Downloading cuDNN 9.5.0 (Ubuntu 22.04 deb repo) ==="
wget -c https://developer.download.nvidia.com/compute/cudnn/9.5.0/local_installers/cudnn-local-repo-ubuntu2204-9.5.0.26_1.0-1_amd64.deb \
    -O cudnn-local-repo-ubuntu2204-9.5.0.26_1.0-1_amd64.deb

echo "=== Downloading TensorRT 10.9.0.34 (tarball) ==="
# Note: requires NVIDIA developer account login
# Download manually from:
# https://developer.nvidia.com/tensorrt/download/10x
# Save as: TensorRT-10.9.0.34.Linux.x86_64-gnu.cuda-12.8.tar.gz
wget -c https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.9.1/tars/TensorRT-10.9.0.34.Linux.x86_64-gnu.cuda-12.8.tar.gz \
    -O TensorRT-10.9.0.34.Linux.x86_64-gnu.cuda-12.8.tar.gz || {
    echo ""
    echo "WARNING: TensorRT download failed (may require login)."
    echo "Please download manually from:"
    echo "  https://developer.nvidia.com/tensorrt/download/10x"
    echo "And save as: TensorRT-10.9.0.34.Linux.x86_64-gnu.cuda-12.8.tar.gz"
    echo "in the packet/ directory."
}

echo ""
echo "=== Download complete ==="
echo "Packages in $(pwd):"
ls -lh *.run *.deb *.tar.gz 2>/dev/null || echo "(some packages missing)"
