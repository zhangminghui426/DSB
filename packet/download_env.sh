#!/bin/bash
# Download installation packages for DSB Docker build
# Run this script before `docker build`
#
# Note: TensorRT requires manual download (NVIDIA developer account login).
# See instructions at the end of this script.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Downloading CUDA 12.8.2 (runfile) ==="
wget -c https://developer.download.nvidia.com/compute/cuda/12.8.2/local_installers/cuda_12.8.2_570.211.01_linux.run \
    -O cuda_12.8.2_570.211.01_linux.run

echo "=== Downloading cuDNN 9.26.1 (Ubuntu 22.04 deb repo) ==="
wget -c https://developer.download.nvidia.com/compute/cudnn/9.26.1/local_installers/cudnn-local-repo-ubuntu2204-9.26.1_1.0-1_amd64.deb \
    -O cudnn-local-repo-ubuntu2204-9.26.1_1.0-1_amd64.deb

echo ""
echo "=========================================="
echo "=== MANUAL DOWNLOAD REQUIRED: TensorRT ==="
echo "=========================================="
echo ""
echo "TensorRT requires NVIDIA developer account login."
echo "Please download manually:"
echo ""
echo "  URL:  https://developer.nvidia.com/tensorrt/download/10x"
echo "  Step:  1. Login with NVIDIA developer account"
echo "         2. Select TensorRT 10.9 GA"
echo "         3. Download Linux x86_64 / CUDA 12.x tarball"
echo ""
echo "  Save as: $(pwd)/TensorRT-10.9.0.34.Linux.x86_64-gnu.cuda-12.8.tar.gz"
echo ""

echo "=== Download complete (CUDA + cuDNN) ==="
echo "Packages in $(pwd):"
ls -lh *.run *.deb *.tar.gz 2>/dev/null || echo "(TensorRT tarball missing - download manually)"
