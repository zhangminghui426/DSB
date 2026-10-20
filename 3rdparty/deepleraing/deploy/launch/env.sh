# Sourced by the .sh wrappers (not executed). All values overridable via env.
PY="${PY:-/home/wika/miniconda3/envs/bevfusion/bin/python}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export PATH="$CUDA_HOME/bin:$PATH"
TORCHRUN="${TORCHRUN:-$(dirname "$PY")/torchrun}"
