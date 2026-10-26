# CUDA-BEVFusion 推理环境（DSB/3rdparty/deeplearning/deploy）
# 用法：. environment.sh（cwd 任意，BEVFUSION_DIR 以脚本位置定）
# 按实际安装路径修改下方 *_Root 变量。

# TensorRT（用户环境：/opt/TensorRT-10.9.0.34）
export TensorRT_Root=${TensorRT_Root:-/opt/TensorRT-10.9.0.34}
export TensorRT_Lib=${TensorRT_Root}/lib
export TensorRT_Inc=${TensorRT_Root}/include
export TensorRT_Bin=${TensorRT_Root}/bin

# CUDA（用户环境：/usr/local/cuda-12.8）
export CUDA_Root=${CUDA_Root:-/usr/local/cuda-12.8}
export CUDA_Lib=${CUDA_Root}/lib64
export CUDA_Inc=${CUDA_Root}/include
export CUDA_Bin=${CUDA_Root}/bin
export CUDA_HOME=${CUDA_Root}
export PATH=${CUDA_Bin}:${TensorRT_Bin}:$PATH
export LD_LIBRARY_PATH=${CUDA_Lib}:${TensorRT_Lib}:$LD_LIBRARY_PATH

# CUDNN
export CUDNN_Lib=${CUDNN_Root:-/usr/local/cuda}/lib64

# libspconv：CUDA 12.x 用 12.8（本仓库已含 libs/libspconv/lib/x86_64_cuda12.8）
export SPCONV_CUDA_VERSION=${SPCONV_CUDA_VERSION:-12.8}
export ConfigurationStatus=Success

# 模型/数据目录（相对 3rdparty/deeplearning）
export BEVFUSION_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export LD_LIBRARY_PATH=${BEVFUSION_DIR}/build:$LD_LIBRARY_PATH  # custom_layernorm.so（main dlopen）
export MODEL_DIR=${MODEL_DIR:-${BEVFUSION_DIR}/model}
export DATA_DIR=${DATA_DIR:-${BEVFUSION_DIR}/../outputs/example-data}

echo "[env] TensorRT=${TensorRT_Root} CUDA=${CUDA_Root} SPCONV=${SPCONV_CUDA_VERSION}"
echo "[env] model=${MODEL_DIR} data=${DATA_DIR}"
