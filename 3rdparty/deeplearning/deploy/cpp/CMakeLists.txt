cmake_minimum_required(VERSION 3.18)
project(bevfusion LANGUAGES CXX)

# =====================================================================
# CUDA-BEVFusion 推理代码移植（NVIDIA-AI-IOT/CUDA-BEVFusion 原样构建）。
# 依赖：CUDA >= 11 / TensorRT >= 8.5 / libprotobuf-dev / libspconv。
# 配置：TENSORRT_ROOT、CUDA_ROOT、CUDNN_ROOT（或直接环境变量），
#       SPCONV_CUDA_VERSION（默认 12.8，对应 libs/libspconv/lib/x86_64_cuda12.8）。
# 运行：
#   cd 3rdparty/deeplearning/deploy/cpp && mkdir -p build && cd build
#   cmake .. && make -j$(nproc)
# 产物：build/bevfusion + build/custom_layernorm.so（TRT plugin，main 内 dlopen）
# =====================================================================

set(CMAKE_BUILD_TYPE "Release")
set(arch ${CMAKE_HOST_SYSTEM_PROCESSOR})
if(NOT DEFINED SPCONV_CUDA_VERSION)
  set(SPCONV_CUDA_VERSION $ENV{SPCONV_CUDA_VERSION})
endif()
if(NOT SPCONV_CUDA_VERSION)
  set(SPCONV_CUDA_VERSION "12.8")
endif()

# ---- 根路径 ----------------------------------------------------------
if(NOT DEFINED TENSORRT_ROOT)
  if(DEFINED ENV{TENSORRT_ROOT})
    set(TENSORRT_ROOT $ENV{TENSORRT_ROOT})
  elseif(EXISTS "/opt/TensorRT-10.9.0.34")
    set(TENSORRT_ROOT "/opt/TensorRT-10.9.0.34")
  endif()
endif()
if(NOT DEFINED CUDA_ROOT)
  if(DEFINED ENV{CUDA_HOME})
    set(CUDA_ROOT $ENV{CUDA_HOME})
  elseif(EXISTS "/usr/local/cuda-12.8")
    set(CUDA_ROOT "/usr/local/cuda-12.8")
  elseif(EXISTS "/usr/local/cuda")
    set(CUDA_ROOT "/usr/local/cuda")
  endif()
endif()

# ---- 依赖库（相对本仓库） --------------------------------------------
set(BEVFUSION_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(spconv_include ${BEVFUSION_ROOT}/libs/libspconv/include)
set(spconv_lib ${BEVFUSION_ROOT}/libs/libspconv/lib/${arch}_cuda${SPCONV_CUDA_VERSION})

message(STATUS "TensorRT: ${TENSORRT_ROOT}")
message(STATUS "CUDA:     ${CUDA_ROOT}")
message(STATUS "libspconv: ${spconv_lib}")

include_directories(
  ${BEVFUSION_ROOT}/src
  ${BEVFUSION_ROOT}/src/common
  ${BEVFUSION_ROOT}/dependencies/stb
  ${BEVFUSION_ROOT}/dependencies/dlpack/include
  ${BEVFUSION_ROOT}/libs/cuosd
  ${spconv_include}
  ${TENSORRT_ROOT}/include
  ${CUDA_ROOT}/include
)

link_directories(
  ${TENSORRT_ROOT}/lib
  ${CUDA_ROOT}/lib64
  ${spconv_lib}
)

file(GLOB_RECURSE CORE_FILES
  ${BEVFUSION_ROOT}/src/bevfusion/*.cu
  ${BEVFUSION_ROOT}/src/bevfusion/*.cpp
  ${BEVFUSION_ROOT}/src/onnx/*.cpp
  ${BEVFUSION_ROOT}/src/common/tensor.cu
  ${BEVFUSION_ROOT}/src/common/tensorrt.cpp
)
file(GLOB_RECURSE VISUALIZE_FILES
  ${BEVFUSION_ROOT}/libs/cuosd/*.cpp
  ${BEVFUSION_ROOT}/libs/cuosd/*.cu
  ${BEVFUSION_ROOT}/src/common/visualize.cu
)

enable_language(CUDA)
find_package(CUDA REQUIRED)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-std=c++17 -O3 -DENABLE_TEXT_BACKEND_STB -Wno-deprecated-declarations -Wno-unused-parameter")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_90,code=sm_90)

find_package(Protobuf REQUIRED)

####################### bevfusion_core ##########################
cuda_add_library(bevfusion_core SHARED ${CORE_FILES})
target_link_libraries(bevfusion_core
  libcudart.so
  libcublasLt.so
  libnvinfer.so
  libnvinfer_plugin.so
  libspconv.so
  ${Protobuf_LIBRARIES}
)
set_target_properties(bevfusion_core PROPERTIES
  BUILD_RPATH ${spconv_lib})

####################### bevfusion ###############################
cuda_add_executable(bevfusion src/main.cpp ${VISUALIZE_FILES})
target_link_libraries(bevfusion
  bevfusion_core
  libnvinfer.so
  libspconv.so
  ${Protobuf_LIBRARIES}
)
set_target_properties(bevfusion PROPERTIES
  BUILD_RPATH ${spconv_lib}
  INSTALL_RPATH ${spconv_lib})

####################### custom_layernorm.so ######################
cuda_add_library(custom_layernorm SHARED src/plugins/custom_layernorm.cu)
target_link_libraries(custom_layernorm libnvinfer.so libnvinfer_plugin.so)
