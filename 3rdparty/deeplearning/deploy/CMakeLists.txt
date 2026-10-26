cmake_minimum_required(VERSION 3.18)
project(bevfusion LANGUAGES CXX)

# =====================================================================
# CUDA-BEVFusion 推理代码移植（NVIDIA-AI-IOT/CUDA-BEVFusion 原样构建）。
# 依赖：CUDA >= 11 / TensorRT >= 8.5 / libprotobuf-dev / libspconv。
# 配置：TENSORRT_ROOT、CUDA_ROOT、CUDNN_ROOT（或直接环境变量），
#       SPCONV_CUDA_VERSION（默认 12.8，对应 libs/libspconv/lib/x86_64_cuda12.8）。
# 运行：
#   cd 3rdparty/deeplearning/deploy && mkdir -p build && cd build
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
  ${BEVFUSION_ROOT}/libs/stb
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

# ---- CUDA host compiler：必须位于 enable_language(CUDA) 之前 -------------
# nvcc 与过新的 GCC（14+ 的 libstdc++ _Float128 模板）不兼容会报
# "math.h:1418 error: template argument 1 is invalid"。显式选一个 nvcc
# 支持的 g++（CUDA 12.8 官方支持 GCC 上限 13），找不到则保持默认。
if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
  set(_cuda_host_candidates g++-13 g++-12 g++-11 g++-10)
  foreach(_h IN LISTS _cuda_host_candidates)
    find_program(_host_cxx NAMES ${_h})
    if(_host_cxx)
      message(STATUS "CUDA host compiler: ${_host_cxx}")
      set(CMAKE_CUDA_HOST_COMPILER ${_host_cxx})
      break()
    endif()
    unset(_host_cxx CACHE)
  endforeach()
endif()

# CUDA 编译器必须与 CUDA_ROOT 匹配：/usr/bin/nvcc 可能是旧版本
# （本机为 CUDA 12.0），会缺少 sm_120 等新架构支持且与 12.8 运行库混用。
# 已存在的 build/ 缓存不会自动更换编译器，需删除 build/ 后重新 configure。
if(NOT DEFINED CMAKE_CUDA_COMPILER AND EXISTS "${CUDA_ROOT}/bin/nvcc")
  set(CMAKE_CUDA_COMPILER "${CUDA_ROOT}/bin/nvcc" CACHE FILEPATH "CUDA compiler")
endif()
# 关闭 CMake 自动追加的默认架构（如 compute_52），架构统一由
# BEVFUSION_CUDA_ARCHS 的 -gencode 控制。
set(CMAKE_CUDA_ARCHITECTURES OFF)

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-std=c++17 -O3 -DENABLE_TEXT_BACKEND_STB -Wno-deprecated-declarations -Wno-unused-parameter")
message(STATUS "CUDA compiler: ${CMAKE_CUDA_COMPILER} (${CMAKE_CUDA_COMPILER_VERSION})")

find_package(Protobuf REQUIRED)

# SM 架构：80/86/89/90 通用；CUDA >= 12.8 追加 Blackwell sm_120（RTX 50 系）。
# 不追加时自定义 kernel 在 sm_120 上会报
# "no kernel image is available for execution on the device"。
set(BEVFUSION_CUDA_ARCHS 80 86 89 90)
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
  list(APPEND BEVFUSION_CUDA_ARCHS 120)
endif()
message(STATUS "CUDA archs: ${BEVFUSION_CUDA_ARCHS}")

function(bevfusion_cuda_target name type)
  if(type STREQUAL "EXECUTABLE")
    add_executable(${name} ${ARGN})
  else()
    add_library(${name} ${type} ${ARGN})
  endif()
  set_target_properties(${name} PROPERTIES CUDA_STANDARD 17)
  foreach(_arch IN LISTS BEVFUSION_CUDA_ARCHS)
    target_compile_options(${name} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_${_arch},code=sm_${_arch}>)
  endforeach()
endfunction()

####################### bevfusion_core ##########################
bevfusion_cuda_target(bevfusion_core SHARED ${CORE_FILES})
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
bevfusion_cuda_target(bevfusion EXECUTABLE src/main.cpp ${VISUALIZE_FILES})
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
bevfusion_cuda_target(custom_layernorm SHARED src/plugins/custom_layernorm.cu)
target_link_libraries(custom_layernorm libnvinfer.so libnvinfer_plugin.so)
