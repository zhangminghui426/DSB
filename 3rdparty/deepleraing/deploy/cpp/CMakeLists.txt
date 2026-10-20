cmake_minimum_required(VERSION 3.18)
project(dsb_infer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- TensorRT ---------------------------------------------------------
# Point TENSORRT_ROOT at the TRT install, e.g.
#   -DTENSORRT_ROOT=/opt/TensorRT-10.9.0.34
# or export TENSORRT_ROOT before configuring.
if(NOT DEFINED TENSORRT_ROOT)
  if(DEFINED ENV{TENSORRT_ROOT})
    set(TENSORRT_ROOT $ENV{TENSORRT_ROOT})
  elseif(EXISTS "/opt/TensorRT-10.9.0.34")
    set(TENSORRT_ROOT "/opt/TensorRT-10.9.0.34")
  elseif(EXISTS "/usr/local/tensorrt")
    set(TENSORRT_ROOT "/usr/local/tensorrt")
  endif()
endif()

find_path(TENSORRT_INCLUDE_DIR NvInfer.h PATHS ${TENSORRT_ROOT}/include)
find_library(TENSORRT_LIB nvinfer PATHS ${TENSORRT_ROOT}/lib ${TENSORRT_ROOT}/lib64)
if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIB)
  message(FATAL_ERROR "TensorRT not found. Set -DTENSORRT_ROOT=<trt-root>")
endif()

# ---- CUDA (used only for device buffers / stream in trt_engine) --------
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
  set(DSB_HAVE_CUDA TRUE)
else()
  set(DSB_HAVE_CUDA FALSE)
endif()

# ---- OpenCV (optional: decode raw jpg input; not required) -------------
find_package(OpenCV QUIET COMPONENTS core imgcodecs imgproc)
if(OpenCV_FOUND)
  set(DSB_HAVE_OPENCV TRUE)
else()
  set(DSB_HAVE_OPENCV FALSE)
endif()

# 三层结构（与 deploy/python 对应）：
#   preprocess/  预处理层（深度投影 / 体素化）
#   inference/   推理层（TRT engine 编排 + trt_engine）
#   ops/         自定义算子（CPU 实现，对应 3rdparty/deepleraing/ops 参考）
#   postprocess/ 后处理层（解码 + NMS）
#   pipeline/config/tensor/main 为编排与基础设施
add_executable(dsb_infer
  main.cpp
  pipeline.cpp
  config.cpp
  preprocess/preprocess.cpp
  inference/inference.cpp
  inference/trt_engine.cpp
  ops/ops.cpp
  postprocess/postprocess.cpp
)
target_include_directories(dsb_infer PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${TENSORRT_INCLUDE_DIR}
)
target_link_libraries(dsb_infer PRIVATE ${TENSORRT_LIB})
if(DSB_HAVE_CUDA)
  target_link_libraries(dsb_infer PRIVATE CUDA::cudart)
  target_compile_definitions(dsb_infer PRIVATE DSB_HAVE_CUDA=1)
endif()
if(DSB_HAVE_OPENCV)
  target_link_libraries(dsb_infer PRIVATE ${OpenCV_LIBS})
  target_compile_definitions(dsb_infer PRIVATE DSB_HAVE_OPENCV=1)
endif()
target_compile_options(dsb_infer PRIVATE -O2 -Wall)

message(STATUS "TensorRT: ${TENSORRT_LIB}")
message(STATUS "CUDA: ${DSB_HAVE_CUDA}  OpenCV: ${DSB_HAVE_OPENCV}")
