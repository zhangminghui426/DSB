cmake_minimum_required(VERSION 3.18)
project(dsb_cpp_infer LANGUAGES CXX)

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

add_executable(dsb_infer
  src/main.cpp
  src/pipeline.cpp
  src/config.cpp
  src/preprocess.cpp
  src/inference.cpp
  src/postprocess.cpp
  src/ops.cpp
  src/trt_engine.cpp
)
target_include_directories(dsb_infer PRIVATE include ${TENSORRT_INCLUDE_DIR})
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
