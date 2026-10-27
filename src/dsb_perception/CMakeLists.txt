cmake_minimum_required(VERSION 3.16)
project(dsb_perception)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(CUDAToolkit REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/BoundingBox.msg"
  "msg/BoundingBoxArray.msg"
  DEPENDENCIES std_msgs
)

# CUDA-BEVFusion core built by 3rdparty/deeplearning/deploy (run_trt.sh).
set(DEPLOY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/deeplearning/deploy" CACHE PATH
    "Path to deploy/ (contains build/libbevfusion_core.so)")
set(BEVFUSION_CORE "${DEPLOY_DIR}/build/libbevfusion_core.so")
if(NOT EXISTS "${BEVFUSION_CORE}")
  message(FATAL_ERROR
    "libbevfusion_core.so not found: ${BEVFUSION_CORE}\n"
    "Build it first: cd 3rdparty/deeplearning/deploy && bash run_trt.sh")
endif()

add_executable(bevfusion_node src/bevfusion_node.cpp)
target_include_directories(bevfusion_node PRIVATE
  "${DEPLOY_DIR}/src"
  ${CUDA_INCLUDE_DIRS}
)
target_link_libraries(bevfusion_node
  "${BEVFUSION_CORE}"
  rclcpp::rclcpp
  ${sensor_msgs_TARGETS}
  ${std_msgs_TARGETS}
  ${visualization_msgs_TARGETS}
  CUDA::cudart
)
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} rosidl_typesupport_cpp)
target_link_libraries(bevfusion_node "${cpp_typesupport_target}")

target_compile_definitions(bevfusion_node PRIVATE
  DSB_DEPLOY_DIR_DEFAULT="${DEPLOY_DIR}"
)

set(_rpath "${DEPLOY_DIR}/build;${DEPLOY_DIR}/libs/libspconv/lib/x86_64_cuda12.8")
set_target_properties(bevfusion_node PROPERTIES
  BUILD_RPATH "${_rpath}"
  INSTALL_RPATH "${_rpath}"
)

install(TARGETS bevfusion_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config launch scripts DESTINATION share/${PROJECT_NAME})

ament_export_dependencies(rosidl_default_runtime)
ament_package()
