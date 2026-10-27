#!/bin/bash
# Publish Livox MID-360 as sensor_msgs/PointCloud2 on /livox/lidar.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
source /opt/ros/jazzy/setup.bash
source "$HERE/install/setup.bash"
CFG="$HERE/src/livox_ros_driver2/config/MID360s_config.json"
exec ros2 run livox_ros_driver2 livox_ros_driver2_node --ros-args \
  -p xfer_format:=0 -p multi_topic:=0 -p data_src:=0 -p publish_freq:=10.0 \
  -p output_data_type:=0 -p frame_id:=livox_frame \
  -p user_config_path:="$CFG"
