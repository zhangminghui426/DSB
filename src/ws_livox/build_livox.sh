#!/bin/bash
# Build livox_ros_driver2 with the system Python (ROS Jazzy), not the conda env.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
source /opt/ros/jazzy/setup.bash
cd "$HERE/src/livox_ros_driver2"
env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u CONDA_PROMPT_MODIFIER \
    -u CONDA_SHLVL -u CONDA_PYTHON_EXE \
    PATH=/usr/bin:/opt/ros/jazzy/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    ./build.sh jazzy
