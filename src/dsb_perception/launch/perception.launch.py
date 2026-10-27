import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    cfg = os.path.join(
        get_package_share_directory("dsb_perception"), "config", "perception.yaml"
    )
    return LaunchDescription(
        [
            Node(
                package="dsb_perception",
                executable="bevfusion_node",
                name="bevfusion_node",
                output="screen",
                parameters=[cfg],
            )
        ]
    )
