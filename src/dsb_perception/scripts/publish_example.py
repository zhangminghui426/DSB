#!/usr/bin/env python3
"""Publish outputs/example-data as the topics consumed by bevfusion_node."""

import argparse
import os

import numpy as np
import rclpy
from PIL import Image as PILImage
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from sensor_msgs_py import point_cloud2

CAMS = [
    ("0-FRONT.jpg", "/camera/front/image_raw"),
    ("1-FRONT_RIGHT.jpg", "/camera/front_right/image_raw"),
    ("2-FRONT_LEFT.jpg", "/camera/front_left/image_raw"),
    ("3-BACK.jpg", "/camera/back/image_raw"),
    ("4-BACK_LEFT.jpg", "/camera/back_left/image_raw"),
    ("5-BACK_RIGHT.jpg", "/camera/back_right/image_raw"),
]
FIELDS = [
    PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
    for i, n in enumerate(("x", "y", "z", "intensity", "time"))
]


def load_tensor(path):
    with open(path, "rb") as f:
        head = np.fromfile(f, dtype="<i4", count=3)
        dims = tuple(int(v) for v in np.fromfile(f, dtype="<i4", count=int(head[1])))
        dt = {2: np.float16, 3: np.float32, 1: np.int32, 4: np.int64}.get(int(head[2]), np.float32)
        return np.fromfile(f, dtype=dt).reshape(dims)


class ExamplePublisher(Node):
    def __init__(self, data_dir, rate):
        super().__init__("example_publisher")
        self.images = [
            np.asarray(PILImage.open(os.path.join(data_dir, f)).convert("RGB"))
            for f, _ in CAMS
        ]
        self.points = load_tensor(os.path.join(data_dir, "points.tensor")).astype(np.float32)
        self.image_pubs = [
            self.create_publisher(Image, topic, 10) for _, topic in CAMS
        ]
        self.lidar_pub = self.create_publisher(PointCloud2, "/lidar/points", 10)
        self.timer = self.create_timer(1.0 / rate, self.publish)
        self.frame = 0
        self.get_logger().info(
            f"publishing {len(CAMS)} cameras + {len(self.points)} LiDAR points"
        )

    def publish(self):
        stamp = self.get_clock().now().to_msg()
        for pub, img in zip(self.image_pubs, self.images):
            msg = Image()
            msg.header.stamp = stamp
            msg.header.frame_id = "camera"
            msg.height, msg.width = img.shape[:2]
            msg.encoding = "rgb8"
            msg.is_bigendian = 0
            msg.step = msg.width * 3
            msg.data = img.tobytes()
            pub.publish(msg)

        header = PointCloud2().header
        header.stamp = stamp
        header.frame_id = "lidar"
        cloud = point_cloud2.create_cloud(header, FIELDS, self.points[:, :5])
        self.lidar_pub.publish(cloud)
        self.frame += 1
        if self.frame % 10 == 1:
            self.get_logger().info(f"frame {self.frame}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default=os.environ.get(
            "DSB_EXAMPLE_DATA",
            os.path.join(os.path.dirname(__file__), "../../../3rdparty/deeplearning/outputs/example-data"),
        ),
    )
    parser.add_argument("--rate", type=float, default=10.0)
    args = parser.parse_args()
    rclpy.init()
    node = ExamplePublisher(args.data_dir, args.rate)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()


if __name__ == "__main__":
    main()
