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
# Livox PointCloud2 (xfer_format=0): x,y,z,intensity float32 + tag,line + timestamp float64 (ns)
LIVOX_FIELDS = [
    PointField("x", 0, PointField.FLOAT32, 1),
    PointField("y", 4, PointField.FLOAT32, 1),
    PointField("z", 8, PointField.FLOAT32, 1),
    PointField("intensity", 12, PointField.FLOAT32, 1),
    PointField("tag", 16, PointField.UINT8, 1),
    PointField("line", 17, PointField.UINT8, 1),
    PointField("timestamp", 18, PointField.FLOAT64, 1),
]
LIVOX_DTYPE = np.dtype([
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("intensity", "<f4"),
    ("tag", "u1"), ("line", "u1"), ("timestamp", "<f8"),
])

def livox_points(points):
    out = np.zeros(len(points), dtype=LIVOX_DTYPE)
    out["x"], out["y"], out["z"] = points[:, 0], points[:, 1], points[:, 2]
    out["intensity"] = points[:, 3]
    out["timestamp"] = points[:, 4].astype(np.float64) * 1e9
    return out


def load_tensor(path):
    with open(path, "rb") as f:
        head = np.fromfile(f, dtype="<i4", count=3)
        dims = tuple(int(v) for v in np.fromfile(f, dtype="<i4", count=int(head[1])))
        dt = {2: np.float16, 3: np.float32, 1: np.int32, 4: np.int64}.get(int(head[2]), np.float32)
        return np.fromfile(f, dtype=dt).reshape(dims)


class ExamplePublisher(Node):
    def __init__(self, data_dir, camera_rate, lidar_rate, livox_fields=False):
        super().__init__("example_publisher")
        self.images = [
            np.asarray(PILImage.open(os.path.join(data_dir, f)).convert("RGB"))
            for f, _ in CAMS
        ]
        self.points = load_tensor(os.path.join(data_dir, "points.tensor")).astype(np.float32)
        self.image_pubs = [
            self.create_publisher(Image, topic, 10) for _, topic in CAMS
        ]
        self.lidar_pub = self.create_publisher(PointCloud2, "/livox/lidar" if livox_fields else "/lidar/points", 10)
        self.livox_fields = livox_fields
        self.cam_timer = self.create_timer(1.0 / camera_rate, self.publish_cameras)
        self.lidar_timer = self.create_timer(1.0 / lidar_rate, self.publish_lidar)
        self.frame = 0
        self.get_logger().info(
            f"publishing {len(CAMS)} cameras @{camera_rate} Hz + "
            f"{len(self.points)} LiDAR points @{lidar_rate} Hz"
        )

    def publish_cameras(self):
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

    def publish_lidar(self):
        header = PointCloud2().header
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "lidar"
        if self.livox_fields:
            cloud = point_cloud2.create_cloud(header, LIVOX_FIELDS, livox_points(self.points))
        else:
            cloud = point_cloud2.create_cloud(header, FIELDS, self.points[:, :5])
        self.lidar_pub.publish(cloud)
        self.frame += 1
        if self.frame % 10 == 1:
            self.get_logger().info(f"lidar frame {self.frame}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default=os.environ.get(
            "DSB_EXAMPLE_DATA",
            os.path.join(os.path.dirname(__file__), "../../../3rdparty/deeplearning/outputs/example-data"),
        ),
    )
    parser.add_argument("--camera-rate", type=float, default=30.0)
    parser.add_argument("--lidar-rate", type=float, default=10.0)
    parser.add_argument("--livox-fields", action="store_true",
                        help="publish Livox-style PointCloud2 (x,y,z,intensity,tag,line,timestamp)")
    args = parser.parse_args()
    rclpy.init()
    node = ExamplePublisher(args.data_dir, args.camera_rate, args.lidar_rate,
                            args.livox_fields)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()


if __name__ == "__main__":
    main()
