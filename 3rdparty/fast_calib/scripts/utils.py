import logging
import os
from collections import Counter

import yaml
import numpy as np
import open3d as o3d
from typing import Optional, Tuple

from mcap.reader import make_reader


def scan_mcap_channels(mcap_path) -> list:
    """扫描 mcap 全部 channel：(topic, schema_name, count)。不依赖解码器。
    优先用 summary（秒级）；无 summary 才全量遍历消息计数。"""
    cnt = Counter()
    names = {}
    with open(mcap_path, "rb") as f:
        reader = make_reader(f)
        summary = reader.get_summary()
        if summary is not None and summary.channels:
            for cid, ch in summary.channels.items():
                t = ch.topic
                cnt[t] = summary.statistics.channel_message_counts.get(cid, 0)
                names[t] = (summary.schemas[ch.schema_id].name
                            if ch.schema_id in summary.schemas else "?")
        else:
            for schema, channel, _ in reader.iter_messages():
                cnt[channel.topic] += 1
                names[channel.topic] = schema.name
    return [(t, names[t], cnt[t]) for t in cnt]


def setup_logger(log_path: str):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def load_yaml_config(yaml_path: str = "config.yaml") -> dict:
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def pointcloud2_to_xyzi(msg, point_field_datatype_to_np) -> np.ndarray:
    dt = np.dtype({
        "names": [f.name for f in msg.fields],
        "formats": [point_field_datatype_to_np[f.datatype] for f in msg.fields],
        "offsets": [f.offset for f in msg.fields],
        "itemsize": msg.point_step,
    })
    arr = np.frombuffer(bytes(msg.data), dtype=dt, count=msg.width * msg.height)
    inten = arr["intensity"] if "intensity" in arr.dtype.names else np.zeros(len(arr))
    ring = arr["ring"] if "ring" in arr.dtype.names else np.zeros(len(arr))
    xyzi = np.stack([arr["x"], arr["y"], arr["z"], inten, ring], axis=-1).astype(np.float32)
    return xyzi[np.isfinite(xyzi[:, :3]).all(axis=1)]


def msg_stamp_ns(ros_msg, msg) -> int:
    try:
        return ros_msg.header.stamp.sec * 10 ** 9 + ros_msg.header.stamp.nanosec
    except AttributeError:
        pass
    try:
        return ros_msg.timestamp.sec * 10 ** 9 + ros_msg.timestamp.nanosec
    except AttributeError:
        return msg.log_time


def save_T_yaml(path, T, extra=None):
    r = [f"{v:.9f}" for v in np.asarray(T[:3, :3], dtype=np.float64).ravel()]
    t = [f"{v:.9f}" for v in np.asarray(T[:3, 3], dtype=np.float64)]
    with open(path, "w", encoding="utf-8") as f:
        f.write("Rcl: [ " + ", ".join(r[0:3]) + ",\n"
                + "       " + ", ".join(r[3:6]) + ",\n"
                + "       " + ", ".join(r[6:9]) + " ]\n")
        f.write("Pcl: [ " + ", ".join(t) + " ]\n")
        if extra:
            for k, v in extra.items():
                f.write(f"{k}: {v}\n")


def save_point_cloud(pc_np: np.ndarray, out_path: str):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_np[:, 0:3])
    intensity = pc_np[:, 3]
    rgb = np.zeros((len(pc_np), 3), dtype=np.float32)
    rgb[:, 0] = intensity
    rgb[:, 1] = intensity
    rgb[:, 2] = intensity
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    o3d.io.write_point_cloud(out_path, pcd)


def colorize_by_holes(pc_xyz, holes3d, hole_radius, hole_palette):
    cols = np.full((len(pc_xyz), 3), 0.75)
    if holes3d is not None:
        for k, h in enumerate(holes3d[:4]):
            cols[np.linalg.norm(pc_xyz - h, axis=1) <= 1.3 * hole_radius] = hole_palette[k % 4]
    return cols


def circle_fit_2d(xy):
    A = np.column_stack([xy[:, 0], xy[:, 1], np.ones(len(xy))])
    b = (xy ** 2).sum(axis=1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0] / 2, sol[1] / 2
    return np.array([cx, cy]), float(np.sqrt(max(sol[2] + cx * cx + cy * cy, 0.0)))


def residual_3d(params, lidar_pts, cam_pts):
    from scipy.spatial.transform import Rotation
    R = Rotation.from_rotvec(params[:3]).as_matrix()
    return ((R @ lidar_pts.T).T + params[3:6] - cam_pts).ravel()
