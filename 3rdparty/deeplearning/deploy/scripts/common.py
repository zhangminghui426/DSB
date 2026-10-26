"""deploy scripts 共用：张量/框 IO 与四张部署图的输入输出名。"""

import os
import os.path as osp

import numpy as np

MAGIC = 0x33FF1101
DTYPE_TO_ID = {"float16": 2, "float32": 3, "int32": 1, "int64": 4}

# 4 个可部署图：ONNX/TRT 输入名 -> torch dump 名；输出名按引擎顺序
GRAPHS = ["camera.backbone", "camera.vtransform", "fuser", "head.bbox"]
FEEDS = {
    "camera.backbone": {"img": "img", "depth": "depth"},
    "camera.vtransform": {"feat_in": "bev360"},
    "fuser": {"camera": "cam_bev", "lidar": "lidar_bev"},
    "head.bbox": {"middle": "middle"},
}
OUTS = {
    "camera.backbone": ["camera_feature", "camera_depth_weights"],
    "camera.vtransform": ["cam_bev"],
    "fuser": ["middle"],
    "head.bbox": ["score", "rot", "dim", "reg", "height", "vel"],
}
STAGE_TENSORS = {
    "camera.backbone": ["camera_feature", "camera_depth_weights"],
    "camera.vtransform": ["cam_bev"],
    "fuser": ["middle"],
    "head.bbox": ["score", "rot", "dim", "reg", "height", "vel"],
}
CAM_NAMES = ["FRONT", "FRONT_RIGHT", "FRONT_LEFT", "BACK", "BACK_LEFT", "BACK_RIGHT"]


def save_tensor(arr, path):
    """写 save_tensor 格式（3xint32 header + dims + data）。"""
    os.makedirs(osp.dirname(path), exist_ok=True)
    arr = np.ascontiguousarray(arr)
    with open(path, "wb") as f:
        f.write(np.array([MAGIC, arr.ndim, DTYPE_TO_ID[arr.dtype.name]], dtype=np.int32).tobytes())
        f.write(np.array(arr.shape, dtype=np.int32).tobytes())
        f.write(arr.tobytes())


def load_tensor(path):
    """读 save_tensor 格式。"""
    with open(path, "rb") as f:
        head = np.fromfile(f, dtype="<i4", count=3)
        dims = tuple(int(v) for v in np.fromfile(f, dtype="<i4", count=int(head[1])))
        dt = {2: np.float16, 3: np.float32, 1: np.int32, 4: np.int64}.get(int(head[2]), np.float32)
        return np.fromfile(f, dtype=dt).reshape(dims)


def dump(name, arr, dump_dir, sub):
    """中间张量 -> <dump_dir>/<sub>/<name>.npy。"""
    os.makedirs(osp.join(dump_dir, sub), exist_ok=True)
    np.save(osp.join(dump_dir, sub, name + ".npy"), np.asarray(arr))


def load_dump(name, dump_dir, sub="torch"):
    return np.load(osp.join(dump_dir, sub, name + ".npy"))


def parse_cpp_boxes(path):
    """解析 C++ 打印的框文本 -> [N,9]（x,y,z,w,l,h,rot,score,id）。"""
    import re
    boxes = []
    for line in open(path):
        m = re.search(
            r"id=(\d+) score=([-\d.]+) pos=\(([-\d.]+),([-\d.]+),([-\d.]+)\) "
            r"size=\(([-\d.]+),([-\d.]+),([-\d.]+)\) rot=([-\d.]+)", line)
        if m:
            g = m.groups()
            boxes.append([float(g[i]) for i in (2, 3, 4, 5, 6, 7, 8, 1, 0)])
    return np.array(boxes, dtype=np.float32)
