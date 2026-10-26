"""deploy 工具共用：save_tensor/load_tensor 与 C++ 框文本解析。"""

import os
import os.path as osp

import numpy as np

MAGIC = 0x33FF1101
DTYPE_TO_ID = {"float16": 2, "float32": 3, "int32": 1, "int64": 4}


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
