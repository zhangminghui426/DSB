"""PCD 文件读取工具。

支持 binary 格式 PCD，字段包含 x y z rgb。
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


def read_pcd(path: str | Path) -> np.ndarray:
    """读取 binary PCD 文件，返回结构化数组 (x, y, z, rgb)。

    返回 shape 为 (N,) 的 numpy 结构化数组，可通过 ['x'] / ['y'] / ['z'] / ['rgb'] 访问。
    """
    path = Path(path)
    with open(path, "rb") as f:
        header_lines: list[bytes] = []
        while True:
            line = f.readline()
            header_lines.append(line)
            if line.startswith(b"DATA"):
                break

    header = b"".join(header_lines).decode("ascii", errors="replace")

    fields: list[str] = []
    sizes: list[int] = []
    types: list[str] = []
    counts: list[int] = []
    points = 0
    data_type = "binary"

    for line in header.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        key = parts[0]
        if key == "FIELDS":
            fields = parts[1:]
        elif key == "SIZE":
            sizes = [int(x) for x in parts[1:]]
        elif key == "TYPE":
            types = parts[1:]
        elif key == "COUNT":
            counts = [int(x) for x in parts[1:]]
        elif key == "POINTS":
            points = int(parts[1])
        elif key == "DATA":
            data_type = parts[1]

    if data_type != "binary":
        raise ValueError(f"仅支持 binary PCD, 当前为: {data_type}")

    # 构建 numpy dtype
    dtype_parts: list[tuple[str, str]] = []
    for i, fname in enumerate(fields):
        t = types[i]
        s = sizes[i]
        if t == "F" and s == 4:
            dtype_parts.append((fname, "f4"))
        elif t == "F" and s == 8:
            dtype_parts.append((fname, "f8"))
        elif t == "U" and s == 1:
            dtype_parts.append((fname, "u1"))
        elif t == "U" and s == 4:
            dtype_parts.append((fname, "u4"))
        elif t == "I" and s == 4:
            dtype_parts.append((fname, "i4"))
        else:
            raise ValueError(f"不支持的字段类型: {t} size={s}")

    dt = np.dtype(dtype_parts)
    with open(path, "rb") as f:
        # 跳过 header
        for _ in header_lines:
            f.readline()
        raw = f.read()
    arr = np.frombuffer(raw, dtype=dt, count=points)
    return arr


def read_pcd_xyz(path: str | Path) -> np.ndarray:
    """读取 PCD 并返回 (N, 3) 的 xyz 数组。"""
    arr = read_pcd(path)
    return np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float32)
