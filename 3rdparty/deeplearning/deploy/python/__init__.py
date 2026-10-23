"""Deploy python 包：保留导出链依赖的 tensor 名常量（graphs）。

Python 部署（bf 包 / scripts / 三层推理）已移除，部署只走 C++（deploy/cpp/）。
export.py / export_bf.py 仍依赖本包提供 DET_HEAD_KEYS / GRAPH_IO。
"""

from deploy.python.graphs import DET_HEAD_KEYS, GRAPH_IO  # noqa: E402,F401

__all__ = ["DET_HEAD_KEYS", "GRAPH_IO"]
