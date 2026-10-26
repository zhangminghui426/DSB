"""Deploy scripts 包：保留导出链依赖的 tensor 名常量（graphs）。

部署只走 C++（deploy/src/）；export.py / tool/model_diff.py 依赖本包提供
DET_HEAD_KEYS / GRAPH_IO。
"""

from deploy.scripts.graphs import DET_HEAD_KEYS, GRAPH_IO  # noqa: E402,F401

__all__ = ["DET_HEAD_KEYS", "GRAPH_IO"]
