"""bev_pool_v2 — pillar-based BEV pooling CUDA op (LSS view transform).

Ported from the FlashOCC plugin (Phigent Robotics, arXiv:2211.17111); the only
BEV pool in this repo. fp32-only kernel; built by the top-level setup.py.
"""

import torch

from . import bev_pool_v2_ext
from .bev_pool import (bev_pool_plugin_fn, bev_pool_v2, bev_pool_v2_reference,
                       bev_pool_v2_sum, scatter_pool)

__all__ = ["bev_pool_v2", "bev_pool_v2_sum", "bev_pool_v2_reference",
           "scatter_pool", "bev_pool_plugin_fn"]
