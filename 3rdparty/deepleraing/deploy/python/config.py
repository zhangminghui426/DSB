"""Deployment configuration shared by export / build / infer entry points.

One dataclass carries every path and precision knob used by the deploy
package (deploy.export, deploy.builder, deploy.pipeline), so the three
CLIs agree on defaults and the runtime split stays consistent.
"""

import os.path as osp
from dataclasses import dataclass, field


@dataclass
class DeployConfig:
    """Paths, precision and behavior knobs for the TRT deployment chain."""

    task: str = "det"  # det | seg
    # directories
    onnx_dir: str = osp.join("outputs", "onnx")
    engine_dir: str = osp.join("outputs", "trt")
    # export
    opset: int = 17
    # build
    build: bool = False
    fp16: bool = False
    autocast_fp16: bool = False
    graphs: str = ""  # comma-separated graph names; empty = all for --task
    # infer
    sample_index: int = 0
    verify: bool = False
    dump: str = ""  # npz output path; empty = no dump
    # model config overrides
    options: list = field(default_factory=list)
