"""Deploy package: split-module ONNX export -> trtexec engine build -> TRT infer.

Entry points (each has a CLI):
  python deploy/exporter.py   CONFIG CKPT --task det [--verify]   # ONNX export
  python deploy/builder.py    --task det [--fp16]                 # engine build
  python deploy/inferencer.py CONFIG CKPT --task det [--build]    # TRT inference

Modules:
  config       DeployConfig: shared paths / precision / behavior knobs
  graphs       GRAPH_IO + DET_HEAD_KEYS: single source of tensor names
  exporter     OnnxExporter: module-split ONNX export + onnxruntime verify
  builder      EngineBuilder: ONNX -> engine via the trtexec CLI only
  runtime      TrtEngine + load_engines: engine load / execute
  inferencer   DetInferencer: full TRT pipeline + decode + parity verify
  preprocess   project_depth (torch depth projection)
  bev_pool_torch   camera_bev_from_depthnet (geometry + bev_pool)
  lidar_branch     LidarBranchEncoder (voxelize + sparse encoder)
"""

from deploy.bev_pool_torch import camera_bev_from_depthnet
from deploy.lidar_branch import LidarBranchEncoder
from deploy.preprocess import project_depth

__all__ = [
    "camera_bev_from_depthnet",
    "LidarBranchEncoder",
    "project_depth",
]
