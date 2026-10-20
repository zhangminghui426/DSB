"""Deploy package: split-module ONNX export -> trtexec engine build -> TRT infer.

Entry points (each has a CLI; run from the model repo 3rdparty/deeplearning):
  python deploy/python/export.py      CONFIG CKPT --task det [--verify]
  python deploy/python/builder.py     --task det [--fp16]
  python deploy/python/pipeline.py    CONFIG CKPT --task det [--build]

三层架构（预处理 / 推理 / 后处理）：
  preprocess   Preprocessor: 原始样本 -> 模型输入（深度投影 / 体素化）
  inference    Inferencer:   TRT engine 编排 + torch 自定义算子 -> 预测
  postprocess  Postprocessor: 预测 -> 检测框（解码 + NMS）
  pipeline     TrtPipeline:  三层编排入口（+ verify / dump）

其他模块：
  config       DeployConfig: shared paths / precision / behavior knobs
  graphs       GRAPH_IO + DET_HEAD_KEYS: single source of tensor names
  export       OnnxExporter: module-split ONNX export + trtexec engine build
  builder      EngineBuilder: ONNX -> engine via the trtexec CLI only
  runtime      TrtEngine + load_engines: engine load / execute
  bev_pool_torch   camera_bev_from_depthnet (geometry + bev_pool)
"""

from deploy.python.bev_pool_torch import camera_bev_from_depthnet
from deploy.python.inference import Inferencer, sparse_encoder_forward
from deploy.python.pipeline import TrtPipeline
from deploy.python.postprocess import Postprocessor
from deploy.python.preprocess import Preprocessor, project_depth

__all__ = [
    "camera_bev_from_depthnet",
    "Inferencer",
    "sparse_encoder_forward",
    "TrtPipeline",
    "Postprocessor",
    "Preprocessor",
    "project_depth",
]
