"""Deploy python package: TRT 推理（预处理 / 推理 / 后处理三层）。

ONNX 导出与 engine 构建是 Python/C++ 推理共用的前置步骤，入口在
deploy/export.py（本包之上）；本包负责 engine 加载与三层推理编排。

Entry points (each has a CLI; run from the model repo 3rdparty/deeplearning):
  python deploy/export_bf.py  CONFIG CKPT [--verify]   # CUDA-BEVFusion 结构 ONNX 导出
  python deploy/python/scripts/pipeline.py  CONFIG --task det [--checkpoint] [--verify]
  python deploy/python/scripts/visualize.py CONFIG CKPT --engine_dir outputs/trt

三层架构（预处理 / 推理 / 后处理）：
  preprocess   Preprocessor: 原始样本 -> 模型输入（深度投影 / 体素化）
  inference    Inferencer:   TRT engine 编排 + torch 自定义算子 -> 预测
  postprocess  Postprocessor: 预测 -> 检测框（解码 + NMS）

其他模块：
  config       DeployConfig: shared paths / precision / behavior knobs
  graphs       GRAPH_IO + DET_HEAD_KEYS: single source of tensor names
  runtime      TrtEngine + load_engines: engine load / execute
  bev_pool_torch   camera_bev_from_depthnet (geometry + bev_pool)

编排入口（TrtPipeline 与可视化）在 scripts/，见 python/README.md。
"""

from deploy.python.bev_pool_torch import camera_bev_from_depthnet
from deploy.python.inference import Inferencer, sparse_encoder_forward
from deploy.python.postprocess import Postprocessor
from deploy.python.preprocess import Preprocessor, project_depth

__all__ = [
    "camera_bev_from_depthnet",
    "Inferencer",
    "sparse_encoder_forward",
    "Postprocessor",
    "Preprocessor",
    "project_depth",
]
