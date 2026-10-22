# TRT 部署（deploy/）

C++ 推理完全移植 NVIDIA CUDA-BEVFusion（deploy/cpp/，不再区分预处理/推理/后处理三层）。
Python 侧负责模型导出（deploy/export_bf.py）。

## 模型导出 + engine 构建（Python）

cd 3rdparty/deeplearning

python deploy/export_bf.py configs/det_convfuser.py ckpts/bevfusion-det-bf.pth \
    --outdir outputs/onnx_bf --engine-dir outputs/trt_bf --verify --fp16

产物：outputs/onnx_bf/ 下 5 个 onnx（camera.backbone / camera.vtransform / fuser /
head.bbox / lidar.backbone.xyz），outputs/trt_bf/ 下 4 个 .plan。

## 推理（C++）

模型按 CUDA-BEVFusion 布局放置（model/<name>/build/*.plan + lidar.backbone.xyz.onnx），
构建与运行见 cpp/README.md。

## 目录

- cpp/：C++ 推理（CUDA-BEVFusion 移植）
- python/：Python 导出/校验（export_bf.py、lean/ SCN 导出）
- tools/：输入导出（export_example_data.py 等）
- launch/：旧启动脚本（历史保留）
