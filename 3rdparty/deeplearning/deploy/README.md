# TRT 部署（deploy/）

Python 与 C++ 推理共用的部署目录。流程：**分模块导出 ONNX → trtexec 构建
engine（前置，Python/C++ 共用）→ 推理**。自定义 CUDA 算子（bev_pool /
体素化 / sparse conv）留在 torch runtime，不进入 ONNX。

```
deploy/
├── export.py        # 前置导出：ONNX 分模块导出 + trtexec 构建 engine（Python/C++ 共用）
├── python/          # Python 推理包（deploy.python：preprocess/inference/postprocess 三层）
├── cpp/             # C++ 推理（dsb_infer，同三层 + ops，见 cpp/README.md）
├── tools/           # export_cpp_input.py（C++ 输入导出）+ trt_perf（性能分析）
└── launch/          # 部署启动脚本（export/infer_trt/visualize_trt/perf_trt）
```

## 1. 前置导出（deploy/export.py）

模型按 CUDA-BEVFusion 布局分模块导出（det/seg 各 4 张图），一次完成
ONNX 导出 + engine 构建（`--onnx-only` 只导出 ONNX）：

| 图 | 输入 → 输出 | 说明 |
|---|---|---|
| `camera.backbone.onnx` | img `(1,6,3,256,704)` → feats `(1,6,256,32,88)` | Swin-T + LSS-FPN |
| `camera.vtransform.onnx` | feats (+depth) → depthnet `(1,6,D+C,32,88)` | det 含 dtransform |
| `fuser.onnx` | cam_bev + lidar_bev → decoder_feat `(1,512,180,180)` | fuser+SECOND+FPN |
| `head.bbox/map.onnx` | decoder_feat → 9 个 pred（含 query_labels）/ masks | 不解码，decode 留在 torch |

```bash
cd 3rdparty/deeplearning

# 导出 ONNX + 构建 TRT engine（--verify 用 onnxruntime 逐图校验；
# engine 构建只走 trtexec CLI：$TRTEXEC 或 PATH，pip-only tensorrt
# 无 trtexec 时设 TRTEXEC=/path/to/TensorRT-10.9.../bin/trtexec）
python deploy/export.py configs/det_convfuser.py ckpts/bevfusion-det-bf.pth \
    --task det --verify --fp16
```

产物：`outputs/onnx/*.onnx`、`outputs/trt/*.engine`。

## 2. Python TRT 推理（deploy/python/pipeline.py）

三层编排（Preprocessor → Inferencer → Postprocessor），直接用 engine，
不需要 `.pth`：

```bash
python deploy/python/pipeline.py configs/det_convfuser.py --task det \
    --engine_dir outputs/trt --sample-index 5 --dump outputs/trt/sample5.npz

# 与 torch 参考对照（需 checkpoint）
python deploy/python/pipeline.py configs/det_convfuser.py --checkpoint ckpts/bevfusion-det-bf.pth \
    --task det --engine_dir outputs/trt --verify
```

结果可视化（`outputs/vis/`，jpg + gif）：

```bash
bash deploy/launch/visualize_trt.sh --sample-index 5                  # 单帧：BEV + 6 相机
bash deploy/launch/visualize_trt.sh --sample-index 5 --frames 10 --fps 4  # 序列 gif
python deploy/python/visualize.py configs/det_convfuser.py \
    --npz outputs/trt/sample5.npz --out outputs/vis                   # 复用已 dump 结果
```

## 3. C++ 推理（deploy/cpp/）

纯 C++ 推理流水线（无 Python），结构与 python 三层对应（preprocess /
inference / postprocess + ops）；输入文件由 `deploy/tools/export_cpp_input.py`
导出。详见 [cpp/README.md](cpp/README.md)。

## 4. 启动脚本（deploy/launch/）

```bash
# 部署 launch/（cd 3rdparty/deeplearning）
bash deploy/launch/export.sh det              # 前置导出（ONNX + engine）
bash deploy/launch/infer_trt.sh det           # TRT 推理（--verify --checkpoint 透传）
bash deploy/launch/visualize_trt.sh det       # 推理结果可视化（jpg/gif -> outputs/vis/）
bash deploy/launch/perf_trt.sh det            # trtexec 逐层 profile -> outputs/perf/
```

模型库训练/评估脚本仍在 `launch/`（prepare_data/train/eval/visualize），
见上级目录 README。

## 5. 性能分析（deploy/tools/trt_perf/）

NVIDIA trt-perf-analysis 工具，生成逐层时延的交互式 HTML 报告。
详见 [tools/trt_perf/README.md](tools/trt_perf/README.md)：

```bash
bash deploy/launch/perf_trt.sh det            # -> outputs/perf/layers_*.json + profile_*.json
bash deploy/tools/trt_perf/run.sh             # -> outputs/perf/report/
```

## TRT 版本说明

TRT >= 10.12 / 11.x 强类型：精度由 ONNX 图表达（Cast/QDQ），
`BuilderFlag.FP16/INT8` 已移除；FP32 ONNX 构建 FP32 engine。需要混合
FP16 时先用 ModelOpt AutoCast（`pip install nvidia-modelopt`），或加
`--autocast-fp16`。TRT < 10.12（docker 目标 10.9.0.34）自动回退弱类型，
`--fp16` 照常生效（版本在构建时探测）。
