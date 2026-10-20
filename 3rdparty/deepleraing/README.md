# DeepLearing（BEVFusion + FlashOCC 重写）

纯 PyTorch 统一感知模型库，零 mmcv 依赖。覆盖：

- BEV 3D 目标检测（BEVFusion det，ConvFuser 融合）
- BEV 语义分割（`seg_fusion_bev256d2_lss`）
- 占据网络预测（FlashOCC / Stereo4D）
- 标定板自动检测（SECOND，见 `models/calib_board_detector/`）

## 目录结构

```
deepleraing/
├── models/            # 模型定义
│   ├── backbones/     # swin, resnet, second, sparse_encoder
│   ├── necks/         # glss, second, custom_fpn
│   ├── heads/         # transfusion, segm, occ_head
│   ├── vtransforms/   # lss, depth_lss, occ_lss
│   ├── fusers/        # conv fuser
│   ├── perception.py  # 统一顶层模型
│   └── calib_board_detector/  # 标定板检测器（SECOND）
├── datasets/          # 数据集与数据处理
├── scripts/           # 训练/评估/测试脚本（仅 .py）
├── deploy/            # 导出/构建/推理包（export/builder/runtime/config/graphs + preprocess/inference/postprocess/pipeline）
├── launch/            # shell 启动脚本（train/eval/export/infer_trt/perf_trt/visualize）
├── configs/           # 训练与部署配置
├── ops/               # 自定义 CUDA 算子（bev_pool_v2 / voxel / iou3d）
├── tools/             # 数据准备与输入导出工具
├── docs/              # 文档（MODEL_IO.md 等）
├── tests/             # 单元测试
└── setup.py           # CUDA 算子编译入口
```

## 环境与编译

```bash
conda create -n DSB python=3.10 -y
conda activate DSB
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# 编译 CUDA 算子（bev_pool_v2 / voxel / iou3d）
python setup.py build_ext --inplace
```

## 数据准备

```bash
# 软链接 nuScenes 数据目录（配置中 dataset_root="data/nuscenes/"，
# 需要 nuscenes_infos_train.pkl / nuscenes_infos_val.pkl / nuscenes_dbinfos_train.pkl）
ln -s /path/to/nuscenes data/nuscenes
```

## 训练 / 评估

```bash
# 训练（单卡；多卡用 torchrun --nproc_per_node=N，官方 recipe batch 4/卡）
python scripts/train.py configs/det_convfuser.py --load_from ckpts/bevfusion-det-bf.pth

# 评估
python scripts/test.py configs/det_convfuser.py ckpts/bevfusion-det-bf.pth --eval bbox
```

常用训练参数：`--batch_size N`、`--epochs N`、`--no_amp`、`--no_eval`、
`--options key=value`（点号式配置覆盖，如 `--options optimizer.lr=2e-4`）。

## 部署（ONNX 分模块导出 + TensorRT 推理）

模型按 CUDA-BEVFusion 布局**分模块导出**（det/seg 各 4 张图），自定义 CUDA
算子（bev_pool / 体素化 / sparse conv）留在 torch runtime，不进入 ONNX：

| 图 | 输入 → 输出 | 说明 |
|---|---|---|
| `camera.backbone.onnx` | img `(1,6,3,256,704)` → feats `(1,6,256,32,88)` | Swin-T + LSS-FPN |
| `camera.vtransform.onnx` | feats (+depth) → depthnet `(1,6,D+C,32,88)` | det 含 dtransform |
| `fuser.onnx` | cam_bev + lidar_bev → decoder_feat `(1,512,180,180)` | fuser+SECOND+FPN |
| `head.bbox/map.onnx` | decoder_feat → 9 个 pred（含 query_labels）/ masks | 不解码，decode 留在 torch |

```bash
# 1. 导出 ONNX 并构建 TRT engine（一步完成；--verify 用 onnxruntime 逐图校验，
#    需 pip install onnx onnxruntime；engine 构建只走 trtexec CLI，$TRTEXEC 或
#    PATH，pip-only tensorrt 无 trtexec 时需 TRTEXEC=/path/to/TensorRT-10.9.../bin/trtexec）
python deploy/export.py configs/det_convfuser.py ckpts/bevfusion-det-bf.pth \
    --task det --verify --fp16

# 2. 保存解码结果
python deploy/pipeline.py configs/det_convfuser.py ckpts/bevfusion-det-bf.pth \
    --engine_dir outputs/trt --sample-index 5 --dump outputs/trt/sample5.npz
```

launch 启动脚本（`launch/`，与 `scripts/` 的 .py 分离）：

```bash
launch/prepare_data.sh            # 数据软链接 + infos pkl 生成
launch/train.sh det               # 训练（det|seg|multitask|occ|m0|stereo|stbase）
launch/eval.sh det                # 评估
launch/visualize.sh det           # 可视化
launch/export.sh det              # 导出 ONNX
launch/infer_trt.sh det           # TRT 推理（加 --build --fp16 --verify 透传）
launch/visualize_trt.sh det       # TRT 推理结果可视化（jpg/gif -> outputs/vis/）
launch/perf_trt.sh det            # trtexec 采集逐层 profile -> outputs/perf/
```

性能分析（NVIDIA trt-perf-analysis 工具随仓库分发到 `tools/trt_perf/`）：

```bash
bash launch/perf_trt.sh det                        # -> outputs/perf/layers_*.json + profile_*.json
bash ../../tools/trt_perf/run.sh                   # 生成交互式 HTML 报告 -> outputs/perf/report/
```

TRT 强类型说明（TRT >= 10.12 / 11.x）：精度由 ONNX 图表达（Cast/QDQ），
`BuilderFlag.FP16/INT8` 已移除；FP32 ONNX 构建 FP32 engine。需要混合 FP16
时先用 ModelOpt AutoCast（`pip install nvidia-modelopt`），或在
`builder.py`/`pipeline.py` 加 `--autocast-fp16`。TRT < 10.12（docker 目标
10.9.0.34）自动回退弱类型，`--fp16` 照常生效（版本在构建时探测）。

## 推理结果可视化

TRT 推理结果渲染到 `outputs/vis/`（与模型/ckpt 目录分离，jpg + gif）：

```bash
# 现场跑 TRT 推理并可视化：单帧（BEV + 6 相机投影 jpg）
launch/visualize_trt.sh --sample-index 5

# 多帧时间序列：连续渲染 10 帧（同一场景连续 sweep），另存 seq gif
launch/visualize_trt.sh --sample-index 5 --frames 10 --fps 4

# 复用已 dump 的 npz（跳过模型推理，仅渲染；checkpoint 可省略，单帧）
python deploy/visualize.py configs/det_convfuser.py \
    --npz outputs/trt/sample5.npz --sample-index 5 --out outputs/vis
```

输出：

```
outputs/vis/
  sample5_det.jpg             # 单帧：BEV 鸟瞰 + 6 相机视图（3D 框投影，含类别/分数）
  sample5-14_seq.gif          # --frames 10：时间序列动画（默认 fps 4）
```

可选参数：`--score_thr 0.3`（过滤低分框）、`--max_boxes 40`（最多框数）、
`--stride 1`（帧间隔，用于降采样）。渲染复用 `scripts/render.py`
（与 torch 侧 `scripts/visualize.py` 同一套绘图）。

## 模型 I/O 约定

详见 [`docs/MODEL_IO.md`](docs/MODEL_IO.md)：坐标系约定、forward 输入键、
内部特征流、检测/分割/占据输出 shape 与类别表。

## 子模块

- `models/calib_board_detector/`：SECOND 标定板检测器（训练与推理，见
  [`models/calib_board_detector/README.md`](models/calib_board_detector/README.md)）
