# DeepLearning（BEVFusion + FlashOCC 重写）

纯 PyTorch 统一感知模型库，零 mmcv 依赖。覆盖：

- BEV 3D 目标检测（BEVFusion det，ConvFuser 融合）
- BEV 语义分割（`seg_fusion_bev256d2_lss`）
- 占据网络预测（FlashOCC / Stereo4D）
- 标定板自动检测（SECOND，见 `models/calib_board_detector/`）

## 目录结构

```
deeplearning/
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
├── deploy/            # TRT 部署（见 deploy/README.md）：export.py 前置导出 + python/ + cpp/ + tools/ + launch/
├── launch/            # 模型库启动脚本（prepare_data/train/eval/visualize/env.sh）
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

详见 [`deploy/README.md`](deploy/README.md)：前置导出（export.py，Python/C++
共用）、Python/C++ 推理、启动脚本与 TRT 版本说明。

```bash
# 1. 前置导出：ONNX 分模块导出 + trtexec 构建 engine（Python/C++ 共用）
python deploy/export.py configs/det_convfuser.py ckpts/bevfusion-det-bf.pth \
    --task det --verify --fp16

# 2. TRT 推理（直接用 engine，不需要 .pth；对照 torch 加 --checkpoint ... --verify）
python deploy/python/pipeline.py configs/det_convfuser.py \
    --engine_dir outputs/trt --sample-index 5 --dump outputs/trt/sample5.npz
```

## 模型 I/O 约定

详见 [`docs/MODEL_IO.md`](docs/MODEL_IO.md)：坐标系约定、forward 输入键、
内部特征流、检测/分割/占据输出 shape 与类别表。

## 子模块

- `models/calib_board_detector/`：SECOND 标定板检测器（训练与推理，见
  [`models/calib_board_detector/README.md`](models/calib_board_detector/README.md)）
