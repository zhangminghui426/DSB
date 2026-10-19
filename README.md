# 自动驾驶感知与标定一体化平台

BEV 感知 + 雷达-相机外参标定的统一项目，整合了 BEVFusion、FlashOCC、SECOND 标定板检测器和 fast_calib 标定算法。

## 项目组成

| 模块 | 位置 | 说明 |
|---|---|---|
| BEV 感知 | `3rdparty/deepleraing/` | BEVFusion + FlashOCC 重写，纯 PyTorch，零 mmcv 依赖 |
| 标定板检测器 | `3rdparty/deepleraing/models/calib_board_detector/` | SECOND 3D 目标检测，用于自动检测标定板 |
| 雷达-相机标定 | `3rdparty/fast_calib/` | 激光雷达与鱼眼相机外参标定算法 |
| Streamlit UI | `app/` | 可视化界面，支持上传 mcap、运行标定、3D 预览 |
| 工程代码 | `src/` | 后续 ROS2 集成代码 |

## 目录结构

```
项目根目录/
├── app/                    # Streamlit UI
│   └── app.py              # 主界面
├── src/                    # 工程代码（ROS2 集成）
├── 3rdparty/
│   ├── deepleraing/       # 深度学习模型库
│   │   ├── models/         # 模型定义
│   │   │   ├── backbones/  # swin, resnet, second, sparse_encoder
│   │   │   ├── necks/      # glss, second, custom_fpn
│   │   │   ├── heads/      # transfusion, segm, occ_head
│   │   │   ├── vtransforms/ # lss, depth_lss, occ_lss
│   │   │   ├── fusers/     # conv fuser
│   │   │   ├── perception.py # 统一顶层模型
│   │   │   └── calib_board_detector/ # 标定板检测器（SECOND）
│   │   ├── datasets/       # 数据集和数据处理
│   │   ├── scripts/        # 训练、测试、导出、可视化脚本（仅 .py）
│   │   ├── deploy/         # 导出/TRT 推理运行时组件（bev_pool/lidar/preprocess/infer_trt）
│   │   ├── launch/         # shell 启动脚本（train/eval/export/infer_trt/visualize）
│   │   ├── configs/        # 配置文件
│   │   ├── ops/            # 自定义 CUDA 算子
│   │   ├── tools/          # 数据准备工具
│   │   └── tests/          # 单元测试
│   └── fast_calib/         # 雷达-相机标定算法
│       ├── scripts/        # 标定主程序、检测器接口
│       ├── config/         # 标定配置
│       └── tools/          # 可视化工具
├── Dockerfile              # Docker 构建文件
├── requirements.txt        # Python 依赖
├── packet/                 #  Docker 安装包（CUDA/cuDNN/TensorRT）
│   └── download_env.sh     # 下载安装包脚本
├── docs/                   # 文档
└── README.md
```

## 快速开始

### 方式一：Docker（推荐）

```bash
# 1. 下载安装包（CUDA + cuDNN 自动下载，TensorRT 需手动下载）
cd packet/
bash download_env.sh
# TensorRT 需手动从 https://developer.nvidia.com/tensorrt/download/10x 下载
# 保存为 packet/TensorRT-10.9.0.34.Linux.x86_64-gnu.cuda-12.8.tar.tar
cd ..

# 2. 构建镜像
docker build -t dsb:latest .

# 3. 启动容器（挂载代码目录）
docker run --gpus all -it --rm \
    -v ~/wk_robot/aigorithm/DSB:/workspace \
    dsb:latest

# 容器内已自动激活 DSB conda 环境，直接编译算子
cd 3rdparty/deepleraing
python setup.py build_ext --inplace
```

Docker 环境配置：
- Ubuntu 22.04 + CUDA 12.8.2 + cuDNN 9.26.1 + TensorRT 10.9.0.34
- Miniconda3，DSB 环境 Python 3.10
- PyTorch 2.8.0 + cu12.8

### 方式二：本地 Conda 环境

```bash
# 创建 conda 环境
conda create -n DSB python=3.10 -y
conda activate DSB

# 安装 PyTorch（cu128）
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128

# 安装其余依赖
pip install -r requirements.txt

# 编译 CUDA 算子
cd 3rdparty/deepleraing
python setup.py build_ext --inplace
```

### 训练 BEV 感知模型（BEVFusion 检测）

```bash
cd 3rdparty/deepleraing

# 1. 准备数据：软链接 nuScenes 数据目录（配置中 dataset_root="data/nuscenes/"，
#    需要 nuscenes_infos_train.pkl / nuscenes_infos_val.pkl / nuscenes_dbinfos_train.pkl）
ln -s /path/to/nuscenes data/nuscenes

# 2. 编译 CUDA 算子（bev_pool_v2 / voxel / iou3d）
python setup.py build_ext --inplace

# 3. 训练（单卡；多卡用 torchrun --nproc_per_node=N，官方 recipe batch 4/卡）
python scripts/train.py configs/det_convfuser.py --load_from ckpts/bevfusion-det-bf.pth

# 4. 评估
python scripts/test.py configs/det_convfuser.py ckpts/bevfusion-det-bf.pth --eval bbox
```

常用训练参数：
- `--batch_size N`：覆盖 `data.samples_per_gpu`（单卡 8GB 默认 2）
- `--epochs N`：覆盖 `max_epochs`
- `--no_amp`：关闭混合精度
- `--no_eval`：跳过训练中评估
- `--options key=value`：点号式配置覆盖，如 `--options optimizer.lr=2e-4`

### 导出 ONNX 与 TensorRT 推理

模型按 CUDA-BEVFusion 布局**分模块导出**（det/seg 各 4 张图），自定义 CUDA
算子（bev_pool / 体素化 / sparse conv）留在 torch runtime，不进入 ONNX：

| 图 | 输入 → 输出 | 说明 |
|---|---|---|
| `camera.backbone.onnx` | img `(1,6,3,256,704)` → feats `(1,6,256,32,88)` | Swin-T + LSS-FPN |
| `camera.vtransform.onnx` | feats (+depth) → depthnet `(1,6,D+C,32,88)` | det 含 dtransform |
| `fuser.onnx` | cam_bev + lidar_bev → decoder_feat `(1,512,180,180)` | fuser+SECOND+FPN |
| `head.bbox/map.onnx` | decoder_feat → 8 个 pred / masks | 不解码，decode 留在 torch |

```bash
cd 3rdparty/deepleraing

# 1. 导出 ONNX（--verify 用 onnxruntime 逐图校验，需 pip install onnx onnxruntime）
python scripts/export.py configs/det_convfuser.py ckpts/bevfusion-det-bf.pth --task det --verify

# 2. 构建 TRT engine（缓存到 outputs/trt/，--fp16 可选）并跑一个 val 样本
python deploy/infer_trt.py configs/det_convfuser.py ckpts/bevfusion-det-bf.pth \
    --engine_dir outputs/trt --build --fp16 --verify

# 3. 保存解码结果
python deploy/infer_trt.py configs/det_convfuser.py ckpts/bevfusion-det-bf.pth \
    --engine_dir outputs/trt --sample-index 5 --dump outputs/trt/sample5.npz
```

### launch 启动脚本

所有 shell 启动脚本统一放在 `launch/`（`scripts/` 只放 .py）：

```bash
launch/prepare_data.sh            # 数据软链接 + infos pkl 生成
launch/train.sh det               # 训练（det|seg|multitask|occ|m0|stereo|stbase）
launch/eval.sh det                # 评估
launch/visualize.sh det           # 可视化
launch/export.sh det              # 导出 ONNX
launch/infer_trt.sh det           # TRT 推理（加 --build --fp16 --verify 透传）
```

`launch/env.sh` 可覆盖：`PY`（conda python）、`CUDA_HOME`、`NGPU`（多卡
torchrun）等环境变量。

### 启动 Streamlit UI

```bash
streamlit run app/app.py
```

### 训练标定板检测器

```bash
cd 3rdparty/deepleraing/models/calib_board_detector
python train.py
```

### 运行标定

```bash
cd 3rdparty/fast_calib
python scripts/calib_lidar_cmaera.py --config config/config.yaml
```

## 已验证功能

- BEV 3D 目标检测：mini-val NDS/mAP 0.585
- BEV 语义分割：mean IoU@max 0.718
- 占据网络预测：mIoU 24.09
- 标定板自动检测：SECOND 3D 检测器
- 雷达-相机外参标定：fast_calib 算法
- Streamlit 可视化界面
