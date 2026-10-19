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
│   │   ├── deploy/         # 导出/构建/推理包（exporter/builder/runtime/inferencer/config/graphs + bev_pool/lidar/preprocess）
│   │   ├── launch/         # shell 启动脚本（train/eval/export/infer_trt/perf_trt/visualize）
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
python deploy/exporter.py configs/det_convfuser.py ckpts/bevfusion-det-bf.pth --task det --verify

# 2. 构建 TRT engine（缓存到 outputs/trt/；ONNX->engine 只走 trtexec CLI：
#    $TRTEXEC 或 PATH；pip-only tensorrt 无 trtexec，需
#    TRTEXEC=/path/to/TensorRT-10.9.../bin/trtexec。TRT >= 10.12 强类型下
#    FP32 ONNX 出 FP32 engine，--fp16 需配合 --autocast-fp16，见 deploy/builder.py）
python deploy/inferencer.py configs/det_convfuser.py ckpts/bevfusion-det-bf.pth \
    --engine_dir outputs/trt --build --fp16 --verify

# 3. 保存解码结果
python deploy/inferencer.py configs/det_convfuser.py ckpts/bevfusion-det-bf.pth \
    --engine_dir outputs/trt --sample-index 5 --dump outputs/trt/sample5.npz
```

### C++ 推理（src/cpp_infer）

纯 C++ 推理流水线（无 Python）：4 个 TRT engine + C++ 侧 torch runtime
（相机深度投影、几何与 bev_pool、体素化 + 稀疏卷积、bbox 解码），
支持 `batch=1` 静态形状，解码结果与 torch 参考一致。

```bash
# 1. 在有 GPU 的机器导出 C++ 输入/权重/参考（一次即可）
python tools/export_cpp_input.py configs/det_convfuser.py ckpts/bevfusion-det-bf.pth \
    --out-dir deploy_inputs --sample-index 0

# 2. 编译（用户机器：Ubuntu 22.04 / CUDA 12.8 / TRT 10.9.0.34）
cd src/cpp_infer
cmake -B build -DTENSORRT_ROOT=/path/to/TensorRT-10.9.0.34
cmake --build build -j

# 3. 运行（engines 目录放 camera.backbone / camera.vtransform / fuser / head.bbox）
./build/dsb_infer --input-dir deploy_inputs --engine-dir ../../3rdparty/deepleraing/outputs/trt
```

详见 `src/cpp_infer/README.md`（流水线图、限制与待验证项）。

### launch 启动脚本

所有 shell 启动脚本统一放在 `launch/`（`scripts/` 只放 .py）：

```bash
launch/prepare_data.sh            # 数据软链接 + infos pkl 生成
launch/train.sh det               # 训练（det|seg|multitask|occ|m0|stereo|stbase）
launch/eval.sh det                # 评估
launch/visualize.sh det           # 可视化
launch/export.sh det              # 导出 ONNX
launch/infer_trt.sh det           # TRT 推理（加 --build --fp16 --verify 透传）
launch/perf_trt.sh det            # trtexec 采集逐层 profile -> outputs/perf/
```

性能分析（NVIDIA trt-perf-analysis 工具已随仓库分发到 `tools/trt_perf/`）：

```bash
# 1. 采集（需 GPU + trtexec；pip-only tensorrt 无 trtexec 时用
#    TRTEXEC 指向 tar 包 bin/trtexec：TRTEXEC=.../bin/trtexec bash launch/perf_trt.sh det）
cd 3rdparty/deepleraing
bash launch/perf_trt.sh det                        # -> outputs/perf/layers_*.json + profile_*.json
# 2. 生成交互式 HTML 报告（首次运行自动从 NVIDIA/TensorRT 官方仓库拉取
#    report_template 的 app.js / netron-assets.js 大文件）-> outputs/perf/report/
bash ../../tools/trt_perf/run.sh
# 3. 预览
python -m http.server 8765 --bind 127.0.0.1 --directory outputs/perf/report
```

TRT 强类型说明（TRT >= 10.12 / 11.x）：精度由 ONNX 图表达（Cast/QDQ），
`BuilderFlag.FP16/INT8` 已移除；FP32 ONNX 构建 FP32 engine。需要混合 FP16
时先用 ModelOpt AutoCast（`pip install nvidia-modelopt`），或在
`builder.py`/`inferencer.py` 加 `--autocast-fp16`。TRT < 10.12（docker 目标
10.9.0.34）自动回退显式 batch 弱类型，`--fp16` 照常生效（版本在构建时探测）。
ONNX→engine 只走 trtexec CLI（`$TRTEXEC` 或 PATH；缺失时构建直接报错，
需设置 TRTEXEC 指向 tar 包 bin/trtexec，见 `deploy/builder.py`）。

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
