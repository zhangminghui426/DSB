# DSB：自动驾驶感知与标定一体化平台

BEV 感知（BEVFusion + FlashOCC）+ 雷达-相机标定（fast_calib）统一项目，纯 PyTorch、零 mmcv 依赖。

## 目录结构

```
DSB/
├── app/                     # Streamlit UI（上传 mcap、运行标定、3D 预览）
├── src/                     # 工程代码（预留 ROS2 集成）
├── 3rdparty/
│   ├── deeplearning/        # 深度学习模型库
│   │   ├── models/          # BEVFusion/FlashOCC 模型 + SECOND 标定板检测器
│   │   ├── datasets/ scripts/ configs/ ops/ tools/ tests/ launch/
│   │   └── deploy/          # TRT 推理部署（C++）
│   │       ├── scripts/     # export.py：分模块 ONNX 导出（5 张图）
│   │       ├── src/         # C++ 推理（CUDA-BEVFusion 移植）
│   │       ├── tool/        # example-data 导出、三端对拍、环境脚本
│   │       └── model/       # ONNX / TRT engine（生成产物）
│   └── fast_calib/          # 雷达-相机外参标定
├── packet/                  # Docker 安装包（CUDA/cuDNN/TensorRT）
├── tools/                   # 工程工具
├── Dockerfile / requirements.txt / docs/
└── README.md
```

## 快速开始

### 方式一：Docker（推荐）

```bash
cd packet && bash download_env.sh       # 下载 CUDA/cuDNN 安装包；TensorRT 手动下载放入 packet/
docker build -t dsb:latest .
docker run --gpus all -it --rm -v ~/wk_robot/aigorithm/DSB:/workspace dsb:latest
cd 3rdparty/deeplearning && python setup.py build_ext --inplace   # 编译 CUDA 算子
```

环境：Ubuntu 22.04 · CUDA 12.8.2 · cuDNN 9.26.1 · TensorRT 10.9.0.34 · Python 3.10 · PyTorch 2.8.0(cu128)

### 方式二：本地 Conda 环境

```bash
conda create -n DSB python=3.10 -y && conda activate DSB
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
cd 3rdparty/deeplearning && python setup.py build_ext --inplace
```

## 常用命令

```bash
cd 3rdparty/deeplearning

# 训练 / 评估（configs 支持 det|seg|multitask|occ|m0|stereo|stbase）
python scripts/train.py configs/det_convfuser.py --load_from ckpts/bevfusion-det-bf.pth
python scripts/test.py configs/det_convfuser.py ckpts/bevfusion-det-bf.pth --eval bbox

# 1) 导出分模块 ONNX（5 张图，含 SCN；--verify 用 onnxruntime 逐图校验）
python deploy/scripts/export.py configs/det_convfuser.py ckpts/bevfusion-det-bf.pth --verify

# 2) 拷贝 ONNX 到 deploy/model/resnet50/ 并构建 TRT engine
#    （head.bbox 自动用 fp32 层构建为 head.bbox.fp32.plan）
DEBUG_MODEL=resnet50 DEBUG_PRECISION=fp16 bash deploy/build_trt_engine.sh

# 3) 生成 C++ 输入（6 相机 jpg + 矩阵/点云 tensor）
python deploy/tool/export_example_data.py configs/det_convfuser.py \
    ckpts/bevfusion-det-bf.pth --sample-index 0

# 4) C++ 推理（输出解码框 + build/cuda-bevfusion.jpg）
cd deploy && bash tool/run.sh
```

```bash
# Streamlit UI
streamlit run app/app.py

# 雷达-相机标定
cd 3rdparty/fast_calib && python scripts/calib_lidar_cmaera.py --config config/config.yaml

# 训练标定板检测器
cd 3rdparty/deeplearning/models/calib_board_detector && python train.py
```

## 详细文档

- [3rdparty/deeplearning/README.md](3rdparty/deeplearning/README.md) — 模型库结构、训练/评估
- [3rdparty/deeplearning/deploy/README.md](3rdparty/deeplearning/deploy/README.md) — TRT 部署（ONNX 导出 + C++ 推理）
- [3rdparty/fast_calib/README.md](3rdparty/fast_calib/README.md) — 雷达-相机外参标定
- [tools/README.md](tools/README.md) — 工程工具说明
- [src/README.md](src/README.md) — ROS2 集成预留说明
- [3rdparty/deeplearning/docs/MODEL_IO.md](3rdparty/deeplearning/docs/MODEL_IO.md) — 模型 I/O 约定

## 已验证功能

- BEV 3D 目标检测（mini-val NDS/mAP 0.585）、语义分割（mean IoU@max 0.718）、占据网络（mIoU 24.09）
- 标定板自动检测（SECOND 3D 检测器）、雷达-相机外参标定（fast_calib）、Streamlit 可视化界面
- C++ TRT 推理端到端：检测框与 torch 参考位置差 ~1–2cm，运行稳定
