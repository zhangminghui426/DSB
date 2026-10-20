# DSB：自动驾驶感知与标定一体化平台

BEV 感知（BEVFusion + FlashOCC）+ 雷达-相机标定（fast_calib）统一项目，纯 PyTorch、零 mmcv 依赖。

## 目录结构

- `app/` — Streamlit UI（上传 mcap、运行标定、3D 预览）· [app.py](app/app.py)
- `src/` — 工程代码（预留 ROS2 集成）· [README](src/README.md)
- `3rdparty/`
  - `deeplearning/` — 深度学习模型库 · [README](3rdparty/deeplearning/README.md)
    - `models/` — BEVFusion/FlashOCC 模型 + SECOND 标定板检测器
    - `datasets/ scripts/ configs/ ops/ tools/ tests/ launch/`
    - `deploy/` — TRT 推理部署 · [README](3rdparty/deeplearning/deploy/README.md)
      - `python/` — Python 推理（scripts/ 入口 + 库三层）· [README](3rdparty/deeplearning/deploy/python/README.md)
      - `cpp/` — C++ 推理（include/ + src/，同三层 + ops）· [README](3rdparty/deeplearning/deploy/cpp/README.md)
      - `tools/` — C++ 输入导出
      - `launch/` — export/infer_trt/visualize_trt 启动脚本
  - `fast_calib/` — 雷达-相机外参标定 · [README](3rdparty/fast_calib/README.md)
- `packet/` — Docker 安装包（CUDA/cuDNN/TensorRT）
- `tools/` — 工程工具 · [README](tools/README.md)
- `Dockerfile / requirements.txt / docs/`
- `README.md`

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

# 导出 ONNX + 构建 TRT engine（Python/C++ 推理共用的前置步骤；--verify 用 onnxruntime 逐图校验）
python deploy/export.py configs/det_convfuser.py ckpts/bevfusion-det-bf.pth --task det --verify --fp16

# TRT 推理（直接用 engine，不需要 .pth）并保存解码结果
python deploy/python/scripts/pipeline.py configs/det_convfuser.py --task det \
    --engine_dir outputs/trt --sample-index 5 --dump outputs/trt/sample5.npz

# 与 torch 参考对照（可选，需 checkpoint）
python deploy/python/scripts/pipeline.py configs/det_convfuser.py --checkpoint ckpts/bevfusion-det-bf.pth \
    --task det --engine_dir outputs/trt --verify

# 部署启动脚本（export / infer / visualize）
bash deploy/launch/export.sh det
bash deploy/launch/infer_trt.sh det
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

- `3rdparty/deeplearning/README.md` — 模型库结构、训练/评估
- `3rdparty/deeplearning/deploy/README.md` — TRT 部署（前置导出 / Python / C++ 推理）
- `3rdparty/deeplearning/deploy/python/README.md` — Python TRT 推理（scripts/ 入口 + 库三层）
- `3rdparty/deeplearning/deploy/cpp/README.md` — C++ 推理流水线、编译与运行
- `3rdparty/fast_calib/README.md` — 雷达-相机外参标定
- `tools/README.md` — 工程工具说明
- `src/README.md` — ROS2 集成预留说明
- `docs/MODEL_IO.md` — 模型 I/O 约定

## 已验证功能

- BEV 3D 目标检测（mini-val NDS/mAP 0.585）、语义分割（mean IoU@max 0.718）、占据网络（mIoU 24.09）
- 标定板自动检测（SECOND 3D 检测器）、雷达-相机外参标定（fast_calib）、Streamlit 可视化界面
