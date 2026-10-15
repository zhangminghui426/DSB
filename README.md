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
│   │   ├── scripts/        # 训练、推理、可视化脚本
│   │   ├── configs/        # 配置文件
│   │   ├── ops/            # 自定义 CUDA 算子
│   │   ├── tools/          # 数据准备工具
│   │   └── tests/          # 单元测试
│   └── fast_calib/         # 雷达-相机标定算法
│       ├── scripts/        # 标定主程序、检测器接口
│       ├── config/         # 标定配置
│       └── tools/          # 可视化工具
├── docs/                   # 文档
└── README.md
```

## 快速开始

### 1. 安装依赖

```bash
conda activate bevfusion   # python 3.10, torch 2.8.0+cu128
cd 3rdparty/deepleraing
python setup.py build_ext --inplace   # 编译 CUDA 算子
```

### 2. 启动 Streamlit UI

```bash
streamlit run app/app.py
```

### 3. 训练标定板检测器

```bash
cd 3rdparty/deepleraing/models/calib_board_detector
python train.py
```

### 4. 运行标定

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
