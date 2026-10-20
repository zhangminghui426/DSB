# SECOND 点云 3D 目标检测复现

基于 spconv 的 SECOND (Sparsely Embedded Convolutional Detection) 简化实现，用于标定板检测的训练流程验证。

## 项目结构

```
second_det/
├── configs/default.py        # 所有超参数(体素化/模型/训练/数据路径)
├── data/
│   ├── pcd/                  # PCD 点云文件 (11帧)
│   ├── annotations.json      # 原始标注 (201帧全量)
│   └── samples.json          # 预处理输出 (11帧有效样本)
├── preprocess/
│   ├── pcd_reader.py         # binary PCD 读取
│   ├── voxelize.py           # 硬体素化
│   └── prepare_data.py       # 标注解析 + 生成 samples.json
├── models/
│   ├── vfe.py                # Voxel Feature Encoder
│   ├── backbone.py           # SpMiddleFPN (spconv 稀疏卷积)
│   ├── rpn.py                # RPN 检测头
│   └── second.py             # SECOND 整体模型 + anchor 生成
├── datasets/dataset.py       # PyTorch Dataset + anchor 匹配
├── train.py                  # 训练入口
└── output/                   # 模型权重输出
```

## 快速开始

```bash
# 1. 安装依赖
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install spconv

# 2. 数据预处理 (已执行过, 生成 data/samples.json)
python -m preprocess.prepare_data

# 3. 训练
python train.py
```

## 环境说明

- 当前环境为 CPU 无 GPU。spconv 已安装但反向传播在纯 CPU 下不可用，backbone 自动 fallback 到致密 3D 卷积。
- 在有 CUDA GPU 的环境下，backbone 会自动切换到真正的 spconv 稀疏卷积路径，内存占用大幅降低。

## 关键配置

| 参数 | 值 | 说明 |
|---|---|---|
| voxel_size | 0.1 x 0.1 x 0.2 m | 体素尺寸 |
| point_cloud_range | [0, -5, -3, 10, 5, 2] | 聚焦标定板区域 |
| grid_size | 100 x 100 x 25 | BEV 网格 |
| max_points_per_voxel | 10 | 每体素最大点数 |
| 检测类别 | calibration_board | 单类 |
| anchor 尺寸 | 0.55 x 1.7 x 1.18 | 标定板标注尺寸 |
| anchor 朝向 | 0°, 90° | 2 种旋转 |

## 训练结果 (5 epoch, CPU)

- 模型参数量: 4.29M
- 训练集 9 帧 / 验证集 2 帧
- val_loss: 0.52 -> 0.16 (5 epoch 内稳定下降)
- 输出权重: output/best_model.pth, output/last_model.pth
