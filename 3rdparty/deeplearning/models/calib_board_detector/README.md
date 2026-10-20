# SECOND 标定板检测器

基于 spconv 的 SECOND 简化实现，用于标定板自动检测（训练 + 推理）。
被 `3rdparty/fast_calib/scripts/detector.py` 调用，替代手动 ROI。

## 目录结构

```
calib_board_detector/
├── configs/default.py       # 超参数（体素化/模型/训练）
├── datasets/dataset.py      # PyTorch Dataset + anchor 匹配
├── models/                  # vfe / backbone / rpn / second
├── preprocess/voxelize.py   # 硬体素化
├── utils/pcd_reader.py      # binary PCD 读取
├── train.py                 # 训练入口
└── infer_visualize.py       # 推理 + 可视化
```

## 训练

```bash
pip install torch spconv
cd 3rdparty/deeplearning/models/calib_board_detector
python train.py
```

数据（PCD + annotations.json）由外部提供，标注解析后生成 `data/samples.json`。

## 环境说明

- 纯 CPU 可训练：spconv 反向传播不可用时 backbone 自动回退到致密 3D 卷积；
  有 CUDA GPU 时自动切换真正的 spconv 稀疏卷积路径，内存占用大幅降低。

## 关键配置（configs/default.py）

| 参数 | 值 | 说明 |
|---|---|---|
| voxel_size | 0.1×0.1×0.2 m | 体素尺寸 |
| point_cloud_range | [0,-5,-3,10,5,2] | 聚焦标定板区域 |
| grid_size | 100×100×25 | BEV 网格 |
| max_points_per_voxel | 10 | 每体素最大点数 |
| 检测类别 | calibration_board | 单类 |
| anchor 尺寸 | 0.55×1.7×1.18 | 标定板标注尺寸 |
| anchor 朝向 | 0°, 90° | 2 种旋转 |

## 训练结果（5 epoch, CPU）

- 模型参数量 4.29M；训练集 9 帧 / 验证集 2 帧
- val_loss 0.52 → 0.16（5 epoch 内稳定下降）
- 输出权重：`output/best_model.pth`、`output/last_model.pth`
