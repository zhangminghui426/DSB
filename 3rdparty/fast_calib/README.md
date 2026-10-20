# Fast-Calib

雷达-鱼眼/针孔相机外参快速标定：Open3D RANSAC 平面拟合 + ArUco PnP + 3D-3D SVD 优化。

## 目录结构

```
fast_calib/
├── config/
│   ├── config.yaml            # 标定配置（相机内参、雷达/相机话题、ROI）
│   └── second_config.yaml     # SECOND 检测器配置
├── scripts/
│   ├── calib_lidar_cmaera.py  # 标定主程序（命令行入口）
│   ├── detector.py            # 标定板检测器接口（SECOND / 手动 ROI）
│   ├── cameras.py             # 相机模型（鱼眼/针孔）
│   └── utils.py               # 工具函数（mcap 解析、点云转换等）
└── tools/
    ├── vis_lidar.py           # 点云可视化（mcap / pcd）
    ├── batch_detect.py        # 批量检测
    └── open3d_pcd.py          # Open3D 查看 pcd 文件
```

SECOND 检测器本体在 `3rdparty/deeplearning/models/calib_board_detector/`
（`scripts/detector.py` 自动挂载该目录）。权重路径由 `second_detector.model_path`
配置（默认 `/tmp/best_model.pth`），缺权重时自动回退手动 ROI 模式。

## 环境依赖

```bash
conda create -n flashocc python=3.10
conda activate flashocc

pip install streamlit numpy opencv-python av open3d scipy \
    mcap mcap-ros2-support pyyaml torch
```

## 快速开始

1. 修改 `config/config.yaml`：
   - `camera_intrinsic.K` / `dist`：相机内参矩阵和畸变系数
   - `data_config.lidar_topic` / `cam_topics`：雷达与相机话题名（相机可配多个）
   - `second_detector.enabled` / `model_path`：是否启用 SECOND 自动检测及权重路径

2. 命令行标定：

```bash
python scripts/calib_lidar_cmaera.py --config config/config.yaml
```

查看标定结果投影：

```bash
python scripts/calib_lidar_cmaera.py --config config/config.yaml \
    --vis outputs/result/cam_gmsl_2_lidar2cam_fastcalib.npy --frame 0
```

Streamlit UI 入口在仓库根：`streamlit run app/app.py`。

## 标定板参数

- 板尺寸：1400mm × 1000mm
- 4 个圆孔中心（板坐标系，单位米）：
  `[-0.25, 0.20, 0]`、`[0.25, 0.20, 0]`、`[-0.25, -0.20, 0]`、`[0.25, -0.20, 0]`
- ArUco marker 尺寸：0.20m；孔半径：0.12m

## 功能说明

- 时间同步：以雷达帧为基准匹配最近相机帧；自动对齐后随机抽 N 帧（默认 50）检测，加速大数据集处理
- SECOND 检测：自动输出每帧 ROI 区域，替代手动 ROI；逐帧 3D 可视化（ROI 内红 / 外灰）+ 图像投影
- 结果输出：`outputs/result/` 保存 `.npy`（4×4 外参矩阵）和 `.yaml`（含重投影误差）

## 输出文件

```
outputs/
├── calib.log
├── result/                    # cam_gmsl_2_lidar2cam_fastcalib.npy / .yaml
└── intermediate/cam_gmsl_2/
    ├── images/                # 每帧 ArUco 检测图
    ├── pointcloud/            # 原始点云 pcd
    ├── roi_pointcloud/        # ROI 裁剪点云
    ├── reproj/                # 3D 孔重投影到图像
    └── proj/                  # 全点云投影到图像
```
