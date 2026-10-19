# Fast-Calib

雷达-鱼眼/针孔相机外参快速标定工具，基于 Open3D RANSAC 平面拟合 + ArUco PnP + 3D-3D SVD 优化。

## 项目结构

```
fast_calib/
├── app.py                    # Streamlit UI 入口
├── start.sh                  # 启动脚本
├── config/
│   └── config.yaml           # 标定配置（相机内参、雷达话题、ROI 等）
├── models/
│   └── second_model/         # SECOND 3D 检测器网络结构
├── scripts/
│   ├── calib_lidar_cmaera.py # 标定主程序
│   ├── second_board_detector.py # SECOND 标定板检测器
│   ├── utils.py              # 工具函数（mcap 解析、点云转换等）
│   ├── cameras.py            # 相机模型（鱼眼/针孔）
│   └── detector.py            # 检测器接口
├── tools/
│   ├── vis_lidar.py          # 点云可视化（mcap / pcd）
│   └── open3d_pcd.py         # Open3D 查看 pcd 文件
└── data/                     # 标定板图片、内参示例
```

## 环境依赖

```bash
conda create -n flashocc python=3.10
conda activate flashocc

pip install streamlit numpy opencv-python av open3d scipy \
    mcap mcap-ros2-support pyyaml torch
```

## 快速开始

### 1. 准备模型权重

将 SECOND 检测器权重放到 `/tmp/best_model.pth`：

```bash
cp /path/to/best_model.pth /tmp/best_model.pth
```

如果不放置权重，程序会自动回退到手动 ROI 模式。

### 2. 修改配置

编辑 `config/config.yaml`：

- `camera_intrinsic.K` / `dist`：相机内参矩阵和畸变系数
- `data_config.lidar_topic`：雷达话题名
- `data_config.cam_topics`：相机话题名（可配多个）
- `second_detector.enabled`：是否启用 SECOND 自动检测
- `second_detector.model_path`：权重路径（默认 `/tmp/best_model.pth`）

### 3. 启动 UI

```bash
bash start.sh
# 或指定端口
bash start.sh 8500
```

浏览器打开 http://localhost:8500

### 4. 命令行直接标定

```bash
python scripts/calib_lidar_cmaera.py --config config/config.yaml
```

查看标定结果投影：

```bash
python scripts/calib_lidar_cmaera.py --config config/config.yaml \
    --vis outputs/result/cam_gmsl_2_lidar2cam_fastcalib.npy \
    --frame 0
```

## 标定板参数

- 物理尺寸：1400mm × 1000mm
- 4 个圆孔中心（板子坐标系，单位米）：
  - `[-0.25, 0.20, 0]`
  - `[0.25, 0.20, 0]`
  - `[-0.25, -0.20, 0]`
  - `[0.25, -0.20, 0]`
- ArUco marker 尺寸：0.20m
- 孔半径：0.12m

## 功能说明

- **时间同步**：以雷达帧为基准，匹配最近的相机帧
- **抽帧机制**：自动时间对齐后随机抽取 N 帧（默认 50）进行标定板检测，加速大数据集处理
- **SECOND 检测**：自动输出每帧 ROI 区域，替代手动 ROI
- **3D 可视化**：逐帧显示点云（ROI 内红色 / ROI 外灰色）和图像投影
- **结果输出**：`outputs/result/` 下保存 `.npy`（4x4 外参矩阵）和 `.yaml`（含重投影误差）

## 输出文件

```
outputs/
├── calib.log                     # 标定日志
├── result/
│   ├── cam_gmsl_2_lidar2cam_fastcalib.npy
│   └── cam_gmsl_2_lidar2cam_fastcalib.yaml
└── intermediate/
    └── cam_gmsl_2/
        ├── images/               # 每帧 ArUco 检测图
        ├── pointcloud/            # 原始点云 pcd
        ├── roi_pointcloud/        # ROI 裁剪点云
        ├── reproj/               # 3D 孔重投影到图像
        └── proj/                  # 全点云投影到图像
```
