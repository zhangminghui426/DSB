# dsb_perception (ROS2 Jazzy)

订阅 6 路相机 + 1 路 LiDAR，调用 `3rdparty/deeplearning/deploy` 里的 CUDA-BEVFusion
TensorRT 引擎，发布 3D 检测框和 RViz Marker。

## 依赖

- ROS2 Jazzy（`/opt/ros/jazzy`）
- 已构建好的 deploy 核心库：`3rdparty/deeplearning/deploy/build/libbevfusion_core.so`
  以及 `deploy/model/<model_name>/build/*.plan`、`lidar.backbone.xyz.onnx`

先构建 deploy（如果还没构建）：

```bash
cd 3rdparty/deeplearning/deploy
bash run_trt.sh          # 编译 C++ + 跑一遍；model/resnet50 下应有 4 个 plan
```

## 构建

```bash
source /opt/ros/jazzy/setup.bash
cd DSB
colcon build --base-paths src --packages-select dsb_perception \
  --cmake-args -DCMAKE_BUILD_TYPE=Release \
               -DCUDAToolkit_ROOT=/usr/local/cuda-12.8 \
               -DPython3_EXECUTABLE=/usr/bin/python3
source install/setup.bash
```

> 用系统 Python 构建；如果在 conda 环境里，rosidl 可能加载到错误的 Python 3.10。

## 运行

```bash
source /opt/ros/jazzy/setup.bash
source DSB/install/setup.bash
ros2 launch dsb_perception perception.launch.py
```

或直接带参数：

```bash
ros2 run dsb_perception bevfusion_node --ros-args \
  --params-file install/dsb_perception/share/dsb_perception/config/perception.yaml
```

## 话题

| 方向 | 话题 | 类型 | 说明 |
|---|---|---|---|
| 订阅 | `/camera/front/image_raw` 等 6 路 | `sensor_msgs/Image` | 1600×900，`rgb8`/`bgr8` |
| 订阅 | `/lidar/points` | `sensor_msgs/PointCloud2` | 字段 `x,y,z,intensity[,time]`，LiDAR 系 |
| 订阅 | `~/matrices`（可选） | `std_msgs/Float32MultiArray` | 4×6×4×4 标定矩阵，运行时可更新 |
| 发布 | `~/boxes` | `dsb_perception/BoundingBoxArray` | 3D 框（z 为重力中心） |
| 发布 | `~/markers` | `visualization_msgs/MarkerArray` | RViz CUBE 标记 |

相机顺序固定为：`FRONT, FRONT_RIGHT, FRONT_LEFT, BACK, BACK_LEFT, BACK_RIGHT`，
必须和 `config/perception.yaml` 里的 `camera2lidar` 等矩阵顺序一致。

## 标定矩阵

`config/perception.yaml` 里的 `camera2lidar / camera_intrinsics / lidar2image /
img_aug_matrix` 各 96 个 float（6×4×4，行优先）。默认值来自
`outputs/example-data` 第 0 帧，仅用于跑通；实车请替换为实际标定结果。

也可以用 `~/matrices` 话题更新：按 `camera2lidar, camera_intrinsics, lidar2image,
img_aug_matrix` 顺序拼接，共 4×96 = 384 个 float32。

## 用示例数据测试

另开终端：

```bash
source /opt/ros/jazzy/setup.bash
source DSB/install/setup.bash
python3 src/dsb_perception/scripts/publish_example.py \
  --data-dir 3rdparty/deeplearning/outputs/example-data
ros2 topic echo /bevfusion_node/boxes --once
```

应看到约 39–41 个框（示例帧）。

## 数据同步（30 Hz 相机 / 10 Hz LiDAR / 50 ms 推理）

节点不是“取最新帧定时推理”，而是 **以 LiDAR 为基准做时间戳同步**：

1. 每路相机维护一个按时间戳排序的缓存（默认 3 帧，30 Hz 约 100 ms）。
2. 每来一帧 LiDAR（10 Hz），对每路相机选时间戳最接近的图像；偏差超过
   `sync_slop`（默认 50 ms）就丢弃该 LiDAR 帧。
3. 同步好的帧放入 **单槽队列**，由独立工作线程执行 TRT 推理，不阻塞 ROS 回调。
4. 推理 50 ms < LiDAR 周期 100 ms，因此正常情况下每个 LiDAR 帧都能处理，
   输出约 10 Hz；相机 30 Hz 中每 3 帧只用 1 帧（最近邻）。
5. 如果某次推理超过 LiDAR 周期，新帧会覆盖未处理的旧帧（drop），
   不会积压延迟。

因此输出频率由 **LiDAR 频率和推理耗时** 决定，而不是相机频率；相机只提供
最近时间戳的视觉观测。

> 测试用 `publish_example.py` 是 Python/DDS 回放，6 路 1600×900 原图
> 30 Hz 的数据量约 0.8 GB/s，回放本身可能达不到 30 Hz；实车请以真实驱动
> 和硬件时间戳为准。

## 参数

见 `config/perception.yaml`：

- `deploy_dir`：deploy 路径；空字符串时使用编译期路径
- `model_name`：产物目录名（当前 `resnet50`，实际相机骨干是 Swin-T）
- `camera_topics`：6 路相机话题
- `lidar_topic`、`frame_id`
- `sync_slop`：相机/雷达时间戳最大允许偏差（秒）
- `max_camera_frames`：每路相机缓存帧数
- `max_points`：LiDAR 最大点数
- `publish_markers`

## 当前限制

- 模型是 **6 相机**配置；单相机需要重新导出 N=1 的 `camera.backbone` 并修改
  C++/本节点的 `num_camera`（见项目讨论）。
- 相机图像必须是 1600×900；其他分辨率会被跳过并输出 warning。
- 默认 `deploy/model/resnet50` 的产物名来自 CUDA-BEVFusion，不代表相机骨干是
  ResNet-50（当前 det/seg 权重是 Swin-T）。
