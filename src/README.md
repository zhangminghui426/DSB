# src/

ROS2 集成代码（Jazzy）：

- [dsb_perception/](dsb_perception/) — 订阅 6 路相机 + LiDAR，调用
  `3rdparty/deeplearning/deploy` 的 CUDA-BEVFusion TensorRT 引擎，发布
  `dsb_perception/BoundingBoxArray` 与 RViz MarkerArray。

构建/运行见 [dsb_perception/README.md](dsb_perception/README.md)。
