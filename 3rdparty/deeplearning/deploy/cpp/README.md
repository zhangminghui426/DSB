# dsb_infer（C++ 推理，3rdparty/deeplearning/deploy/cpp）

部署总览（前置导出 / Python 推理 / 启动脚本）见
[`../README.md`](../README.md)。

C++ 端到端推理：加载 TRT engine，复用与 `deploy/python/scripts/pipeline.py` 相同的
图拆分（`deploy/python/graphs.py`），CPU 侧实现 camera 几何 / bev_pool /
降采样、lidar 体素化 + 稀疏编码，det 解码在 CPU 完成。

```
img ──► [TRT camera.backbone] ── feats ──► project_depth ──► [TRT camera.vtransform] ── depthnet
        │                                                                                │
        │                                 softmax(depth) × feats                         │
        │                                        │                                      │
        │                        compute_geometry + bev_pool (CPU)                      │
        │                                        │                                      │
        │                                  downsample_conv (CPU)                         │
        │                                        │                                      │
        │                                    cam_bev (1,80,180,180)                      │
        │                                        │                                      │
points ──► voxelize + sparse_encoder (CPU) ──── lidar_bev (1,256,180,180)               │
        │                                        │                                      │
        │                              [TRT fuser] ── decoder_feat (1,512,180,180)        │
        │                                        │                                      │
        │                              [TRT head.bbox] ── 9 preds ── decode (C++ CPU) ──► boxes / scores / labels
```

## 三层架构（与 deploy/python 对应）

Python 与 C++ 按同一三层结构组织（预处理 / 推理 / 后处理），C++ 推理用到的
自定义算子集中在 `src/ops/`（头在 `include/dsb_infer/ops/`），其参考定义
（Python/CUDA 实现）见 `3rdparty/deeplearning/ops/`（bev_pool_v2 / voxel / iou3d）。

| 层 | C++ 目录 / 文件 | Python 模块 | 职责 |
| --- | --- | --- | --- |
| 预处理 | `src/preprocess/preprocess.cpp`（头 `include/dsb_infer/preprocess/preprocess.hpp`） | `deploy/python/preprocess.py` | 加载导出输入、lidar 深度投影、点云体素化 |
| 推理 | `src/inference/`（inference.cpp + trt_engine.cpp） | `deploy/python/inference.py` + bev_pool_torch | TRT engine 编排（几何/bev_pool/降采样/稀疏编码走 `ops/`） |
| 算子 | `src/ops/ops.cpp` | `3rdparty/deeplearning/ops/`（参考） | CPU 自定义算子：bev_pool / 体素化 / 稀疏卷积 |
| 后处理 | `src/postprocess/postprocess.cpp` | `deploy/python/postprocess.py` | 检测解码（NMS + bbox coder + 过滤） |
| 编排 | `src/pipeline.cpp` | `deploy/python/scripts/pipeline.py` | 三层组合 + verify / dump |

## 构建

```bash
cd 3rdparty/deeplearning/deploy/cpp
mkdir -p build && cd build
cmake .. -DTENSORRT_ROOT=/opt/TensorRT-10.9.0.34
make -j
```

## 运行

输入文件用 `deploy/tools/export_cpp_input.py` 从一张 val 样本导出（从仓库根运行）：

```bash
python deploy/tools/export_cpp_input.py configs/det_convfuser.py ckpts/bevfusion-det-bf.pth \
    --sample-index 5 --out-dir outputs/cpp_input
```

```bash
./3rdparty/deeplearning/deploy/cpp/build/dsb_infer \
    --engine-dir outputs/trt \
    --input-dir outputs/cpp_input \
    --dump-dir outputs/cpp_out
```

`--dump-dir` 写出 cam/lidar BEV、decoder_feat 与 boxes/scores/labels 的二进制
（`cam_bev.bin`/`lidar_bev.bin`/`decoder_feat.bin`/`boxes.bin`/`scores.bin`/`labels.bin`）；
存在 `<input-dir>/ref/decoder_feat.bin` 时自动对比 decoder_feat（torch 参考）。

## 输入文件布局（deploy/tools/export_cpp_input.py 产出）

```
<out>/
  inputs/
    img.bin          # (1,6,3,256,704) float32，模型输入（已归一化）
    points.bin       # int32 n + (n,5) float32
    matrices.bin     # 19 个 4×4 float32：6 camera2lidar + 6 intrins + 6 img_aug + 1 lidar_aug
  weights/
    downsample.bin   # 降采样 conv 权重（det 配置）
    sparse_encoder.bin  # 稀疏编码器权重
  ref/
    decoder_feat.bin # torch 参考（可选）
  meta.txt           # 配置（key = value），含 sparse / downsample 层表
```

## 目录结构

```
deploy/cpp/
  CMakeLists.txt               # dsb_infer 工程
  README.md
  include/dsb_infer/           # 公共头文件
    config.hpp                 #   推理配置
    tensor.hpp                 #   张量封装
    pipeline.hpp               #   三层编排
    preprocess/preprocess.hpp  #   预处理层
    inference/inference.hpp    #   推理层
    inference/trt_engine.hpp   #   TRT engine 加载/执行
    ops/ops.hpp                #   自定义 CPU 算子（对应 3rdparty/deeplearning/ops/ 参考）
    postprocess/postprocess.hpp#   后处理层
  src/                         # 实现
    main.cpp                   #   CLI 入口
    pipeline.cpp  config.cpp
    preprocess/  inference/  ops/  postprocess/
```

## 已知限制

1. **检测解码在 CPU**：`postprocess.cpp` 实现 TransFusion 解码（NMS + one_hot +
   bbox coder），未用 TRT 的 NMS 插件；结果应与 torch 路径一致（对照
   `deploy/tools/export_cpp_input.py` 产出的 `ref/`）。
2. **只支持 batch=1 与固定输入尺寸**：engine 为固定 shape（batch 1）。
3. **9 个 head 输出**：`head.bbox` 输出 8 个 pred tensor + `query_labels`
   （float 承载 proposal 类别 id，来自图内 topk，解码时转 int）。tensor 名以
   `deploy/python/graphs.py` 为准。
4. **相机/LiDAR 分支的 CPU 算子未做 SIMD 优化**，仅供功能验证。
5. **CMake 需要 TensorRT 与 CUDA**：`-DTENSORRT_ROOT` 指向 TRT 安装目录。
