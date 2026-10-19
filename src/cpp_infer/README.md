# DSB C++ Inference (det)

纯 C++ 推理流水线（`src/cpp_infer/`），与 `scripts/export.py` 分模块导出的
4 个 TensorRT engine 衔接，在 C++/CUDA 侧完成 torch runtime 部分
（相机预处理与深度投影、几何与 bev_pool、体素化与稀疏卷积、bbox 解码），
整链路不依赖 Python。

## 流水线

```
img (1,6,3,256,704)
  ├─ [TRT camera.backbone] ──────── feats (1,6,256,32,88)
  ├─ project_depth (C++ CPU) ────── depth (1,6,1,256,704)
  └─ [TRT camera.vtransform] ────── depthnet (1,6,198,32,88)
        ├─ softmax(depth bins) × feats ── x (1,6,118,32,88,80)
        ├─ get_geometry (C++ CPU) ─────── geom (6,118,32,88,3)
        ├─ bev_pool (C++ CPU) ─────────── (80,360,360)
        └─ downsample (C++ CPU) ───────── cam_bev (1,80,180,180)
points (N,5)
  ├─ hard voxelize (C++ CPU) ────── feats/coords/sizes
  └─ sparse encoder (C++ CPU) ───── lidar_bev (1,256,180,180)
[TRT fuser] ── decoder_feat (1,512,180,180)
[TRT head.bbox] ── 8 preds ── decode (C++ CPU) ── boxes / scores / labels
```

- BEV 网格：相机侧 360×360 @0.3 m，经 downsample(×2) 对齐 lidar 侧 180×180 @0.6 m；
  lidar 分支 Z 维（6→2）在 sparse encoder `conv_out` 折叠进通道（128×2=256）。
- 解码：与 `TransFusionHead.get_bboxes`（`nms_type=None`）+
  `TransFusionBBoxCoder.decode` 一致：dense heatmap NMS(3×3) → 全类别 top-200
  proposals → one-hot 打分 → 中心/尺寸/朝向/速度反算 → `post_center_range` 过滤。

## 依赖与编译（用户机器）

- Ubuntu 22.04 / CUDA 12.8 / TensorRT 10.9.0.34（`$TENSORRT_ROOT`）
- g++ ≥ 9（C++17）、CUDA runtime（仅用于 TRT 设备缓冲）
- OpenCV 可选（当前未用于解码 raw 图，输入为预处理后的 bin）

```bash
cd src/cpp_infer
cmake -B build -DTENSORRT_ROOT=/path/to/TensorRT-10.9.0.34
cmake --build build -j
```

## 输入导出（先在有 GPU 的机器上执行一次）

```bash
python tools/export_cpp_input.py configs/det_convfuser.py ckpts/bevfusion-det-bf.pth \
    --out-dir deploy_inputs --sample-index 0
```

产出：

```
deploy_inputs/
├── meta.txt              # C++ 端几何/结构/decode 参数
├── meta.json             # 同上（人读版）
├── inputs/img.bin        # (6,3,256,704) fp32 归一化图像（eval 协议）
├── inputs/points.bin     # int32 n + (n,5) fp32 雷达点
├── inputs/matrices.bin   # 19×4×4：6×camera2lidar + 6×intrins + 6×img_aug + lidar_aug
├── weights/downsample.bin
├── weights/sparse_encoder.bin   # 卷积/BatchNorm 参数（含每层 offset 表）
└── ref/                  # torch 参考输出（decoder_feat.bin + boxes.json）
```

## 运行

```bash
# 先生成 4 个 engine（见 launch/infer_trt.sh --build），放在 engines/
dsb_infer --input-dir deploy_inputs --engine-dir engines [--dump-dir out]
```

- 对照：有 `ref/` 时自动打印 `decoder_feat` 与 torch 的最大相对误差；
  建议先验证该误差 < 1e-3，再核对 boxes。
- `--dump-dir`：保存 `cam_bev.bin / lidar_bev.bin / decoder_feat.bin /
  boxes.bin / scores.bin / labels.bin` 供 python 侧二次对照。

## 工程内核对口

| 阶段 | C++ | torch 参照 |
| --- | --- | --- |
| 深度投影 | `project_depth` | `deploy/preprocess.py` |
| 几何 | `compute_geometry` | `models/vtransforms/base.py get_geometry` |
| bev_pool | `bev_pool`（CPU 段累加） | `ops/bev_pool_v2/bev_pool.py`（CUDA kernel） |
| downsample | `downsample_conv` | `depth_lss.py downsample` |
| 体素化 | `voxelize` | `ops/voxel/src/voxelization.cpp` |
| 稀疏编码 | `sparse_encoder_forward` | `models/backbones/sparse_encoder.py` |
| 解码 | `decode_det` | `models/heads/transfusion.py + models/core/bbox_coder.py` |

## 已知限制 / 待验证项

1. **稀疏卷积为 CPU 参考实现**（正确性优先，速度未优化）。
   体素化后约 18 万 voxel，SparseEncoder 全链路 CPU 预计数十秒级；
   需要实时性能时再实现 CUDA 版（kernel 结构已按层拆分，可直接替换）。
2. **spconv 权重 K 维邻域顺序按 z→y→x 枚举**（与 `tools/export_cpp_input.py`
   写入 meta 的 offsets 一一对应）。若对照发现数值错位，只需调整
   `export_cpp_input.py` 中 `collect_sparse_layers` 的 offsets 生成顺序后重导。
3. **TRT engine tensor 名**按 `deploy/trt_common.py` 的输入输出名
   （`img/feats/depth/depthnet/cam_bev/lidar_bev/decoder_feat` + 8 个 head 输出）
   访问；若导出脚本改名，`main.cpp` 相应调整。
4. `batch=1` 静态形状，与 `deploy/trt_build.py` 默认一致。
5. 图像输入为 eval 协议预处理后的归一化张量（bin）；raw 图解码 + 预处理
   的 OpenCV 入口未实现。
