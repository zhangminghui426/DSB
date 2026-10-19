# dsb_infer（C++ 推理）

C++ 端到端推理：加载 TRT engine，复用与 `deploy/inferencer.py` 相同的
图拆分（`deploy/graphs.py`），CPU 侧实现 camera 几何 / bev_pool / 降采样、
lidar 体素化 + 稀疏编码，det 解码在 CPU 完成。

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

## 构建

```bash
mkdir -p build && cd build
cmake .. -DTENSORRT_ROOT=/opt/TensorRT-10.9.0.34
make -j
```

## 运行

输入文件用 `tools/export_cpp_input.py` 从一张 val 样本导出：

```bash
python tools/export_cpp_input.py configs/det_convfuser.py ckpts/bevfusion-det-bf.pth \
    --index 5 --out outputs/cpp_input
```

```bash
./build/dsb_infer \
    --config configs/det_convfuser.py \
    --engine-dir outputs/trt \
    --input-dir outputs/cpp_input \
    --dump outputs/cpp_out \
    --verify
```

`--dump` 写出 boxes/scores/labels 的二进制（`boxes.bin`/`scores.bin`/`labels.bin`），
`--verify` 在有 `<input-dir>/ref/decoder_feat.bin` 时对比 decoder_feat（torch 参考）。

## 输入文件布局（tools/export_cpp_input.py 产出）

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
```

## 目录结构

```
src/cpp_infer/
  CMakeLists.txt
  include/dsb_infer/{config,ops,trt_engine,pipeline}.hpp
  src/{decode,preprocess,sparse,trt_engine,pipeline,main}.cpp
  README.md
```

## 已知限制

1. **检测解码在 CPU**：`decode.cpp` 实现 TransFusion 解码（NMS + one_hot +
   bbox coder），未用 TRT 的 NMS 插件；结果应与 torch 路径一致（建议对照
   `tools/trt_perf/` 的 reference dump）。
2. **只支持 batch=1 与固定输入尺寸**：engine 为固定 shape（batch 1）。
3. **9 个 head 输出**：`head.bbox` 输出 8 个 pred tensor + `query_labels`
   （float 承载 proposal 类别 id，来自图内 topk，解码时转 int）。tensor 名以
   `deploy/graphs.py` 为准。
4. **相机/LiDAR 分支的 CPU 算子未做 SIMD 优化**，仅供功能验证。
5. **CMake 需要 TensorRT 与 CUDA**：`-DTENSORRT_ROOT` 指向 TRT 安装目录。
