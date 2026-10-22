# bf（deploy/python/bf/）—— CUDA-BEVFusion 结构 Python 部署

镜像 deploy/cpp/（CUDA-BEVFusion 移植）：加载 4 个 .plan + 自定义 CUDA 算子
+ spconv.pytorch 稀疏编码器，数据流与 C++ `bevfusion.cpp` forward 对齐。

## 与 C++ 的对应关系

| Python（bf/） | C++（deploy/cpp/src/bevfusion/） |
| --- | --- |
| runtime.py（4 个 .plan 加载执行） | bevfusion.cpp（TRT engine 编排） |
| cuda_ops.compute_depth | camera-depth.cu |
| cuda_ops.compute_geometry | camera-geometry.cu（排序/interval 在 Python） |
| cuda_ops.bevpool | camera-bevpool.cu |
| cuda_ops.voxelize | lidar-voxelization.cu |
| sparse.py（spconv.pytorch + ckpt 权重） | lidar-scn.cpp（libspconv + onnx） |
| infer.decode | head-transbbox.cu decode_kernel |
| infer.normalize | camera-normalization.cu |

## 目录结构

```
bf/
├── __init__.py
├── setup.py            # 编译 bevfusion_ops（python setup.py build_ext --inplace）
├── src/
│   └── bevfusion_ops.cu   # 4 个自定义 CUDA 算子（depth/geometry/bevpool/voxelize）
├── cuda_ops.py         # 算子封装（几何排序、interval、体素分桶等 torch 侧逻辑）
├── runtime.py          # .plan 加载 + TRT 执行（IO fp16:chw）
├── sparse.py           # spconv.pytorch SparseEncoder（权重来自 ckpt）
└── infer.py            # 推理编排 + head decode（BevfusionBf）
```

## 编译自定义算子（需 nvcc）

```bash
cd 3rdparty/deeplearning/deploy/python/bf
python setup.py build_ext --inplace
```

产物 bevfusion_ops.*.so 与 bf/ 同目录，import 即用。沙箱无 GPU/nvcc，
编译与运行必须在目标机（conda env bevfusion）。

## 模型与数据

- 4 个 .plan：`3rdparty/deeplearning/model/<name>/build/*.plan`
  （由 deploy/build_trt_engine.sh 构建，--fp16 + IO fp16:chw，Python/C++ 共用）
- SCN 权重：直接加载训练 checkpoint（`models.build_model` + `load_model_checkpoint`），
  与导出的 lidar.backbone.xyz.onnx 同源
- 输入：example-data 目录（6 jpg + camera2lidar/camera_intrinsics/lidar2image/
  img_aug_matrix .tensor + points.tensor）

## 运行

```bash
cd 3rdparty/deeplearning
python deploy/python/scripts/run_bf.py configs/det_convfuser.py ckpts/bevfusion-det-bf.pth \
    --data outputs/example-data --model model/resnet50
```

输出：检测框列表（id/score/position/size/rot/vel）。

## 与 C++ 结果对拍

- 数据流与 C++ 一致（normalize -> backbone -> depth -> bevpool -> vtransform
  -> sparse -> fuser -> head -> decode）
- 同一份 example-data + 同一 ckpt 权重；C++ 跑
  `./deploy/cpp/build/bevfusion outputs/example-data resnet50 fp16`
  对比检测框数量与坐标（float 精度差异允许在 1e-2 量级）
