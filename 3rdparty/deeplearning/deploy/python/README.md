# Python TRT 推理（deploy/python/）

Python 侧 TRT 推理，结构与 C++ 侧（`deploy/cpp/`：include/ + src/）对齐：
**scripts/ 放命令行入口，包（deploy.python）放库实现**。

```
deploy/python/
  README.md
  scripts/                 # 命令行入口
    pipeline.py            #   三层推理（Preprocessor -> Inferencer -> Postprocessor）
    visualize.py           #   推理结果可视化（jpg / gif）
  __init__.py              # deploy.python 包（导出三层组件）
  preprocess.py            # Preprocessor：深度投影 / 体素化
  inference.py             # Inferencer：TRT engine 编排 + torch 自定义算子
  postprocess.py           # Postprocessor：检测解码 + NMS
  runtime.py               # TrtEngine + load_engines
  builder.py               # EngineBuilder（trtexec CLI 构建 engine）
  config.py                # DeployConfig
  graphs.py                # GRAPH_IO + DET_HEAD_KEYS（tensor 名唯一来源）
  bev_pool_torch.py        # camera_bev_from_depthnet（几何 + bev_pool）
```

入口脚本通过 `_DL_ROOT`（上 4 级 = 模型库根）注入 `sys.path` 后导入
`deploy.python` 库与训练侧模块，需在 `3rdparty/deeplearning` 下运行。

## 用法（cd 3rdparty/deeplearning）

```bash
# 前置导出（Python/C++ 共用）：ONNX 分模块导出 + trtexec 构建 engine
python deploy/export.py configs/det_convfuser.py ckpts/bevfusion-det-bf.pth \
    --task det --verify --fp16

# 三层推理：直接用 engine，不需要 .pth
python deploy/python/scripts/pipeline.py configs/det_convfuser.py --task det \
    --engine_dir outputs/trt --sample-index 5 --dump outputs/trt/sample5.npz

# 与 torch 参考对照（需 checkpoint）
python deploy/python/scripts/pipeline.py configs/det_convfuser.py \
    --checkpoint ckpts/bevfusion-det-bf.pth --task det \
    --engine_dir outputs/trt --verify

# 可视化（复用 dump 或在线推理）
python deploy/python/scripts/visualize.py configs/det_convfuser.py \
    --npz outputs/trt/sample5.npz --out outputs/vis
# 或通过 launch 脚本
bash deploy/launch/infer_trt.sh det
bash deploy/launch/visualize_trt.sh --sample-index 5
```

## 说明

- 仅构建 engine（已有 ONNX）：`python deploy/python/builder.py --task det --fp16`
- C++ 推理输入/权重/参考导出：`deploy/tools/export_cpp_input.py`（见
  `deploy/cpp/README.md`）
- 部署总览见 [`../README.md`](../README.md)

## CUDA-BEVFusion 结构导出（deploy/export_bf.py）

C++ 侧 `deploy/cpp/` 已完全移植 NVIDIA CUDA-BEVFusion 推理代码，模型导出
需与其结构对齐（5 个 onnx，engine 命名与 `deploy/cpp/src/main.cpp`
`create_core` 一致）：

```
camera.backbone    img + depth -> camera_feature(B,N,fH,fW,C) + camera_depth_weights(B,N,D,fH,fW)
camera.vtransform  feat_in(1,80,360,360) -> feat_out(1,80,180,180)
fuser              camera(1,80,180,180) + lidar(1,256,180,180) -> middle(1,512,180,180)
head.bbox          middle -> score/rot/dim/reg/height/vel
lidar.backbone.xyz voxels + coors -> dense BEV（libspconv 解析）
```

导出 + 构建：

```bash
cd 3rdparty/deeplearning
python deploy/export_bf.py configs/det_convfuser.py ckpts/bevfusion-det-bf.pth \
    --outdir outputs/onnx_bf --engine-dir outputs/trt_bf --verify --fp16
```

说明：
- `camera.backbone` 合并导出（backbone+neck+dtransform+depthnet），
  `camera_depth_weights` 已 softmax（bevpool kernel 直接乘）；
  `camera_feature` 为 HWC 布局（`(B,N,fH,fW,C)`）。
- `head.bbox` 输出 6 个原始预测（decode 由 C++ `head-transbbox.cu` kernel 完成）。
- `lidar.backbone.xyz.onnx` 由 `deploy/lean/`（NVIDIA qat/lean 移植）导出，
  依赖 GPU（模型 `.half()` trace）与 spconv；非量化模型自动跳过 PTQ 处理。
- 运行输入：`deploy/tools/export_example_data.py`（example-data：6 jpg +
  `.tensor` 矩阵 + points.tensor），见 `deploy/cpp/README.md`。

