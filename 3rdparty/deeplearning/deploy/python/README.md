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
