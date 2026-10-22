# TRT 部署（deploy/）

C++ 推理完全移植 NVIDIA CUDA-BEVFusion（deploy/cpp/），不再区分预处理/推理/后处理三层，
直接采用参考实现 src/bevfusion/ 平铺结构。Python 侧负责模型导出（deploy/export_bf.py），
按 CUDA-BEVFusion 模型结构导出 5 个 onnx 并构建 engine。

## 目录结构

```
deploy/
├── README.md           # 本文件（部署总览）
├── export_bf.py        # 模型导出 + engine 构建主入口（CUDA-BEVFusion 结构）
├── python/             # Python 导出/校验库
│   ├── README.md       # Python 侧说明
│   ├── lean/           # NVIDIA qat/lean 移植（SCN onnx 导出）
│   │   ├── exptool.py  #   SparseConvolution 符号 trace / onnx 图构建
│   │   ├── funcs.py    #   BN 融合 / ReLU 融合 / 权重加载
│   │   ├── tensor.py   #   tensor 文件格式（magic 0x33ff1101）
│   │   └── quantize.py #   PTQ 量化接口（非量化模型跳过）
│   └── (scripts/builder 等历史保留)
├── cpp/                # C++ 推理（CUDA-BEVFusion 完整移植，见 cpp/README.md）
├── tools/
│   ├── export_example_data.py  # 生成 example-data（6 jpg + .tensor）
│   └── export_cpp_input.py     # 旧输入导出（历史保留）
└── launch/             # 旧启动脚本（历史保留，新流程不依赖）
```

## 使用流程

1. 导出模型 + 构建 engine（Python，见 python/README.md）
2. 按 main.cpp 布局放置模型（见 cpp/README.md「模型布局」）
3. 生成 example-data 输入（见 cpp/README.md「数据」）
4. 构建并运行 C++ 推理（见 cpp/README.md）

## 产物

- outputs/onnx_bf/：5 个 onnx
  - camera.backbone.onnx：img+depth → camera_feature(HWC) + camera_depth_weights
  - camera.vtransform.onnx：feat_in(1,80,360,360) → feat_out(1,80,180,180)
  - fuser.onnx：camera+lidar → middle(1,512,180,180)
  - head.bbox.onnx：middle → score/rot/dim/reg/height/vel
  - lidar.backbone.xyz.onnx：稀疏编码器（libspconv 运行时解析，不走 TRT）
- outputs/trt_bf/：4 个 .plan（camera.backbone / camera.vtransform / fuser / head.bbox）

## TRT 版本说明

- TRT 10.x：trtexec 支持 legacy `--fp16`，直接构建 FP16 engine（推荐）
- TRT >= 10.12 / 11.x：强类型（precision 由 ONNX 图表达），`--fp16` 被移除，
  需要 ModelOpt AutoCast 转混合 FP16 onnx（--autocast-fp16，需 pip install nvidia-modelopt），
  否则构建 FP32 engine
- 版本在构建时自动探测（builder.py 检查 trtexec --help 是否含 --fp16）

## 自定义算子归属

bev_pool / 深度投影 / 体素化 / 稀疏卷积等自定义算子在 CUDA-BEVFusion 中都是
C++ 侧独立 CUDA kernel（src/bevfusion/*.cu）或 libspconv 运行时，不进入 TRT engine；
TRT engine 只覆盖 camera.backbone / vtransform / fuser / head.bbox 四个网络部分。
