# Python 导出/校验（deploy/python/）

C++ 推理前置：模型导出 + engine 构建。推理本体在 deploy/cpp/（CUDA-BEVFusion 移植）。
本目录负责把训练侧模型按 CUDA-BEVFusion 模型结构导出为 5 个 onnx；
engine 构建统一走 CUDA-BEVFusion 原版脚本 deploy/build_trt_engine.sh（Python/C++ 共用）。

## 目录结构

```
deploy/
├── export_bf.py          # 主入口：CUDA-BEVFusion 结构 ONNX 导出 + verify
├── lean/                 # NVIDIA qat/lean 移植（SCN onnx 导出，去除 mmdet3d 依赖）
│   ├── exptool.py        #   SparseConvolution 符号 trace、onnx 节点构建
│   ├── funcs.py          #   layer_fusion_bn（BN 融合进卷积权重）
│   ├── tensor.py         #   tensor 文件格式工具
│   └── quantize.py       #   PTQ 量化接口（非量化模型自动跳过）
├── build_trt_engine.sh  # CUDA-BEVFusion 原版 engine 构建脚本（Python/C++ 共用）
├── python/               # Python 推理库（历史保留，preprocess/inference/postprocess/runtime）
└── tools/
    └── export_example_data.py  # example-data 输入导出（见 cpp/README.md）
```

## 导出模型结构

5 个 onnx 与 CUDA-BEVFusion 完全对齐（engine 绑定名与 deploy/cpp/src/main.cpp
create_core 一致）：

| onnx | 输入 → 输出 | 说明 |
| --- | --- | --- |
| camera.backbone | img(B,N,3,256,704) + depth → camera_feature(B,N,32,88,C) + camera_depth_weights(B,N,D,32,88) | backbone+neck+dtransform+depthnet 合并导出；camera_feature 为 HWC 布局（bevpool kernel 按此取数）；depth_weights 已 softmax |
| camera.vtransform | feat_in(1,80,360,360) → feat_out(1,80,180,180) | = vtransform.downsample，输入是 bev_pool 输出 |
| fuser | camera(1,80,180,180) + lidar(1,256,180,180) → middle(1,512,180,180) | transfusion decoder |
| head.bbox | middle(1,512,180,180) → score/rot/dim/reg/height/vel | decode 由 C++ head-transbbox.cu kernel 完成（reg 按 (2,num) 布局） |
| lidar.backbone.xyz | voxels(1,5) + coors(1,4) → dense BEV(1,256,180,180) | 稀疏编码器；libspconv 运行时解析，不走 TRT |

注意：此结构与旧版导出（img→feats、feats+depth→depthnet 分离 + 外部 downsample
kernel）不兼容，必须按上表重新导出。

## 导出

```bash
cd 3rdparty/deeplearning && python deploy/export_bf.py configs/det_convfuser.py ckpts/bevfusion-det-bf.pth --outdir outputs/onnx_bf --verify
```

参数：--outdir（onnx 输出目录，默认 outputs/onnx_bf）、--verify（用 onnxruntime
CPU 逐图与 torch 对照，worst rel diff < 5e-3）。

产物：outputs/onnx_bf/ 下 5 个 onnx。

## engine 构建（CUDA-BEVFusion 原版脚本）

脚本在 deploy/ 根目录（Python/C++ 共用，模型统一放 3rdparty/deeplearning/model/<name>/）：

```bash
cd 3rdparty/deeplearning
cp outputs/onnx_bf/camera.backbone.onnx outputs/onnx_bf/camera.vtransform.onnx outputs/onnx_bf/fuser.onnx outputs/onnx_bf/head.bbox.onnx model/resnet50/
DEBUG_MODEL=resnet50 DEBUG_PRECISION=fp16 bash deploy/build_trt_engine.sh
```

脚本特性（与参考一致）：trtexec --fp16 + IO 格式强制 fp16:chw（fuser 输出 HALF，
与 main.cpp transfusion.cu 断言匹配）、workspace 2GB、导出 layer info JSON、
已存在 .plan 跳过。产物：model/resnet50/build/*.plan（camera.backbone /
camera.vtransform / fuser / head.bbox）。

## SCN onnx 导出说明

- 需要 GPU：模型 .half() 推理 trace（单点占位输入 voxels(1,5)+coors(1,4)）
- BN 融合：参考是 PTQ 模型（BN 已折叠），我们的 FP 模型在导出前自动执行
  funcs.layer_fusion_bn（conv+bn 融合进权重，ReLU 保留为独立节点）
- 非量化模型跳过 PTQ 属性设置（quantize 为 stub）
- act_type：spconv 2.x 是 tv.gemm.Activation 枚举，导出时自动序列化为
  字符串（None_→"None"、ReLU→"ReLU"，与 3DSparseConvolution parser 一致）
- 权重布局：spconv 2.x OKI (Cout,K,D,H,Cin)，fuse_bn 按输出通道缩放（布局无关）

## 模型布局

engine 命名与 deploy/cpp/src/main.cpp 读取路径一致（model/<name>/build/*.plan），
lidar.backbone.xyz.onnx 直接放 model/<name>/，放置方法见
[cpp/README.md#模型布局](../cpp/README.md#模型布局)。
