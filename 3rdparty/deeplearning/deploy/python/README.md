# Python 导出/校验（deploy/python/）

C++ 推理前置：模型导出 + engine 构建。推理本体在 deploy/cpp/（CUDA-BEVFusion）。

## 目录结构

- export_bf.py：CUDA-BEVFusion 结构导出主入口（5 个 onnx + engine 构建 + verify）
- lean/：NVIDIA qat/lean 移植（SCN onnx 导出，去除 mmdet3d 依赖）
- python/：Python 推理库（builder.py trtexec 构建等，历史保留）
- tools/export_example_data.py：example-data 输入导出（见 cpp/README.md）

## 导出内容

5 个 onnx 与 CUDA-BEVFusion 模型结构对齐：

- camera.backbone：img + depth → camera_feature（HWC）+ camera_depth_weights
  （backbone+neck+dtransform+depthnet 合并导出，softmax 在图中）
- camera.vtransform：feat_in 360x360 → feat_out 180x180（= downsample）
- fuser：camera + lidar → middle
- head.bbox：middle → score/rot/dim/reg/height/vel（decode 由 C++ kernel 完成）
- lidar.backbone.xyz：稀疏编码器（voxels + coors → dense BEV，libspconv 解析）

## 导出 + 构建

cd 3rdparty/deeplearning
python deploy/export_bf.py configs/det_convfuser.py ckpts/bevfusion-det-bf.pth \
    --outdir outputs/onnx_bf --engine-dir outputs/trt_bf --verify --fp16

参数：--verify 用 onnxruntime 逐图与 torch 对照（tol 5e-3）；--onnx-only 只导出
onnx；--fp16 构建 FP16 engine（TRT 10.x 走 trtexec --fp16，TRT 11 用
--autocast-fp16 需要 nvidia-modelopt）。

## 说明

- SCN onnx 需要 GPU（模型 .half() trace，BN 自动融合进权重）
- 产物：outputs/onnx_bf/*.onnx（5）+ outputs/trt_bf/*.plan（4）
- 单独重建 engine：python deploy/python/builder.py --onnx_dir outputs/onnx_bf \
  --engine_dir outputs/trt_bf --graphs camera.backbone,camera.vtransform,fuser,head.bbox --fp16
