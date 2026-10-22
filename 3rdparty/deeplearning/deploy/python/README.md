# Python 导出/校验（deploy/python/）

C++ 推理前置：模型导出 + engine 构建。推理本体在 deploy/cpp/（CUDA-BEVFusion）。

## CUDA-BEVFusion 结构导出（deploy/export_bf.py）

5 个 onnx 与 CUDA-BEVFusion 对齐：camera.backbone（img+depth → camera_feature
HWC + camera_depth_weights，softmax 在图中）、camera.vtransform（feat_in 360x360
→ feat_out 180x180 = downsample）、fuser（camera+lidar → middle）、head.bbox
（middle → score/rot/dim/reg/height/vel，decode 由 C++ kernel 完成）、
lidar.backbone.xyz（稀疏编码器，libspconv 解析）。

cd 3rdparty/deeplearning
python deploy/export_bf.py configs/det_convfuser.py ckpts/bevfusion-det-bf.pth \
    --outdir outputs/onnx_bf --engine-dir outputs/trt_bf --verify --fp16

说明：SCN onnx 需要 GPU（模型 .half() trace，BN 自动融合进权重）；
--fp16 对 TRT 10.x 生效（trtexec --fp16），TRT 11 用 --autocast-fp16。
