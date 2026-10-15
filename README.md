# bevfusion_flashocc — 统一纯 PyTorch BEV 感知仓

`../bevfusion`(MIT-BEVFusion 重写,det/seg/multitask)与 `../flashocc`(FlashOCC 重写,
occupancy)的合并项目:**零 mmcv/mmdet/mmdet3d/mmengine/torchpack 依赖**,纯 torch;
**bevpool 统一为 BEVPoolv2**(`ops/bev_pool_v2`,BEVDepth 式 ranks API + 折叠适配
wrapper `bev_pool_v2_sum` 服务 det/seg 调用点);**统一顶层模型 `BevPerception`**,
heads 字典区分三任务:`object`(TransFusionHead,det)/ `map`(BEVSegmentationHead,seg)/
`occ`(BEVOCCHead2D,occupancy),配置启用任意子集。旧两仓保留作参照。

| | det | seg | multitask | occ(r50/M0/4d-stereo/stbase) |
|---|---|---|---|---|
| 编码器 | Swin-T+GLSSFPN+DepthLSS / lidar SparseEncoder | 同 det(LSS) | 同 det | R50\|Swin-B + CustomFPN + BEVPoolv2 LSS(+立体/时序) |
| 解码 | ConvFuser → SECOND/SECONDFPN | 同 det | 同 det | CustomResNet+FPN_LSS(独立 occ 链) |
| 头 | object | map | object+map | occ |
| 金标准 | mini-val NDS/mAP **0.585** | mean IoU@max **0.718** | 无官方权重 | mIoU **24.09** + 官方 81 pred.npz 逐 token **100%** |
| 权重 | ckpts/bevfusion-det-bf.pth(582 strict) | -bf.pth(465 strict) | 从零训 | flashocc-r50-256x704-bff.pth(430 strict) |

**已验证**(RTX 5070 Laptop sm_120 / torch 2.8.0+cu128 / CUDA 12.8 / Python 3.10):
上表全部 strict 加载与金指标;bev_pool_v2_sum 与纯 torch 参照逐位等价(fwd/bwd);
occ 官方 dump 逐体素 100.0000%;训练闭环(AMP+EMA+MultiStep/Cyclic/Cosine+resume)。

## 目录结构

```
bevfusion_flashocc/
├── configs/            # common.py(共享+occ 段) + det_convfuser / seg_fusion_bev256d2_lss /
│                       # multitask_fusion / flashocc_r50(_m0/_4d_stereo/stbase_512x1408)
├── datasets/           # definitions(det/seg/occ 三分类法单一来源)/ dataset(NuScenesDataset+
│                       # CBGSDataset+NuScenesOccDataset)/ dataloader(collate 按样本键分派)/
│                       # postprocess(devkit det + map IoU + occ mIoU, evaluate 按结果键分派)/
│                       # geometry/ devkit_compat / preprocess/(23 变换并集)
├── models/             # perception.py(BevPerception 统一顶层+build_model)/
│                       # occ_encoder.py(OccEncoder/Stereo4D,不含头)/ ema.py /
│                       # backbones/{swin,resnet,second,sparse_encoder} / necks/{glss,second,custom_fpn}/
│                       # vtransforms/{base,lss,depth_lss}(det/seg, v2 调用点)+{occ_lss,occ_lss_depth,depthnet}/
│                       # fusers/conv / heads/{transfusion,segm_vanilla,occ_head} / core/ utils/
├── ops/                # bev_pool_v2(唯一 bevpool: v2 + v2_sum 折叠适配 + scatter/ONNX 符号)/
│                       # voxel/ iou3d/;setup.py 一次编译 3 扩展
├── scripts/            # config/trainer(并集引擎)/tester + train/test/visualize(det|seg|occ)/
│                       # vis_bev/vis_occ/render/export + env/train/eval/visualize/prepare_data .sh
├── tools/              # convert_nuscenes / create_gt_database / remap_pretrained /
│                       # create_data(occ infos 富化)/ remap_occ(flashocc→统一树前缀)
├── tests/              # 10 个测试(ops/dataset/model_ckpt/geometry/match/swin/sparse/heads/ema/vtransform)
├── ckpts/              # 5 官方权重 + flashocc -bff.pth(remap 产物)
├── docs/MODEL_IO.md    # 前向 I/O 契约(det/seg + occ)
└── data -> ../../data/nuscenes
```

## 安装

```bash
conda activate bevfusion   # python 3.10, torch 2.8.0+cu128, spconv-cu126, numba,
                           # pyquaternion, shapely, tqdm, pyyaml, tensorboard, nuscenes-devkit
pip install open3d         # 仅 scripts/vis_occ.py 需要(懒导入,其余功能非必需)

# 编译 3 个 CUDA 扩展(sm_120;其他 GPU 改 TORCH_CUDA_ARCH_LIST)
TORCH_CUDA_ARCH_LIST="12.0" PATH=/usr/local/cuda-12.8/bin:$PATH \
  python setup.py build_ext --inplace
```

## 数据准备

与旧仓共享 `data/nuscenes -> ../../data/nuscenes`:
- det/seg infos:`scripts/prepare_data.sh [--with-gt-db]`(convert_nuscenes → 323/81)
- occ infos:bevdetv2 pkl + `gts/`;富化步骤 `python tools/create_data.py --dataroot data/nuscenes --version v1.0-mini --tag bevdetv2-nuscenes`
- 官方权重 remap(已随仓提供产物):`python tools/remap_pretrained.py ckpts/bevfusion-det.pth ckpts/bevfusion-det-bf.pth`、`python tools/remap_occ.py ckpts/flashocc-r50-256x704.pth ckpts/flashocc-r50-256x704-bff.pth`

## 训练 / 评测

```bash
# 快捷脚本(任务表统一): det|seg|multitask|occ|m0|stereo|stbase
scripts/train.sh occ --debug_iters 2 --no_eval     # 冒烟
scripts/eval.sh det                                 # 金标准 NDS/mAP 0.585
scripts/eval.sh seg                                 # 金标准 IoU@max 0.718
scripts/eval.sh occ --save_dir outputs/results      # 金标准 mIoU 24.09 + pred.npz 转储

# 直接入口(多卡 torchrun --nproc_per_node=N)
python scripts/train.py configs/det_convfuser.py --load_from ckpts/bevfusion-det-bf.pth
python scripts/train.py configs/flashocc_r50.py --load_from ckpts/flashocc-r50-256x704-bff.pth
python scripts/test.py configs/multitask_fusion.py CKPT          # det+seg 联合评测
# 常用: --epochs N --batch_size N --work_dir DIR --resume_from CKPT --no_amp
#       --no_eval --debug_iters N --options a.b=c(点分覆盖)
```

训练日志(仅主进程):stdout + `outputs/<run>/scalars.jsonl`(逐 log 步 lr/各 loss/
显存/耗时,val 后附指标行)+ `outputs/<run>/tf_logs/`(tensorboard,
`log_config.tensorboard=False` 关闭)。EMA(cfg `ema` 非空时)每 iter 更新、
每 epoch 存 `epoch_N_ema.pth`(occ 官方评测工件格式)。

## 可视化

```bash
# det/seg: 每场景 GIF(6 相机面板 + BEV)
scripts/visualize.sh det --show_gt
python scripts/visualize.py configs/seg_fusion_bev256d2_lss.py ckpts/bevfusion-seg-bf.pth --task seg
# occ: 离线 pred.npz 渲染 / 在线推理 BEV+环视条带
python scripts/visualize.py --task occ --results_dir outputs/results --out outputs/vis
python scripts/visualize.py configs/flashocc_r50.py ckpts/flashocc-r50-256x704-bff.pth --task occ --with_rgb
# occ 批量合成图(6 相机+pred/gt BEV+图例+地平面反投影+点云高度)
python scripts/vis_bev.py outputs/results --root_path data/nuscenes --draw_gt
# occ 交互式 3D 体素(open3d,需 X 显示;--interactive 才阻塞旋转)
python scripts/vis_occ.py outputs/results --root_path data/nuscenes --draw_gt --vis_frames 8
```

三入口默认输出统一到 `outputs/vis`。ONNX 导出(det/seg,4 子图):
`python scripts/export.py configs/det_convfuser.py CKPT --verify`。

## 测试套件(仓库根目录运行)

```bash
python tests/test_ops.py            # bev_pool_v2 / v2_sum / scatter 参照 + voxel + iou3d
python tests/test_dataset.py        # det/seg/occ 三管线契约 + collate 双分派 + mIoU 数学
python tests/test_model_ckpt.py     # 582/465/430 三 strict + 三路径前反向 + M0 形状
python tests/test_geometry.py tests/test_match.py tests/test_heads.py
python tests/test_swin.py tests/test_sparse_encoder.py
python tests/test_ema.py tests/test_vtransform.py
```

## 数值保真要点(重构红线,权重已烘焙这些行为)

1. **BGR 归一化怪癖**(occ):PIL(RGB) 经 mmcv `imnormalize(to_rgb=True)` 实为通道 0↔2
   再按 RGB 统计归一 —— `datasets/preprocess/loading.py::mmlab_normalize` 逐位复刻
2. occ `bda = flip @ (scale @ rot)`;`sensor2keyego` 走 float64 再回 float32;
   frustum `(u,v,d)` 序、SID 公式、ranks 沿用原 torch.range 含端点语义
3. occ `collapse_z` 为 z-major 通道折叠;head `view(B,Dx,Dy,Dz,18)`;
   CE:`reduction='none'` → ×mask_camera → `sum/(mask.sum()+eps)`;
   mIoU 为相机 mask 内前 17 类 nanmean(free 除外)
4. **bevpool v2 统一**:det/seg 调用点经 `bev_pool_v2_sum`(depth=ones 退化配置,
   ranks_feat 为排列保证反向逐位等价);轴序 (Dz,Dy,Dx)=(z,x,y) 为 BEVFusion 约定
   (与 occ 的 (z,y,x) 不同,切勿混用);内核 fp32-only,几何路径 autocast 关闭
5. CustomResNet downsample 是 3×3 **带 bias 无 BN** 的裸 Conv(键名承重);
   EMA 覆盖 params + float buffers,衰减 `0.999·(1-e^-x/2000)`
6. spconv 1.x→2.x 键布局(remap_pretrained);det/seg 几何 fp32 岛;
   训练 autocast `cache_enabled=False`(Swin+梯度检查点与 AMP 缓存冲突)

## 已知限制

- 三头(det+seg+occ)**联合训练配置未提供**:occ 管线(BGR 怪癖+bda)与 det/seg 管线
  (ImageAug3D+lidar_aug)增广协议不同,统一需新变换设计;架构(heads 字典)已就绪
- 未移植:radar 分支、CenterHead、DynamicScatter/PointPillars、pointnet/paconv、
  panoptic/ray-iou/TRT 端到端(ONNX 导出保留 det/seg 4 子图)
- 4d-stereo/stbase 无本地官方权重,仅前向/反向/对齐路径验证
- occ 官方全量指标(32.08/37.84/43.52)需 v1.0-trainval + det 预训练
