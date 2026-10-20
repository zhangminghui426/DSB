# BEVFusion(bf)模型输入 / 输出说明

以检测配置 `configs/det_convfuser.py`(camera+lidar 融合)与分割配置
`configs/seg_fusion_bev256d2_lss.py` 为准。B = batch size。

## 坐标系约定(实测确认)

- **LiDAR 自车架**:x = 右,y = 前,z = 上(用相邻帧 ego 位移标定验证)。
- **3D 框**:(x, y, z, w, l, h, yaw, vx, vy) —— z 为**重力中心**;yaw 绕 z 轴(LiDAR 架);
  速度为 LiDAR 架下 (vx, vy)。`LiDARInstance3DBoxes.corners` 按**底中心**解释 z,
  因此渲染/取角点前需 `z -= h/2`(官方 visualize.py 同)。
- **相机内参**:标准针孔(z 为光轴深度),与 `lidar2camera`(OpenCV 轴:右/下/前)配合。
- **BEV 网格索引**:mask 为 x-first,即 `masks[k, x_idx, y_idx]`。

## 输入(forward 的关键字参数 = collate 后的 batch 键)

| 键 | 维度 | 含义 |
|---|---|---|
| `img` | (B, 6, 3, 256, 704) float | 6 路环视图像,顺序 [FRONT, FRONT_RIGHT, FRONT_LEFT, BACK, BACK_LEFT, BACK_RIGHT];ImageNet 归一化;经 ImageAug3D(test:resize 0.48 + 中心裁 256×704;train:resize∈[0.38,0.55] + 随机裁/翻转/旋转 ±5.4°) |
| `points` | list[B] of (N_i, 5) float | LiDAR 点云,列 = (x, y, z, intensity, Δt);Δt 为相对关键帧秒差(关键帧=0)。det:test 合并 9 个 sweeps(N≈20 万)、train 仅关键帧(sweeps_num=0);seg:train/test 均 9 sweeps |
| `camera_intrinsics` | (B, 6, 4, 4) | 针hole 内参 K(原图尺度) |
| `camera2ego` | (B, 6, 4, 4) | 相机 → 自车 |
| `camera2lidar` | (B, 6, 4, 4) | 相机 → LiDAR(vtransform  frustum 反投影用) |
| `lidar2ego` | (B, 4, 4) | LiDAR → 自车 |
| `lidar2camera` | (B, 6, 4, 4) | LiDAR → 相机 |
| `lidar2image` | (B, 6, 4, 4) | = K @ lidar2camera(可视化/GTDepth 用) |
| `img_aug_matrix` | (B, 6, 4, 4) | 图像增广后变换(原图像素 → 增广图像素,含 resize/crop/flip/rotate) |
| `lidar_aug_matrix` | (B, 4, 4) | LiDAR BEV 增广(GlobalRotScaleTrans + RandomFlip3D;test 为单位阵) |
| `metas` | list[B] dict | token / timestamp / filename(6 原图路径)/ lidar_path / box_type_3d 类等 |
| `depths` | (B, 6, 256, 704) float | GTDepth:LiDAR 稀疏深度投影到各增广相机(DepthLSSTransform 不启用深度损失,故仅占位) |
| `gt_bboxes_3d` | list[B] LiDARInstance3DBoxes,(M_i, 9) | 训练 GT 框(仅训练) |
| `gt_labels_3d` | list[B] (M_i,) int64 | 10 类索引(仅训练) |
| `gt_masks_bev` | (B, 6, 200, 200) int64 0/1 | BEV 地图 GT,±50 m @ 0.5 m,6 类 [drivable_area, ped_crossing, walkway, stop_line, carpark_area, divider](seg 任务) |

## 内部特征流(det convfuser 配置)

```
img ─ Swin-T(out_indices 1-3) ─ (B·6, {192,384,768}, {32×88,16×44,8×22})
    ─ GeneralizedLSSFPN ─ 3×(B·6, 256, 32×88)
    ─ DepthLSSTransform(深度 bin 118 = [1,60,0.5];frustum 32×88;bev_pool)
    ─ (B, 80, 180×180)                      # xbound ±54 m @0.3 m =360 bin,downsample 2
points ─ Voxelization(0.075 m,±54 m,≤10 点/体素,120k/160k 体素)
       ─ SparseEncoder(spconv,sparse 1440×1440×41)─ (B, 256, 180×180)
ConvFuser: cat(80+256) ─ conv3x3 ─ (B, 256, 180×180)
SECOND ─ [(B,128,180²), (B,256,90²)] ─ SECONDFPN ─ (B, 512, 180×180)
TransFusionHead: 200 proposals,1 层 decoder,hidden 128
seg 分支:voxel 0.1 m/±51.2 m → decoder (B,512,128×128) → BEVSegmentationHead
          → BEVGridTransform(±51.2@0.8 → ±50@0.5)→ (B,6,200×200)
```

## 输出(实测 shape,B=1 示例)

### 检测头原始输出 `pred_dict`

`model.heads['object'](bev_feat, metas)` 返回 **tuple(1 个特征级)→ list(B 个样本)→ dict**:
`pred_dict[0][b]` 的键(P = num_proposals = 200):

| 键 | shape | 含义 |
|---|---|---|
| `center` | (B, 2, P) | 提案框 BEV 中心偏移(x, y,网格单位,解码时 ×voxel×out_size_factor + 锚点) |
| `height` | (B, 1, P) | 高度残差 |
| `dim` | (B, 3, P) | 尺寸 (w, l, h) |
| `rot` | (B, 2, P) | 航向 (sin θ, cos θ) |
| `vel` | (B, 2, P) | 速度 (vx, vy) |
| `heatmap` | (B, 10, P) | 提案分类 logits(10 类) |
| `query_heatmap_score` | (B, 10, P) | 提案锚点处的 dense heatmap 分数(置信度先验) |
| `dense_heatmap` | (B, 10, 180, 180) | BEV 网格上的 dense 分类 heatmap(辅助损失用) |

### 检测推理输出 `model(**batch)`(eval 模式)

返回 **list[B] dict**:

| 键 | shape / 类型 | 含义 |
|---|---|---|
| `boxes_3d` | LiDARInstance3DBoxes,`.tensor` (200, 9) float32 | 解码后 3D 框 (x, y, z, w, l, h, yaw, vx, vy);z 重力中心;LiDAR 架;**不做 NMS**(config `nms_type: null`),200 个提案全返回 |
| `scores_3d` | (200,) float32 | 每框置信度 = sigmoid(heatmap) × query_heatmap_score |
| `labels_3d` | (200,) **int32** | 类别索引 0-9 |

### 检测训练输出(train 模式)

loss 字典(值均为标量 tensor;键含 "loss" 的求和 = 总 loss):

| 键 | 含义 |
|---|---|
| `loss/object/loss_heatmap` | dense heatmap 的 GaussianFocalLoss |
| `loss/object/layer_-1_loss_cls` | 提案分类 FocalLoss(最后一层 decoder;多层时另有 layer_0… ) |
| `loss/object/layer_-1_loss_bbox` | 框回归 L1Loss(code_weights 末两位 vel 权重 0.2) |
| `stats/object/matched_ious` | 匈牙利匹配后的平均 IoU(无梯度,监控用) |

### 分割输出

- eval:`masks_bev` (B, 6, 200, 200) float32,**sigmoid 概率 [0,1]**,x-first,±50 m @ 0.5 m;
  同 dict 附 `gt_masks_bev` (B, 6, 200, 200) int64 0/1 供评测
- train:loss 字典 6 个标量:`loss/map/{drivable_area, ped_crossing, walkway,
  stop_line, carpark_area, divider}/focal`(逐类 sigmoid focal/BCE)
- 头内部:decoder BEV (B, 512, 128, 128) → classifier → (B, 6, 128, 128) →
  BEVGridTransform 双线性重采样 → (B, 6, 200, 200)

### 中间 BEV 特征(供调试)

- det:decoder 输出 (B, 512, 180, 180)(±54 m @ 0.6 m)
- seg:decoder 输出 (B, 512, 128, 128)(±51.2 m @ 0.8 m)

**评测**
- det:`format_results` → `results_nusc.json`(NuScenesBox:translation/size(wlh)/
  rotation 四元数/velocity)→ devkit `DetectionEval(detection_cvpr_2019)` →
  mAP、NDS、每类 AP@距离档、mATE/mASE/mAOE/mAVE/mAAE
- seg:BEV IoU @ 阈值 [0.35…0.65] → 每类 iou@max 与 mean iou@max

## 类别表

- 检测 10 类:car, truck, construction_vehicle, bus, trailer, barrier, motorcycle,
  bicycle, pedestrian, traffic_cone
- 分割 6 类:drivable_area, ped_crossing, walkway, stop_line, carpark_area, divider
  (divider = road_divider + lane_divider 合并)

## Occupancy branch (heads.occ) contract

Batch keys (occ datasets, `PackOcc3DInputs`):
- `img_inputs`: 7-tuple of tensors —
  `imgs (B, N[, *num_frame], 3, H, W)` (BGR-quirk normalized, cam order
  FL/F/FR/BL/B/BR), `sensor2egos (B, N, 4, 4)`, `ego2globals (B, N, 4, 4)`,
  `intrins (B, N, 3, 3)`, `post_rots (B, N, 3, 3)`, `post_trans (B, N, 3)`,
  `bda (B, 3, 3)`; sequential (4D) variants stack frames cam-major and poses
  frame-major
- `gt_depth (B, N, H, W)` sparse metric depth (train, depth-supervised variants)
- `voxel_semantics (B, 200, 200, 16)` uint8 labels (0..17, 17=free),
  `mask_lidar` / `mask_camera` same shape (train + eval GT)
- `metas`: list[B] dicts (token / scene_name / occ_gt_path / poses)

Grid convention: x/y in [-40, 40] m @ 0.4 m (200 cells), z in [-1, 5.4] m
(16 cells); BEV renderings flip both axes (+x up, +y left).

Feature flow (OccEncoder): imgs -> ResNetBackbone(out_indices [2,3]) |
Swin-B([0,2,3]) -> CustomFPN (B*N, 256, fH, fW) -> LSSViewTransformer(BEVPoolv2,
frustum (D, fH, fW, 3), collapse_z z-major) -> (B, C*Dz, 200/16?, ...) ->
CustomResNet + FPN_LSS -> (B, 256, 200, 200)? — see models/occ_encoder.py;
BEVOCCHead2D: channel-to-height view(B, Dx, Dy, Dz, 18) -> logits
(B, 18*Dz, Dx, Dy); eval argmax -> `pred_occ (200, 200, 16)` uint8 per sample.

Stereo4D: num_frame = key + num_adj + extra_ref; loop fid high->0; extra ref
frame contributes only the /4-stride stereo feature; adjacent frames run under
no_grad; BEV concat order [prev, key]; `align_after_view_transfromation`
(test-time) grid_sample-realigns adjacent BEV feats to the key ego frame.
