"""nuScenes MULTITASK: joint detection + BEV segmentation in one fusion model.

Same geometry/encoders/decoder as det_convfuser.py (±54m, voxel 0.075,
DepthLSSTransform, SparseEncoder 1440x1440x41), plus the seg branch:

- pipelines carry gt_masks_bev (LoadBEVSegmentation; sweeps_num=9 as in the
  seg recipe — denser lidar context for the shared BEV features)
- heads = TransFusionHead (object) + BEVSegmentationHead (map), trained
  jointly: loss/object/* + loss/map/* (loss_scale 1.0 each)
- BEV feature grid = 108m / (0.075 * out_size_factor 8) = 180x180, cell 0.6m
  -> the map head resamples input_scope ±54@0.6 to output ±50@0.5 (200x200,
  matching the LoadBEVSegmentation canvas)
- eval merges both heads per sample; dataset.evaluate() runs devkit det eval
  (mAP/NDS) AND BEV IoU in one pass

Single-GPU (8GB) recipe: batch 1 + Swin gradient checkpointing (with_cp) +
AMP (batch 2 OOMs from allocator fragmentation even with with_cp; try
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to push it).

NOTE: no official multitask checkpoint exists — train from scratch. The
trainer's --load_from is strict=True, so single-task ckpts/*-bf.pth weights
will NOT load into this two-head model.

Usage:
    python scripts/train.py configs/multitask_fusion.py
    python scripts/test.py configs/multitask_fusion.py CKPT --eval bbox
"""

from configs.common import *  # noqa: F401,F403

point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
voxel_size = [0.075, 0.075, 0.2]
max_epochs = 6
checkpoint_config = dict(interval=1, max_keep_ckpts=1)
batch_size = 1  # single-GPU (8GB): dual heads + 9-sweep lidar is the heaviest recipe

train_pipeline = make_train_pipeline(
    point_cloud_range, sweeps_num=9, max_epochs=max_epochs, with_bev_seg=True
)
test_pipeline = make_test_pipeline(point_cloud_range, with_bev_seg=True)
data = make_data_config(train_pipeline, test_pipeline, samples_per_gpu=batch_size)

model = dict(
    type="BevPerception",
    encoders=dict(
        camera=dict(
            # with_cp (gradient checkpointing): required on 8GB
            backbone=swin_backbone(drop_path_rate=0.2, with_cp=True),
            neck=CAMERA_NECK,
            vtransform=dict(
                type="DepthLSSTransform",
                in_channels=256,
                out_channels=80,
                image_size=image_size,
                feature_size=[32, 88],
                xbound=[-54.0, 54.0, 0.3],
                ybound=[-54.0, 54.0, 0.3],
                zbound=[-10.0, 10.0, 20.0],
                dbound=[1.0, 60.0, 0.5],
                downsample=2,
            ),
        ),
        lidar=dict(
            voxelize=dict(
                max_num_points=10,
                point_cloud_range=point_cloud_range,
                voxel_size=voxel_size,
                max_voxels=[120000, 160000],
            ),
            backbone=sparse_encoder([1440, 1440, 41]),
        ),
    ),
    fuser=FUSER,
    decoder=DECODER,
    heads=dict(
        occ=None,
        object=dict(
            type="TransFusionHead",
            num_proposals=200,
            auxiliary=True,
            in_channels=512,
            hidden_channel=128,
            num_classes=10,
            num_decoder_layers=1,
            num_heads=8,
            nms_kernel_size=3,
            ffn_channel=256,
            dropout=0.1,
            bn_momentum=0.1,
            activation="relu",
            train_cfg=dict(
                dataset="nuScenes",
                point_cloud_range=point_cloud_range,
                grid_size=[1440, 1440, 41],
                voxel_size=voxel_size,
                out_size_factor=8,
                gaussian_overlap=0.1,
                min_radius=2,
                pos_weight=-1,
                code_weights=[1, 1, 1, 1, 1, 1, 1, 1, 0.2, 0.2],
                assigner=dict(
                    type="HungarianAssigner3D",
                    iou_calculator=dict(type="BboxOverlaps3D", coordinate="lidar"),
                    cls_cost=dict(type="FocalLossCost", gamma=2.0, alpha=0.25, weight=0.15),
                    reg_cost=dict(type="BBoxBEVL1Cost", weight=0.25),
                    iou_cost=dict(type="IoU3DCost", weight=0.25),
                ),
            ),
            test_cfg=dict(
                dataset="nuScenes",
                grid_size=[1440, 1440, 41],
                out_size_factor=8,
                voxel_size=[0.075, 0.075],
                pc_range=[-54.0, -54.0],
                nms_type=None,
            ),
            common_heads=dict(
                center=[2, 2], height=[1, 2], dim=[3, 2], rot=[2, 2], vel=[2, 2]
            ),
            bbox_coder=dict(
                type="TransFusionBBoxCoder",
                pc_range=[-54.0, -54.0],
                post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                score_threshold=0.0,
                out_size_factor=8,
                voxel_size=[0.075, 0.075],
                code_size=10,
            ),
            loss_cls=dict(
                type="FocalLoss",
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                reduction="mean",
                loss_weight=1.0,
            ),
            loss_heatmap=dict(type="GaussianFocalLoss", reduction="mean", loss_weight=1.0),
            loss_bbox=dict(type="L1Loss", reduction="mean", loss_weight=0.25),
        ),
        map=dict(
            type="BEVSegmentationHead",
            in_channels=512,
            grid_transform=dict(
                input_scope=[[-54.0, 54.0, 0.6], [-54.0, 54.0, 0.6]],
                output_scope=[[-50, 50, 0.5], [-50, 50, 0.5]],
            ),
            classes=map_classes,
            loss="focal",
        ),
    ),
    loss_scale=dict(object=1.0, map=1.0),
)

# lr linear-scaled for single-GPU batch1: official det 2e-4 @ batch4 x 1/4 = 5e-5
optimizer = dict(
    type="AdamW",
    lr=5.0e-5,
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0),
        )
    ),
)
lr_config = dict(
    policy="CosineAnnealing",
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=0.33333333,
    min_lr_ratio=1.0e-3,
)
