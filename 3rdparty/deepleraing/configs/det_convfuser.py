"""nuScenes detection: camera(Swin-T + DepthLSS) + lidar(SparseEncoder)
-> ConvFuser -> SECOND/SECONDFPN decoder -> TransFusionHead (heads.object).
Single-GPU (8GB) recipe: batch 2 + Swin gradient checkpointing (with_cp) +
AMP / 6 epochs / AdamW 1e-4 cosine. ±54m, voxel 0.075.
Multi-GPU official recipe: --batch_size 4 --options optimizer.lr=2e-4
(with_cp=False).

Usage:
    python scripts/train.py configs/det_convfuser.py --load_from ckpts/bevfusion-det-bf.pth
    python scripts/test.py configs/det_convfuser.py ckpts/bevfusion-det-bf.pth --eval bbox
"""

from configs.common import *  # noqa: F401,F403

point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
voxel_size = [0.075, 0.075, 0.2]
max_epochs = 6
batch_size = 2  # single-GPU (8GB) default; override: --batch_size N (official: 8 GPUs x 4)
checkpoint_config = dict(interval=1, max_keep_ckpts=1)

train_pipeline = make_train_pipeline(
    point_cloud_range, sweeps_num=0, max_epochs=max_epochs
)
test_pipeline = make_test_pipeline(point_cloud_range)
data = make_data_config(train_pipeline, test_pipeline, samples_per_gpu=batch_size)

model = dict(
    type="BevPerception",
    encoders=dict(
        camera=dict(
            # with_cp (gradient checkpointing): required for batch>=2 on 8GB
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
        map=None,
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
    ),
)

# lr linear-scaled for single-GPU batch2: official 2e-4 @ batch4 x 2/4 = 1e-4
optimizer = dict(type="AdamW", lr=1.0e-4, weight_decay=0.01)
lr_config = dict(
    policy="CosineAnnealing",
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=0.33333333,
    min_lr_ratio=1.0e-3,
)
