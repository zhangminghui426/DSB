"""nuScenes BEV segmentation: camera(Swin-T + LSS) + lidar(SparseEncoder)
-> ConvFuser -> SECOND/SECONDFPN decoder -> BEVSegmentationHead (6 classes).
Single-GPU (8GB) recipe: batch 2 + Swin gradient checkpointing (with_cp) +
AMP / 20 epochs / AdamW 5e-5 (+paramwise no-decay) / cyclic LR. ±51.2m, voxel 0.1.
Multi-GPU official recipe: --batch_size 4 --options optimizer.lr=1e-4
model.encoders.camera.backbone.with_cp=False.

Usage:
    python scripts/train.py configs/seg_fusion_bev256d2_lss.py --load_from ckpts/bevfusion-seg-bf.pth
    python scripts/test.py configs/seg_fusion_bev256d2_lss.py ckpts/bevfusion-seg-bf.pth --eval map
"""

from configs.common import *  # noqa: F401,F403

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.1, 0.1, 0.2]
max_epochs = 20
checkpoint_config = dict(interval=1, max_keep_ckpts=1)
batch_size = 2  # single-GPU (8GB) default, measured with with_cp; override: --batch_size N

train_pipeline = make_train_pipeline(
    point_cloud_range, sweeps_num=9, max_epochs=max_epochs, with_bev_seg=True
)
test_pipeline = make_test_pipeline(point_cloud_range, with_bev_seg=True)
data = make_data_config(train_pipeline, test_pipeline, samples_per_gpu=batch_size)

model = dict(
    type="BevPerception",
    encoders=dict(
        camera=dict(
            # with_cp (gradient checkpointing): required for batch>=2 on 8GB
            backbone=swin_backbone(drop_path_rate=0.3, with_cp=True),
            neck=CAMERA_NECK,
            vtransform=dict(
                type="LSSTransform",
                in_channels=256,
                out_channels=80,
                image_size=image_size,
                feature_size=[32, 88],
                xbound=[-51.2, 51.2, 0.4],
                ybound=[-51.2, 51.2, 0.4],
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
                max_voxels=[90000, 120000],
            ),
            backbone=sparse_encoder([1024, 1024, 41]),
        ),
    ),
    fuser=FUSER,
    decoder=DECODER,
    heads=dict(
        occ=None,
        object=None,
        map=dict(
            type="BEVSegmentationHead",
            in_channels=512,
            grid_transform=dict(
                input_scope=[[-51.2, 51.2, 0.8], [-51.2, 51.2, 0.8]],
                output_scope=[[-50, 50, 0.5], [-50, 50, 0.5]],
            ),
            classes=map_classes,
            loss="focal",
        ),
    ),
)

# lr linear-scaled for single-GPU batch2: official 1e-4 @ batch4 x 2/4 = 5e-5
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
lr_config = dict(policy="cyclic")
