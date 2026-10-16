"""Multitask fusion config (detection + segmentation).

Camera + LiDAR fusion for both detection and BEV map segmentation.
"""

from configs.common import *  # noqa: F401,F403

point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
voxel_size = [0.075, 0.075, 0.2]
max_epochs = 24
batch_size = 2

model = dict(
    type="BevPerception",
    encoders=dict(
        camera=dict(
            backbone=dict(type="SwinTransformer", embed_dims=96, depths=[2,2,6,2], num_heads=[3,6,12,24]),
            neck=dict(type="GeneralizedLSSFPN", out_channels=256),
            vtransform=dict(
                type="DepthLSSTransform",
                in_channels=256,
                out_channels=80,
                grid_config=dict(
                    x=[-54.0, 54.0, 0.3],
                    y=[-54.0, 54.0, 0.3],
                    z=[-10.0, 10.0, 20.0],
                    depth=[1.0, 60.0, 0.5],
                ),
            ),
        ),
        lidar=dict(
            voxelize=dict(
                max_num_points=10,
                point_cloud_range=point_cloud_range,
                voxel_size=voxel_size,
                max_voxels=[120000, 160000],
            ),
            backbone=dict(type="SparseEncoder", in_channels=4, output_channels=256),
        ),
    ),
    fuser=dict(type="ConvFuser", in_channels=[80, 256], out_channels=256),
    decoder=dict(type="SECOND", in_channels=256, out_channels=512),
    heads=dict(
        object=dict(
            type="TransFusionHead",
            num_proposals=200,
            in_channels=512,
            hidden_channel=128,
            num_classes=10,
        ),
        map=dict(
            type="BEVSegmentationHead",
            in_channels=512,
            num_classes=6,
        ),
    ),
)

optimizer = dict(type="AdamW", lr=2e-4, weight_decay=0.01)
