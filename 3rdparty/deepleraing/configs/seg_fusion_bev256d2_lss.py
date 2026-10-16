"""Segmentation fusion config (BEV 256x256, LSS).

Camera + LiDAR BEV fusion for map segmentation.
"""

from configs.common import *  # noqa: F401,F403

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.1, 0.1, 0.2]
max_epochs = 24
batch_size = 2

model = dict(
    type="BevPerception",
    encoders=dict(
        camera=dict(
            backbone=dict(type="ResNet", depth=50, out_indices=(2, 3)),
            neck=dict(type="GeneralizedLSSFPN", in_channels=(512, 1024, 2048), out_channels=256),
            vtransform=dict(
                type="LSSTransform",
                in_channels=256,
                out_channels=80,
                grid_config=dict(
                    x=[-51.2, 51.2, 0.8],
                    y=[-51.2, 51.2, 0.8],
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
        map=dict(
            type="BEVSegmentationHead",
            in_channels=512,
            num_classes=6,
        ),
    ),
)

optimizer = dict(type="AdamW", lr=2e-4, weight_decay=0.01)
