"""FlashOCC-Swin-B 4D stereo 512x1408 config.

High-resolution stereo occupancy with Swin-B backbone and temporal aggregation.
"""

from configs.common import *  # noqa: F401,F403

point_cloud_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
voxel_size = [0.4, 0.4, 0.4]
max_epochs = 24
batch_size = 1
sequential = True
num_frame = 3
image_size = (512, 1408)

model = dict(
    type="BevPerception",
    encoders=dict(
        camera=dict(
            backbone=dict(
                type="SwinTransformer",
                embed_dims=128,
                depths=[2, 2, 18, 2],
                num_heads=[4, 8, 16, 32],
                out_indices=(0, 2, 3),
            ),
            neck=dict(type="CustomFPN", in_channels=(128, 512, 1024), out_channels=256),
            vtransform=dict(
                type="OccLSSDepth",
                in_channels=256,
                out_channels=80,
                image_size=image_size,
                grid_config=dict(
                    x=[-40.0, 40.0, 0.4],
                    y=[-40.0, 40.0, 0.4],
                    z=[-1.0, 5.4, 0.4],
                    depth=[1.0, 60.0, 0.5],
                ),
            ),
        ),
    ),
    heads=dict(
        occ=dict(
            type="BEVOCCHead2D",
            in_channels=256,
            out_channels=256,
            num_classes=18,
            num_frame=num_frame,
        ),
    ),
)

optimizer = dict(type="AdamW", lr=1e-4, weight_decay=0.01)
