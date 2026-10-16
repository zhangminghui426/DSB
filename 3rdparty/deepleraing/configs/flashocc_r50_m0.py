"""FlashOCC-R50-M0 config (medium resolution).

Occupancy prediction with medium resolution BEV grid.
"""

from configs.common import *  # noqa: F401,F403

point_cloud_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
voxel_size = [0.8, 0.8, 0.8]
max_epochs = 24
batch_size = 4

model = dict(
    type="BevPerception",
    encoders=dict(
        camera=dict(
            backbone=dict(type="ResNet", depth=50, out_indices=(2, 3)),
            neck=dict(type="CustomFPN", in_channels=(512, 1024, 2048), out_channels=256),
            vtransform=dict(
                type="OccLSS",
                in_channels=256,
                out_channels=80,
                grid_config=dict(
                    x=[-40.0, 40.0, 0.8],
                    y=[-40.0, 40.0, 0.8],
                    z=[-1.0, 5.4, 0.8],
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
        ),
    ),
)

optimizer = dict(type="AdamW", lr=2e-4, weight_decay=0.01)
