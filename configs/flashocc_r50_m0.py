"""Occupancy M0 (r50, 256x704) — lighter head (out_dim 128) and coarser depth
bins (1.0 m). Same training recipe as M1. No official checkpoint on disk
(download: see FlashOCC README model zoo).

Usage:
    python scripts/train.py configs/flashocc_r50_m0.py
"""

from configs.common import *  # noqa: F401,F403

max_epochs = 24
val_begin = 20
optimizer = dict(type="AdamW", lr=1.0e-4, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(policy="MultiStep", milestones=[24], gamma=0.1,
                 warmup="linear", warmup_iters=200, warmup_ratio=0.001)
ema = dict(decay=0.999, tau=2000, init_updates=10560)

grid_config = dict(occ_grid_config, depth=[1.0, 45.0, 1.0])  # noqa: F405

train_pipeline = make_occ_train_pipeline(grid_config)
test_pipeline = make_occ_test_pipeline()
data = make_occ_data_config(train_pipeline, test_pipeline,
                            samples_per_gpu=1, workers_per_gpu=4)

model = make_occ_model(numC_Trans=64, head_out_dim=128, grid_config=grid_config)
# sanity: M0 head output = 128 * 16 = 2048 channels on predicter.2
assert model["heads"]["occ"]["out_dim"] == 128
