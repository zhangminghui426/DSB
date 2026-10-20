"""Occupancy M1 (r50, 256x704, 18 classes) — matches the official checkpoint
ckpts/flashocc-r50-256x704.pth (remapped: -bff.pth; mini val mIoU 24.09,
official full-val 32.08). Occ head only (heads.occ).

Single-GPU (8GB) recipe: batch 1 + ResNet with_cp + AMP / 24 epochs /
AdamW 1e-4 + clip 5 / LinearLR warmup(200 it) -> MultiStep([24]) + MEGVII EMA.
Official multi-GPU recipe: 8 GPUs x batch 4 (auto_scale_lr base 32).

Usage:
    python scripts/train.py configs/flashocc_r50.py
    python scripts/test.py configs/flashocc_r50.py ckpts/flashocc-r50-256x704-bff.pth
"""

from configs.common import *  # noqa: F401,F403

# official occ recipe runtime (explicit per leaf)
max_epochs = 24
val_begin = 20  # in-training eval only from epoch 20
optimizer = dict(type="AdamW", lr=1.0e-4, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(policy="MultiStep", milestones=[24], gamma=0.1,
                 warmup="linear", warmup_iters=200, warmup_ratio=0.001)
ema = dict(decay=0.999, tau=2000, init_updates=10560)

train_pipeline = make_occ_train_pipeline(occ_grid_config)
test_pipeline = make_occ_test_pipeline()
data = make_occ_data_config(train_pipeline, test_pipeline,
                            samples_per_gpu=1, workers_per_gpu=4)

model = make_occ_model(numC_Trans=64, head_out_dim=256)
