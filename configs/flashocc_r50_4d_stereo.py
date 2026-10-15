"""Occupancy 4D-Stereo (r50, 256x704, BEVStereo4DOCC port).

Temporal (1 adjacent frame) + stereo cost-volume depth, SID depth bins,
numC_Trans=80, depth-supervised (loss_depth_weight 0.05). Official full-val
mIoU 37.84 (with det pretrain; no local checkpoint — verify by forward/backward
only). scripts/test.py auto-enables align_after_view_transfromation at eval
(official protocol; --no_aavt to disable).

Single-GPU (8GB) recipe: batch 1 + with_cp + AMP / 24 epochs / AdamW 1e-4.

Usage:
    python scripts/train.py configs/flashocc_r50_4d_stereo.py
"""

from configs.common import *  # noqa: F401,F403

max_epochs = 24
val_begin = 20
optimizer = dict(type="AdamW", lr=1.0e-4, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(policy="MultiStep", milestones=[24], gamma=0.1,
                 warmup="linear", warmup_iters=200, warmup_ratio=0.001)
ema = dict(decay=0.999, tau=2000, init_updates=10560)

numC_Trans = 80
multi_adj_frame_id_cfg = (1, 2, 1)
num_adj = len(range(*multi_adj_frame_id_cfg))

# sequential pipelines: adjacent frames share augmentation, cam-major stacking
train_pipeline = make_occ_train_pipeline(occ_grid_config, sequential=True)
test_pipeline = make_occ_test_pipeline(sequential=True)
data = make_occ_data_config(train_pipeline, test_pipeline,
                            samples_per_gpu=1, workers_per_gpu=4,
                            stereo=True,
                            multi_adj_frame_id_cfg=multi_adj_frame_id_cfg)

model = make_occ_model(numC_Trans=numC_Trans, head_out_dim=256, stereo=True,
                       num_adj=num_adj)
