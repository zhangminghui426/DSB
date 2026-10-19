"""Occupancy stbase 4D-Stereo (Swin-B, 512x1408) — extension variant.

Swin-B backbone (embed 128, depths [2,2,18,2], window 12) with stage-0
output as the stereo reference feature; FPN_LSS image neck (1536->512, no
extra upsample); official recipe lr 2e-4 (8 GPUs x 4). Official full-val
mIoU 43.52 — requires the bevdet-stbase-4d-stereo-512x1408-cbgs pretrain
(not present locally; no metric golden for this variant).

Full-size 512x1408 TRAINING does not fit an 8GB GPU (official: multi-GPU);
eval forward fits with AMP. Verified by parameter count + full-size eval
forward + reduced-size (256x704) train forward/backward smoke.

Usage:
    python scripts/train.py configs/flashocc_stbase_4d_stereo_512x1408.py
"""

from configs.common import *  # noqa: F401,F403

max_epochs = 24
val_begin = 20
optimizer = dict(type="AdamW", lr=2.0e-4, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(policy="MultiStep", milestones=[24], gamma=0.1,
                 warmup="linear", warmup_iters=200, warmup_ratio=0.001)
ema = dict(decay=0.999, tau=2000, init_updates=10560)

numC_Trans = 80
multi_adj_frame_id_cfg = (1, 2, 1)
num_adj = len(range(*multi_adj_frame_id_cfg))
data_config_stbase = dict(occ_data_config, input_size=(512, 1408))

train_pipeline = make_occ_train_pipeline(occ_grid_config, sequential=True,
                                         data_config=data_config_stbase)
test_pipeline = make_occ_test_pipeline(sequential=True,
                                       data_config=data_config_stbase)
data = make_occ_data_config(train_pipeline, test_pipeline,
                            samples_per_gpu=1, workers_per_gpu=4,
                            stereo=True,
                            multi_adj_frame_id_cfg=multi_adj_frame_id_cfg)

model = make_occ_model(
    numC_Trans=numC_Trans,
    head_out_dim=256,
    stereo=True,
    num_adj=num_adj,
    input_size=(512, 1408),
    backbone_cfg=dict(
        type="SwinTransformer",
        pretrain_img_size=224,
        patch_size=4,
        window_size=12,
        mlp_ratio=4,
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        strides=(4, 2, 2, 2),
        out_indices=[0, 2, 3],  # stage0 = stereo reference feature
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        with_cp=True,
    ),
    neck_cfg=dict(type="FPN_LSS", in_channels=512 + 1024, out_channels=512,
                  extra_upsample=None, input_feature_index=(0, 1),
                  scale_factor=2),
    vt_in_channels=512,
)
