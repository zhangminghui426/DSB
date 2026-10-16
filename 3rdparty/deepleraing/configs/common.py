"""Common config defaults shared by all leaf configs.

Leaf configs import this and override what they need. No ${} interpolation
or directory-inheritance — plain Python imports.
"""

# ---- runtime ----
seed = 0
deterministic = False
cudnn_benchmark = True
max_epochs = 24
fp16 = dict(loss_scale=512.0)

# ---- optimizer ----
optimizer = dict(
    type="AdamW",
    lr=2e-4,
    weight_decay=1e-2,
)

# ---- scheduler ----
lr_schedule = dict(
    type="CosineAnnealingLR",
    T_max=max_epochs,
    eta_min=1e-6,
)

# ---- logging / ckpt ----
log_interval = 50
ckpt_interval = 1
work_dir = "outputs/default"

# ---- EMA ----
ema = dict(
    enabled=True,
    decay=0.999,
    update_after_step=100,
    update_every=1,
)
