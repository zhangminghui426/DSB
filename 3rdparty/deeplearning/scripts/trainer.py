"""Training engine: torchrun DDP + torch.amp (mmengine Runner replacement).

Unified training engine for det/seg/multitask/occ recipes:
- AdamW + optional paramwise_cfg (custom_keys decay_mult/lr_mult)
- grad clip (optimizer_config.grad_clip.max_norm)
- AMP via torch.amp.GradScaler (fp16.loss_scale: 'dynamic' or {growth_interval})
- LR: 'MultiStep' (by-epoch milestones*gamma), 'CosineAnnealing' or 'cyclic'
  (mmcv CyclicLrUpdaterHook semantics), all with optional iteration-based
  warmup (linear/constant/exp)
- MEGVII-style EMA (models/ema.py): updated every iteration, saved as
  epoch_N_ema.pth ({epoch, state_dict, updates}) — the official eval artifact
- checkpoint every checkpoint_config.interval epochs keeping max_keep_ckpts;
  resume support (incl. EMA state); eval hook from epoch val_begin
- dataset.set_epoch propagation (CBGS seed / ObjectPaste stop_epoch /
  GridMask schedule); eval AMP auto-off for occ-only models (fp32 protocol)
"""

import datetime
import json
import logging
import math
import os
import os.path as osp
import re
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- utilities

def init_dist(backend="nccl"):
    """ Initialize torchrun-style distributed training (env vars)."""
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend=backend, timeout=datetime.timedelta(seconds=5400))
    return rank, local_rank, world_size


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_distributed() else 0


def get_world_size():
    return dist.get_world_size() if is_distributed() else 1


def is_main_process():
    return get_rank() == 0


def to_device(data, device):
    """ Recursively move tensors / containers to device."""
    if isinstance(data, torch.Tensor):
        return data.to(device, non_blocking=True)
    if isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return type(data)(to_device(v, device) for v in data)
    return data


def annealing_cos(start, end, factor):
    """ mmcv annealing_cos: end + 0.5*(start-end)*(1+cos(pi*factor))."""
    cos_out = math.cos(math.pi * factor) + 1
    return end + 0.5 * start * cos_out - 0.5 * end * cos_out


def annealing_linear(start, end, factor):
    """ mmcv annealing_linear: start + (end-start)*factor."""
    return start + (end - start) * factor


def set_random_seed(seed, deterministic=False, rank=0):
    seed = seed + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np

    np.random.seed(seed % (2**32))
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def parse_losses(losses):
    """ {'loss_occ': t, ...} -> (total loss tensor, {name: float} log vars)."""
    log_vars = {}
    for name, value in losses.items():
        if isinstance(value, torch.Tensor):
            log_vars[name] = float(value.detach())
    loss = sum(v for k, v in losses.items() if k.startswith("loss") and isinstance(v, torch.Tensor))
    log_vars["loss"] = float(loss.detach())
    if is_distributed():
        reduced = loss.detach().clone()
        dist.all_reduce(reduced.div_(dist.get_world_size()))
        log_vars["loss"] = float(reduced)
    return loss, log_vars


# ---------------------------------------------------------------- LR schedules

class LRSchedule:
    """ Base: computes lr per param group; warmup blends over the regular lr."""

    def __init__(self, cfg, optimizer, iters_per_epoch, max_epochs):
        self.cfg = dict(cfg or {})
        self.optimizer = optimizer
        self.iters_per_epoch = iters_per_epoch
        self.max_epochs = max_epochs
        for group in optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])
        self.base_lrs = [g["initial_lr"] for g in optimizer.param_groups]
        self.max_iters = iters_per_epoch * max_epochs
        self.warmup = self.cfg.get("warmup", None)
        if self.warmup is not None:
            assert self.warmup in ("constant", "linear", "exp")
            self.warmup_iters = self.cfg.get("warmup_iters", 1000)
            self.warmup_ratio = self.cfg.get("warmup_ratio", 1.0 / 3)

    def regular_lr(self, epoch, cur_iter):
        raise NotImplementedError

    def get_lr(self, epoch, cur_iter):
        regular = self.regular_lr(epoch, cur_iter)
        if self.warmup is not None and cur_iter < self.warmup_iters:
            if self.warmup == "constant":
                return [lr * self.warmup_ratio for lr in regular]
            if self.warmup == "linear":
                # mmengine LinearLR: factor = start + (1-start)*i/end
                k = (1 - cur_iter / self.warmup_iters) * (1 - self.warmup_ratio)
                return [lr * (1 - k) for lr in regular]
            if self.warmup == "exp":
                k = self.warmup_ratio ** (1 - cur_iter / self.warmup_iters)
                return [lr * k for lr in regular]
        return regular

    def step(self, epoch, cur_iter):
        for group, lr in zip(self.optimizer.param_groups, self.get_lr(epoch, cur_iter)):
            group["lr"] = lr


class MultiStepLR(LRSchedule):
    """ mmengine MultiStepLR (by_epoch): lr = base * gamma^(milestones passed)."""

    def regular_lr(self, epoch, cur_iter):
        milestones = self.cfg.get("milestones", [])
        gamma = self.cfg.get("gamma", 0.1)
        passed = sum(1 for m in milestones if epoch >= m)
        return [base * (gamma ** passed) for base in self.base_lrs]


class CosineAnnealingLR(LRSchedule):
    """ Epoch-based cosine annealing to base*min_lr_ratio."""

    def regular_lr(self, epoch, cur_iter):
        min_lr_ratio = self.cfg.get("min_lr_ratio", None)
        min_lr = self.cfg.get("min_lr", None)
        out = []
        for base_lr in self.base_lrs:
            target = base_lr * min_lr_ratio if min_lr_ratio is not None else (min_lr or 0.0)
            out.append(annealing_cos(base_lr, target, epoch / self.max_epochs))
        return out


class CyclicLR(LRSchedule):
    """ mmcv CyclicLrUpdaterHook defaults (iter-based, cos annealing)."""

    def __init__(self, cfg, optimizer, iters_per_epoch, max_epochs):
        super().__init__(cfg, optimizer, iters_per_epoch, max_epochs)
        target_ratio = self.cfg.get("target_ratio", (10, 1e-4))
        if isinstance(target_ratio, (int, float)):
            target_ratio = (float(target_ratio), float(target_ratio) / 1e5)
        self.target_ratio = (float(target_ratio[0]), float(target_ratio[1]))
        self.cyclic_times = self.cfg.get("cyclic_times", 1)
        self.step_ratio_up = self.cfg.get("step_ratio_up", 0.4)
        self.anneal_strategy = self.cfg.get("anneal_strategy", "cos")
        self.anneal_func = annealing_cos if self.anneal_strategy == "cos" else annealing_linear
        max_iter_per_phase = self.max_iters // self.cyclic_times
        iter_up_phase = int(self.step_ratio_up * max_iter_per_phase)
        self.lr_phases = [
            [0, iter_up_phase, max_iter_per_phase, 1, self.target_ratio[0]],
            [iter_up_phase, max_iter_per_phase, max_iter_per_phase,
             self.target_ratio[0], self.target_ratio[1]],
        ]

    def regular_lr(self, epoch, cur_iter):
        out = []
        for base_lr in self.base_lrs:
            lr = base_lr
            for (start_iter, end_iter, phase_iters, start_ratio, end_ratio) in self.lr_phases:
                it = cur_iter % phase_iters
                if start_iter <= it < end_iter:
                    progress = it - start_iter
                    lr = self.anneal_func(
                        base_lr * start_ratio,
                        base_lr * end_ratio,
                        progress / (end_iter - start_iter),
                    )
                    break
            out.append(lr)
        return out


def build_lr_schedule(cfg, optimizer, iters_per_epoch, max_epochs):
    policy = (cfg or {}).get("policy", "MultiStep")
    if policy == "MultiStep":
        return MultiStepLR(cfg, optimizer, iters_per_epoch, max_epochs)
    if policy == "CosineAnnealing":
        return CosineAnnealingLR(cfg, optimizer, iters_per_epoch, max_epochs)
    if policy == "cyclic":
        return CyclicLR(cfg, optimizer, iters_per_epoch, max_epochs)
    raise ValueError(f"Unsupported lr_config policy: {policy}")


# ---------------------------------------------------------------- optimizer

def build_optimizer(model, cfg):
    """ AdamW with optional paramwise_cfg.custom_keys (decay_mult/lr_mult by
    name substring), mmcv DefaultOptimizerConstructor semantics (subset)."""
    cfg = dict(cfg)
    opt_type = cfg.pop("type", "AdamW")
    assert opt_type == "AdamW", f"only AdamW is ported, got {opt_type}"
    lr = cfg.pop("lr")
    weight_decay = cfg.pop("weight_decay", 0.01)
    paramwise = cfg.pop("paramwise_cfg", None) or {}
    custom_keys = paramwise.get("custom_keys", {})

    groups = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        key = "default"
        best_len = -1
        for ck in custom_keys:
            if ck in name and len(ck) > best_len:
                key, best_len = ck, len(ck)
        g = groups.setdefault(
            key,
            dict(
                params=[],
                lr_mult=custom_keys.get(key, {}).get("lr_mult", 1.0),
                decay_mult=custom_keys.get(key, {}).get("decay_mult", 1.0),
            ),
        )
        g["params"].append(param)

    param_groups = [
        dict(params=g["params"], lr=lr * g["lr_mult"], weight_decay=weight_decay * g["decay_mult"])
        for g in groups.values()
    ]
    return torch.optim.AdamW(param_groups, **cfg)


# ---------------------------------------------------------------- checkpointing

def save_checkpoint(model, optimizer, scaler, epoch, global_iter, cfg_text,
                    work_dir, filename, ema=None):
    raw = model.module if isinstance(model, DDP) else model
    state = dict(
        state_dict=raw.state_dict(),
        optimizer=optimizer.state_dict(),
        scaler=scaler.state_dict(),
        epoch=epoch,
        iter=global_iter,
        meta=dict(config=cfg_text, time=datetime.datetime.now().isoformat()),
    )
    if ema is not None:
        state["ema"] = ema.state_dict()
    path = osp.join(work_dir, filename)
    torch.save(state, path)
    return path


def auto_remove_old_ckpts(work_dir, max_keep):
    """ Keep the newest max_keep epoch_N.pth (plus matching epoch_N_ema.pth)."""
    if max_keep is None or max_keep < 0:
        return
    nums = []
    for f in os.listdir(work_dir):
        m = re.fullmatch(r"epoch_(\d+)\.pth", f)
        if m:
            nums.append(int(m.group(1)))
    nums.sort()
    while len(nums) > max_keep:
        n = nums.pop(0)
        for f in (f"epoch_{n}.pth", f"epoch_{n}_ema.pth"):
            p = osp.join(work_dir, f)
            if osp.exists(p):
                os.remove(p)


def load_model_checkpoint(model, path, map_location="cpu", strict=True):
    """ Accepts EMA releases ({epoch, state_dict, updates}), full training
    checkpoints, or bare state dicts."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    raw = model.module if isinstance(model, DDP) else model
    msg = raw.load_state_dict(state, strict=strict)
    return msg, ckpt


# ---------------------------------------------------------------- trainer

class Trainer:
    def __init__(self, model, cfg, train_loader, val_loader=None, val_dataset=None,
                 work_dir="outputs", load_from=None, resume_from=None,
                 device="cuda", debug_iters=None):
        self.cfg = cfg
        self.device = device
        self.work_dir = work_dir
        self.debug_iters = debug_iters  # if set, stop each epoch after N iters
        os.makedirs(work_dir, exist_ok=True)
        self.model = model.to(device)

        self.fp16_cfg = cfg.get("fp16", None)
        self.use_amp = self.fp16_cfg is not None
        scaler_kwargs = {}
        if self.use_amp and isinstance(self.fp16_cfg, dict):
            ls = self.fp16_cfg.get("loss_scale", "dynamic")
            if isinstance(ls, dict):
                scaler_kwargs["growth_interval"] = ls.get("growth_interval", 2000)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp, **scaler_kwargs)

        self.optimizer = build_optimizer(self.model, cfg.get("optimizer"))
        grad_clip = (cfg.get("optimizer_config") or {}).get("grad_clip", None)
        self.grad_clip_max_norm = grad_clip["max_norm"] if grad_clip else None

        self.max_epochs = cfg.get("max_epochs", 24)
        self.iters_per_epoch = len(train_loader)
        self.lr_schedule = build_lr_schedule(
            cfg.get("lr_config"), self.optimizer, self.iters_per_epoch, self.max_epochs
        )

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.val_dataset = val_dataset
        self.eval_interval = (cfg.get("evaluation") or {}).get("interval", 1)
        self.val_begin = cfg.get("val_begin", 1)

        ckpt_cfg = cfg.get("checkpoint_config") or {}
        self.ckpt_interval = ckpt_cfg.get("interval", 1)
        self.max_keep_ckpts = ckpt_cfg.get("max_keep_ckpts", -1)

        log_cfg = cfg.get("log_config") or {}
        self.log_interval = log_cfg.get("interval", 50)

        # training logs: tensorboard (tf_logs/) + JSON-lines scalars, main process only
        self.tb_writer = None
        self.scalar_path = None
        if is_main_process():
            self.scalar_path = osp.join(work_dir, "scalars.jsonl")
            if log_cfg.get("tensorboard", True):
                try:
                    from torch.utils.tensorboard import SummaryWriter

                    self.tb_writer = SummaryWriter(osp.join(work_dir, "tf_logs"))
                except ImportError:
                    logger.warning("tensorboard not installed; scalars.jsonl only")

        self.start_epoch = 0
        self.global_iter = 0

        if load_from:
            msg, _ = load_model_checkpoint(self.model, load_from)
            if is_main_process():
                logger.info(
                    "loaded %s (missing=%d unexpected=%d)",
                    load_from, len(msg.missing_keys), len(msg.unexpected_keys),
                )
        # MEGVII-style EMA on the unwrapped model (official eval artifact);
        # created BEFORE resume so resume can restore the EMA state, and AFTER
        # load_from so the EMA copy starts from the loaded weights
        self.ema = None
        ema_cfg = cfg.get("ema", None)
        if ema_cfg:
            from models.ema import ModelEMA

            self.ema = ModelEMA(
                self.model,
                decay=ema_cfg.get("decay", 0.9990),
                tau=ema_cfg.get("tau", 2000),
                updates=ema_cfg.get("init_updates", 0),
            )

        if resume_from:
            self.resume(resume_from)

        self.raw_model = self.model
        if is_distributed():
            self.model = DDP(
                self.model,
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=cfg.get("find_unused_parameters", False),
            )

    def resume(self, path):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        raw = self.model.module if isinstance(self.model, DDP) else self.model
        raw.load_state_dict(ckpt["state_dict"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler"])
        self.start_epoch = ckpt.get("epoch", 0) + 1
        self.global_iter = ckpt.get("iter", 0)
        if self.ema is not None and "ema" in ckpt:
            self.ema.load_state_dict(ckpt["ema"])
        if is_main_process():
            logger.info("resumed from %s @ epoch %d", path, self.start_epoch)

    def log_step(self, epoch, inner_iter, log_vars, lr_now, t_data, t_iter):
        mem = torch.cuda.max_memory_allocated() // (1024**2)
        msg = (
            f"Epoch [{epoch + 1}/{self.max_epochs}][{inner_iter + 1}/{self.iters_per_epoch}]  "
            f"lr: {lr_now:.3e}, memory: {mem}M, "
            f"data_time: {t_data:.3f}, time: {t_iter:.3f}, "
        )
        msg += ", ".join(f"{k}: {v:.4f}" for k, v in log_vars.items())
        logger.info(msg)
        if self.tb_writer is not None:
            self.tb_writer.add_scalar("train/lr", lr_now, self.global_iter)
            for k, v in log_vars.items():
                self.tb_writer.add_scalar(f"train/{k}", v, self.global_iter)
        if self.scalar_path is not None:
            record = dict(mode="train", epoch=epoch + 1, iter=inner_iter + 1,
                          global_iter=self.global_iter, lr=lr_now, memory=int(mem),
                          data_time=round(t_data, 4), time=round(t_iter, 4))
            record.update({k: round(v, 6) for k, v in log_vars.items()})
            with open(self.scalar_path, "a") as f:
                f.write(json.dumps(record) + "\n")

    def train(self):
        if is_main_process():
            logger.info(
                "start_epoch=%d max_epochs=%d amp=%s ema=%s",
                self.start_epoch, self.max_epochs, self.use_amp, self.ema is not None,
            )
        for epoch in range(self.start_epoch, self.max_epochs):
            self.train_one_epoch(epoch)
            if is_main_process() and (epoch + 1) % self.ckpt_interval == 0:
                path = save_checkpoint(
                    self.model, self.optimizer, self.scaler, epoch, self.global_iter,
                    json.dumps({}, default=str)[:2000], self.work_dir,
                    f"epoch_{epoch + 1}.pth", ema=self.ema,
                )
                logger.info("saved %s", path)
                if self.ema is not None:
                    logger.info("saved %s", self.ema.save_checkpoint(self.work_dir, epoch))
                if self.max_keep_ckpts and self.max_keep_ckpts > 0:
                    auto_remove_old_ckpts(self.work_dir, self.max_keep_ckpts)
            if (
                self.val_loader is not None
                and self.val_dataset is not None
                and (epoch + 1) >= self.val_begin
                and (epoch + 1) % self.eval_interval == 0
            ):
                self.evaluate(epoch)
        if is_main_process() and self.tb_writer is not None:
            self.tb_writer.close()

    def train_one_epoch(self, epoch):
        self.model.train()
        sampler = self.train_loader.sampler
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        ds = self.train_loader.dataset
        if hasattr(ds, "set_epoch"):
            ds.set_epoch(epoch)
        inner = getattr(ds, "dataset", None)  # CBGS wrapper -> wrapped dataset
        if inner is not None and hasattr(inner, "set_epoch"):
            inner.set_epoch(epoch)

        prev_end = time.time()
        for i, data in enumerate(self.train_loader):
            t_load_start = time.time()
            t_data = t_load_start - prev_end

            data = to_device(data, self.device)
            self.lr_schedule.step(epoch, self.global_iter)
            lr_now = self.optimizer.param_groups[0]["lr"]

            self.optimizer.zero_grad(set_to_none=True)
            # cache_enabled=False: the autocast weight-cast cache makes
            # gradient-checkpointing recomputation diverge from the forward
            # dispatch (observed CheckpointError with Swin-B + AMP)
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.use_amp,
                                cache_enabled=False):
                losses = self.model(**data)
            loss, log_vars = parse_losses(losses)

            self.scaler.scale(loss).backward()
            if self.grad_clip_max_norm is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for g in self.optimizer.param_groups for p in g["params"]],
                    self.grad_clip_max_norm,
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.ema is not None:
                self.ema.update(self.raw_model)

            self.global_iter += 1
            t_iter = time.time() - t_load_start
            if is_main_process() and (i + 1) % self.log_interval == 0:
                self.log_step(epoch, i, log_vars, lr_now, t_data, t_iter)
            prev_end = time.time()
            if self.debug_iters is not None and (i + 1) >= self.debug_iters:
                if is_main_process():
                    logger.info("debug_iters=%d reached, stopping epoch", self.debug_iters)
                break

    @torch.no_grad()
    def evaluate(self, epoch):
        from scripts.tester import run_inference

        eval_cfg = self.cfg.get("evaluation") or {}
        use_amp = eval_cfg.get("use_amp", None)
        if use_amp is None:
            raw = self.model.module if isinstance(self.model, DDP) else self.model
            occ_only = (hasattr(raw, "encoders") and "occ" in raw.encoders
                        and raw.decoder is None)
            use_amp = not occ_only
        outputs = run_inference(self.model, self.val_loader, self.device,
                                use_amp=use_amp)
        if is_main_process():
            logger.info("Epoch %d evaluation:", epoch + 1)
            metrics = self.val_dataset.evaluate(outputs, verbose=False)
            for k, v in metrics.items():
                logger.info("  %s: %s", k, v)
            if self.tb_writer is not None:
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        self.tb_writer.add_scalar(f"val/{k}", v, epoch + 1)
            if self.scalar_path is not None:
                record = dict(mode="val", epoch=epoch + 1)
                record.update({k: v for k, v in metrics.items()})
                with open(self.scalar_path, "a") as f:
                    f.write(json.dumps(record) + "\n")
        if is_distributed():
            dist.barrier()
        self.model.train()
