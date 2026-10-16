"""SECOND 训练脚本。

用法:
  python train.py
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from configs.default import get_config
from datasets.dataset import SECONDDataset, collate_batch
from models.second import SECOND


def build_loss(
    cls_pred: torch.Tensor,
    loc_pred: torch.Tensor,
    cls_target: torch.Tensor,
    loc_target: torch.Tensor,
    cfg,
) -> tuple[torch.Tensor, dict]:
    """计算分类和回归损失。

    Args:
        cls_pred:    (B, num_rot * num_class, Dy, Dx)。
        loc_pred:    (B, num_rot * 7, Dy, Dx)。
        cls_target:  (B, Dy, Dx, num_rot) 0=neg, 1=pos, -1=ignore。
        loc_target: (B, Dy, Dx, num_rot, 7)。
    """
    B, _, Dy, Dx = cls_pred.shape
    num_rot = cfg.model.num_anchor_per_loc

    # reshape cls_pred: (B, num_rot, num_class, Dy, Dx) -> (B, Dy, Dx, num_rot)
    cls_pred = cls_pred.view(B, num_rot, cfg.model.num_class, Dy, Dx)
    cls_pred = cls_pred.permute(0, 3, 4, 1, 2).contiguous()  # (B, Dy, Dx, num_rot, num_class)
    cls_pred = cls_pred.squeeze(-1)  # (B, Dy, Dx, num_rot)

    # 有效样本 (非忽略)
    valid_mask = cls_target >= 0  # (B, Dy, Dx, num_rot)
    pos_mask = cls_target == 1

    # 分类损失 (BCE with logits)
    cls_loss = nn.functional.binary_cross_entropy_with_logits(
        cls_pred[valid_mask], cls_target[valid_mask], reduction="mean"
    )

    # 回归损失 (Smooth L1, 只对正样本)
    loc_pred = loc_pred.view(B, num_rot, 7, Dy, Dx)
    loc_pred = loc_pred.permute(0, 3, 4, 1, 2).contiguous()  # (B, Dy, Dx, num_rot, 7)

    if pos_mask.sum() > 0:
        loc_loss = nn.functional.smooth_l1_loss(
            loc_pred[pos_mask], loc_target[pos_mask], reduction="mean"
        )
    else:
        loc_loss = torch.tensor(0.0, device=cls_pred.device)

    total_loss = cfg.train.cls_weight * cls_loss + cfg.train.loc_weight * loc_loss
    return total_loss, {
        "cls_loss": cls_loss.item(),
        "loc_loss": loc_loss.item() if isinstance(loc_loss, torch.Tensor) else loc_loss,
        "total_loss": total_loss.item(),
        "pos_ratio": pos_mask.float().mean().item(),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    cfg,
    device: torch.device,
    epoch: int,
) -> dict:
    model.train()
    total_metrics = {"cls_loss": 0.0, "loc_loss": 0.0, "total_loss": 0.0}
    n_batches = 0

    for batch_idx, batch in enumerate(loader):
        vf = batch["voxel_features"].to(device)
        vc = batch["voxel_coords"].to(device)
        np_pts = batch["num_points"].to(device)
        cls_t = batch["cls_target"].to(device)
        loc_t = batch["loc_target"].to(device)

        cls_pred, loc_pred = model(
            vf, vc, np_pts,
            cfg.voxel.voxel_size,
            cfg.voxel.point_cloud_range,
        )

        loss, metrics = build_loss(cls_pred, loc_pred, cls_t, loc_t, cfg)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for k in total_metrics:
            total_metrics[k] += metrics[k]
        n_batches += 1

        if batch_idx % cfg.train.log_interval == 0:
            print(
                f"  Epoch {epoch} [{batch_idx}/{len(loader)}] "
                f"cls={metrics['cls_loss']:.4f} loc={metrics['loc_loss']:.4f} "
                f"total={metrics['total_loss']:.4f} pos_ratio={metrics['pos_ratio']:.4f}"
            )

    return {k: v / max(n_batches, 1) for k, v in total_metrics.items()}


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, cfg, device: torch.device) -> dict:
    model.eval()
    total_metrics = {"cls_loss": 0.0, "loc_loss": 0.0, "total_loss": 0.0}
    n_batches = 0
    for batch in loader:
        vf = batch["voxel_features"].to(device)
        vc = batch["voxel_coords"].to(device)
        np_pts = batch["num_points"].to(device)
        cls_t = batch["cls_target"].to(device)
        loc_t = batch["loc_target"].to(device)
        cls_pred, loc_pred = model(vf, vc, np_pts, cfg.voxel.voxel_size, cfg.voxel.point_cloud_range)
        loss, metrics = build_loss(cls_pred, loc_pred, cls_t, loc_t, cfg)
        for k in total_metrics:
            total_metrics[k] += metrics[k]
        n_batches += 1
    return {k: v / max(n_batches, 1) for k, v in total_metrics.items()}


def main() -> None:
    cfg = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 数据
    train_set = SECONDDataset(cfg, split="train")
    val_set = SECONDDataset(cfg, split="val")
    train_loader = DataLoader(
        train_set, batch_size=cfg.train.batch_size, shuffle=True,
        collate_fn=collate_batch, num_workers=0,
    )
    val_loader = DataLoader(
        val_set, batch_size=cfg.train.batch_size, shuffle=False,
        collate_fn=collate_batch, num_workers=0,
    )
    print(f"训练集: {len(train_set)} 帧, 验证集: {len(val_set)} 帧")

    # 模型
    model = SECOND(
        in_channels=4,
        vfe_filters=cfg.model.vfe_num_filters,
        sparse_filters=cfg.model.sparse_num_filters,
        upsample_filters=cfg.model.rpn_upsample_filters,
        grid_size=cfg.voxel.grid_size,
        num_anchor_per_loc=cfg.model.num_anchor_per_loc,
        num_class=cfg.model.num_class,
    ).to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )

    output_dir = Path(cfg.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    for epoch in range(1, cfg.train.num_epochs + 1):
        print(f"\n=== Epoch {epoch}/{cfg.train.num_epochs} ===")
        train_metrics = train_one_epoch(model, train_loader, optimizer, cfg, device, epoch)
        val_metrics = validate(model, val_loader, cfg, device)
        print(
            f"  Train: total={train_metrics['total_loss']:.4f} "
            f"cls={train_metrics['cls_loss']:.4f} loc={train_metrics['loc_loss']:.4f}"
        )
        print(
            f"  Val:   total={val_metrics['total_loss']:.4f} "
            f"cls={val_metrics['cls_loss']:.4f} loc={val_metrics['loc_loss']:.4f}"
        )

        if val_metrics["total_loss"] < best_val:
            best_val = val_metrics["total_loss"]
            torch.save(model.state_dict(), output_dir / "best_model.pth")
            print(f"  保存最优模型 (val_total={best_val:.4f})")

    torch.save(model.state_dict(), output_dir / "last_model.pth")
    print(f"\n训练完成, 最优 val_loss={best_val:.4f}")


if __name__ == "__main__":
    main()
