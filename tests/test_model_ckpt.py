"""Gate C: unified BevPerception integration + official checkpoint loading.

Run from repo root:
    python tests/test_model_ckpt.py

Checks:
1. det model (camera+lidar convfuser, heads.object) loads
   ckpts/bevfusion-det-bf.pth STRICTLY (zero missing / unexpected).
2. Fake-batch forward (train mode, fp16 autocast) -> finite losses -> backward.
3. Eval-mode forward -> get_bboxes sane outputs.
4. Same for the seg model (heads.map + bevfusion-seg-bf.pth).
5. occ model (heads.occ) loads the remapped official FlashOCC EMA checkpoint
   ckpts/flashocc-r50-256x704-bff.pth STRICTLY (430 keys); M0 shape assertions;
   fake occ-batch train forward/backward + eval pred_occ shape.
"""

import sys

import os.path as osp; sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), ".."))

import math

import torch

from datasets.geometry.boxes import LiDARInstance3DBoxes
from models import build_model
from scripts.config import load_config

DET_CFG = "configs/det_convfuser.py"
SEG_CFG = "configs/seg_fusion_bev256d2_lss.py"
OCC_CFG = "configs/flashocc_r50.py"
OCC_CKPT = "ckpts/flashocc-r50-256x704-bff.pth"
DEV = "cuda:0"


def total_loss(losses):
    """Trainer-style reduction (scripts/trainer.py parse_losses equivalent)."""
    return sum(v for k, v in losses.items() if k.startswith("loss"))


def rigid(yaw, t):
    m = torch.eye(4)
    c, s = math.cos(yaw), math.sin(yaw)
    m[0, 0], m[0, 1] = c, -s
    m[1, 0], m[1, 1] = s, c
    m[:3, 3] = torch.tensor(t)
    return m


def make_fake_batch(B=1, N=30000, with_seg_gt=False):
    torch.manual_seed(0)
    img = torch.randn(B, 6, 3, 256, 704, device=DEV)
    pts = torch.rand(N, 5, device=DEV) * torch.tensor([100, 100, 7, 1, 1], device=DEV) + torch.tensor(
        [-50, -50, -4.5, 0, 0], device=DEV
    )
    points = [pts for _ in range(B)]

    # 6-camera ring extrinsics (camera->lidar), nuScenes-like mounting
    cam_yaws = [0.0, -0.96, 0.96, math.pi, 2.18, -2.18]
    camera2lidar = torch.stack(
        [
            rigid(yaw, [1.7 * math.cos(yaw), 1.7 * math.sin(yaw), 1.5])
            for yaw in cam_yaws
        ]
    )[None].repeat(B, 1, 1, 1).to(DEV)

    K = torch.eye(4)
    K[0, 0] = K[1, 1] = 280.0  # focal at 256x704 scale
    K[0, 2], K[1, 2] = 352.0, 128.0
    camera_intrinsics = K[None, None].repeat(B, 6, 1, 1).to(DEV)

    lidar2camera = torch.inverse(camera2lidar)
    lidar2image = torch.matmul(camera_intrinsics, lidar2camera)
    img_aug = torch.eye(4)[None, None].repeat(B, 6, 1, 1).to(DEV)
    lidar2ego = torch.eye(4)[None].repeat(B, 1, 1).to(DEV)
    camera2ego = torch.matmul(lidar2ego[:, None], camera2lidar)
    lidar_aug = torch.eye(4)[None].repeat(B, 1, 1).to(DEV)

    gt = LiDARInstance3DBoxes(
        torch.tensor(
            [
                [12.0, 3.0, -1.0, 1.9, 4.6, 1.7, 0.3, 2.0, 0.0],
                [-8.0, 5.0, -1.2, 0.7, 0.8, 1.75, -0.5, 0.5, 0.5],
                [20.0, -15.0, -0.9, 2.5, 6.5, 2.9, 1.2, -1.0, 0.2],
                [5.0, 5.0, -1.1, 0.6, 1.7, 1.6, 0.0, 0.0, 0.0],
                [-30.0, 10.0, -1.0, 2.0, 4.5, 1.6, 2.0, 3.0, 0.0],
            ],
            device=DEV,
        ),
        box_dim=9,
        with_yaw=True,
        origin=(0.5, 0.5, 0),
    )
    batch = dict(
        img=img,
        points=points,
        camera2ego=camera2ego,
        lidar2ego=lidar2ego,
        lidar2camera=lidar2camera,
        lidar2image=lidar2image,
        camera_intrinsics=camera_intrinsics,
        camera2lidar=camera2lidar,
        img_aug_matrix=img_aug,
        lidar_aug_matrix=lidar_aug,
        metas=[
            {
                "box_type_3d": LiDARInstance3DBoxes,
                "timestamp": 1532402927.647951,
                "token": f"fake{i}",
                "lidar_path": "fake.bin",
            }
            for i in range(B)
        ],
        depths=None,
        gt_bboxes_3d=[gt for _ in range(B)],
        gt_labels_3d=[torch.tensor([0, 8, 1, 9, 5], device=DEV) for _ in range(B)],
    )
    if with_seg_gt:
        batch["gt_masks_bev"] = torch.randint(0, 2, (B, 6, 200, 200), device=DEV).long()
    return batch


def make_fake_occ_batch(B=1):
    """img_inputs 7-tuple + occ GT tensors (train contract of the occ path)."""
    torch.manual_seed(0)
    imgs = torch.randn(B, 6, 3, 256, 704, device=DEV)
    cam_yaws = [0.0, -0.96, 0.96, math.pi, 2.18, -2.18]
    sensor2egos = torch.stack(
        [rigid(yaw, [1.7 * math.cos(yaw), 1.7 * math.sin(yaw), 1.5])
         for yaw in cam_yaws]
    )[None].repeat(B, 1, 1, 1).to(DEV)
    ego2globals = torch.eye(4, device=DEV)[None, None].repeat(B, 6, 1, 1)
    K = torch.eye(3, device=DEV)
    K[0, 0] = K[1, 1] = 280.0 * 256 / 704 * (704 / 256)  # focal at 256x704 scale
    K[0, 2], K[1, 2] = 352.0, 128.0
    intrins = K[None, None].repeat(B, 6, 1, 1)
    post_rots = torch.eye(3, device=DEV)[None, None].repeat(B, 6, 1, 1)
    post_trans = torch.zeros(B, 6, 3, device=DEV)
    bda = torch.eye(3, device=DEV)[None].repeat(B, 1, 1)
    voxel_semantics = torch.randint(0, 18, (B, 200, 200, 16), device=DEV).long()
    mask = (torch.rand(B, 200, 200, 16, device=DEV) > 0.3).float()
    return dict(
        img_inputs=[imgs, sensor2egos, ego2globals, intrins, post_rots,
                    post_trans, bda],
        voxel_semantics=voxel_semantics,
        mask_camera=mask,
        mask_lidar=mask,
        metas=[{"token": f"fake_occ{i}"} for i in range(B)],
    )


def load_ckpt(model, path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    msg = model.load_state_dict(state, strict=True)
    assert not msg.missing_keys and not msg.unexpected_keys, (
        f"missing={msg.missing_keys[:5]} unexpected={msg.unexpected_keys[:5]}"
    )
    n = len(state)
    print(f"[OK] strict load {path}: {n} keys, 0 missing / 0 unexpected")


def test_det():
    cfg = load_config(DET_CFG)
    model = build_model(cfg.model).to(DEV)
    load_ckpt(model, "ckpts/bevfusion-det-bf.pth")

    batch = make_fake_batch(B=1)

    # train-mode forward + backward under AMP (scale=1 backward for a
    # deterministic finiteness check; GradScaler step-skipping dynamics are
    # exercised in the real training loop)
    model.train()
    with torch.autocast("cuda", dtype=torch.float16):
        losses = model(**batch)
    assert all(torch.isfinite(v).all() for v in losses.values()
               if isinstance(v, torch.Tensor)), {
        k: float(v) for k, v in losses.items()
    }
    loss = total_loss(losses)
    print("[OK] det train loss:", round(loss.item(), 4), f"({len(losses)} terms)")
    loss.backward()
    gnorms = [
        p.grad.abs().sum().item()
        for n, p in model.named_parameters()
        if p.grad is not None
    ]
    assert len(gnorms) > 100, f"params with grads: {len(gnorms)}"
    assert all(math.isfinite(g) for g in gnorms), "nonfinite grads at scale=1"
    assert sum(gnorms) > 0
    print(f"[OK] det backward: {len(gnorms)} params with finite grads, abs-sum {sum(gnorms):.3g}")

    # eval-mode forward -> detections
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        outputs = model(**batch)
    out = outputs[0]
    assert "boxes_3d" in out and "scores_3d" in out and "labels_3d" in out
    print(
        "[OK] det eval:",
        type(out["boxes_3d"]).__name__,
        tuple(out["boxes_3d"].tensor.shape),
        "scores", tuple(out["scores_3d"].shape),
        "max score", round(out["scores_3d"].max().item(), 4),
    )
    assert torch.isfinite(out["scores_3d"]).all()
    del model, batch, losses, outputs
    torch.cuda.empty_cache()


def test_seg():
    cfg = load_config(SEG_CFG)
    model = build_model(cfg.model).to(DEV)
    load_ckpt(model, "ckpts/bevfusion-seg-bf.pth")

    batch = make_fake_batch(B=1, with_seg_gt=True)
    batch.pop("gt_bboxes_3d")
    batch.pop("gt_labels_3d")

    model.train()
    with torch.autocast("cuda", dtype=torch.float16):
        losses = model(**batch)
    loss = total_loss(losses)
    assert torch.isfinite(loss).all()
    print("[OK] seg train loss:", round(loss.item(), 4), f"({len(losses)} terms)")
    loss.backward()

    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        outputs = model(**batch)
    out = outputs[0]
    assert out["masks_bev"].shape == (6, 200, 200), out["masks_bev"].shape
    assert ((out["masks_bev"] >= 0) & (out["masks_bev"] <= 1)).all()
    print("[OK] seg eval: masks_bev", tuple(out["masks_bev"].shape), "in [0,1]")
    del model, batch, outputs
    torch.cuda.empty_cache()


def test_occ():
    cfg = load_config(OCC_CFG)
    model = build_model(cfg.model).to(DEV)
    load_ckpt(model, OCC_CKPT)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[OK] occ params {n_params / 1e6:.2f}M (official 44.74M)")
    assert abs(n_params / 1e6 - 44.74) < 0.1, n_params

    batch = make_fake_occ_batch(B=1)
    model.train()
    with torch.autocast("cuda", dtype=torch.float16):
        losses = model(**batch)
    loss = total_loss(losses)
    assert torch.isfinite(loss).all(), losses
    print("[OK] occ train loss:", round(loss.item(), 4), sorted(losses))
    loss.backward()
    gnorms = [p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None]
    assert all(math.isfinite(g) for g in gnorms), "nonfinite occ grads"

    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        outputs = model(**batch)
    assert outputs[0]["pred_occ"].shape == (200, 200, 16), outputs[0]["pred_occ"].shape
    print("[OK] occ eval: pred_occ", tuple(outputs[0]["pred_occ"].shape))

    # M0 head-width / depth-bin assertions on the unified tree
    cfg_m0 = load_config("configs/flashocc_r50_m0.py")
    m0 = build_model(cfg_m0.model)
    keys = dict(m0.state_dict())
    assert keys["heads.occ.final_conv.conv.weight"].shape == (128, 256, 3, 3)
    assert keys["heads.occ.predicter.0.weight"].shape == (256, 128)
    assert keys["heads.occ.predicter.2.weight"].shape[0] == 18 * 16
    assert m0.encoders["occ"].img_view_transformer.D == 44  # depth bins at 1.0 m
    print(f"[OK] M0 built: {len(keys)} keys, D=44")
    del model, m0, batch, outputs
    torch.cuda.empty_cache()


if __name__ == "__main__":
    print("=== det model ===")
    test_det()
    print("=== seg model ===")
    test_seg()
    print("=== occ model ===")
    test_occ()
    print("GATE C PASSED")
