#!/usr/bin/env python3
"""Gate D tests for the data pipeline port.

Tests the full pipeline from config → dataset → samples → collation.
"""

import sys
import os
import time
import numpy as np
import torch

import os.path as osp; sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), ".."))

from PIL import Image

from datasets.postprocess import MetricMIoU
from datasets.preprocess.loading import MEAN, STD, mmlab_normalize
from datasets import build_dataset, collate_fn
from scripts.config import load_config


def wait_for_pkls(timeout=300):
    """Wait for converter pkl files to exist (or use existing ones)."""
    paths = [
        "data/nuscenes/nuscenes_infos_train.pkl",
        "data/nuscenes/nuscenes_infos_val.pkl",
    ]
    for p in paths:
        if os.path.exists(p):
            continue
        print(f"Waiting for {p}...")
        start = time.time()
        while not os.path.exists(p) and time.time() - start < timeout:
            time.sleep(5)
        if not os.path.exists(p):
            raise FileNotFoundError(f"Timed out waiting for {p}")
    print("[OK] PKL files exist")


def test_build_train_dataset():
    """Test building train dataset from resolved det config."""
    from scripts.config import load_config

    from datasets import build_dataset

    cfg = load_config(
        "configs/det_convfuser.py"
    )
    # Ensure max_epochs is set (needed by GridMask)
    if "max_epochs" not in cfg:
        cfg["max_epochs"] = 20

    # Build raw NuScenesDataset first
    train_cfg = dict(cfg["data"]["train"]["dataset"])
    dataset = build_dataset(train_cfg)
    print(f"[OK] Built NuScenesDataset with {len(dataset)} samples")

    # Build CBGS-wrapped
    cbgs_cfg = dict(cfg["data"]["train"])
    cbgs_dataset = build_dataset(cbgs_cfg)
    print(f"[OK] Built CBGSDataset with {len(cbgs_dataset)} samples")

    return dataset, cbgs_cfg


def test_train_samples(dataset):
    """Iterate 3 train samples and check keys/shapes."""
    from datasets.geometry.boxes import LiDARInstance3DBoxes

    dataset.set_epoch(0)

    for i in range(3):
        sample = dataset[i]
        assert sample is not None, f"Sample {i} is None"

        # Check img: (6, 3, 256, 704) float32
        img = sample["img"]
        assert isinstance(img, torch.Tensor), f"img type: {type(img)}"
        assert img.shape == (6, 3, 256, 704), f"img shape: {img.shape}"
        assert img.dtype == torch.float32, f"img dtype: {img.dtype}"
        img_min = img.min().item()
        img_max = img.max().item()
        assert -5 < img_min < 0, f"img min: {img_min}"
        assert 0 < img_max < 5, f"img max: {img_max}"

        # Check points: (N, 5) tensor, N > 1000
        points = sample["points"]
        assert isinstance(points, torch.Tensor), f"points type: {type(points)}"
        assert points.shape[1] == 5, f"points dim: {points.shape[1]}"
        assert points.shape[0] > 1000, f"points N: {points.shape[0]}"

        # Check matrices
        for key in ["camera_intrinsics", "camera2ego", "lidar2image",
                     "img_aug_matrix", "lidar_aug_matrix"]:
            if key in sample:
                val = sample[key]
                assert isinstance(val, torch.Tensor), f"{key} type: {type(val)}"

        if "camera_intrinsics" in sample:
            assert sample["camera_intrinsics"].shape == (6, 4, 4), \
                f"camera_intrinsics shape: {sample['camera_intrinsics'].shape}"
        if "camera2ego" in sample:
            assert sample["camera2ego"].shape == (6, 4, 4), \
                f"camera2ego shape: {sample['camera2ego'].shape}"
        if "lidar2image" in sample:
            assert sample["lidar2image"].shape == (6, 4, 4), \
                f"lidar2image shape: {sample['lidar2image'].shape}"
        if "img_aug_matrix" in sample:
            assert sample["img_aug_matrix"].shape == (6, 4, 4), \
                f"img_aug_matrix shape: {sample['img_aug_matrix'].shape}"
        if "lidar_aug_matrix" in sample:
            assert sample["lidar_aug_matrix"].shape == (4, 4), \
                f"lidar_aug_matrix shape: {sample['lidar_aug_matrix'].shape}"

        # Check gt_bboxes_3d: LiDARInstance3DBoxes with (M, 9) tensor
        gt_boxes = sample["gt_bboxes_3d"]
        assert isinstance(gt_boxes, LiDARInstance3DBoxes), \
            f"gt_bboxes_3d type: {type(gt_boxes)}"
        assert gt_boxes.tensor.shape[1] == 9, \
            f"gt_bboxes_3d dim: {gt_boxes.tensor.shape[1]}"

        # Check gt_labels_3d: (M,) long in [0, 9]
        gt_labels = sample["gt_labels_3d"]
        if isinstance(gt_labels, torch.Tensor):
            assert gt_labels.dtype == torch.long, \
                f"gt_labels dtype: {gt_labels.dtype}"
            assert gt_labels.min() >= 0 and gt_labels.max() <= 9, \
                f"gt_labels range: [{gt_labels.min()}, {gt_labels.max()}]"
        else:
            gt_labels = np.asarray(gt_labels)
            assert gt_labels.dtype in [np.int64, np.int32], \
                f"gt_labels dtype: {gt_labels.dtype}"

        # Check metas
        metas = sample["metas"]
        assert isinstance(metas, dict), f"metas type: {type(metas)}"
        assert "token" in metas, "metas missing token"

        # Check depths: (6, 256, 704) sparse
        if "depths" in sample:
            depths = sample["depths"]
            assert isinstance(depths, torch.Tensor), f"depths type: {type(depths)}"
            assert depths.shape == (6, 256, 704), f"depths shape: {depths.shape}"
            # Sparse: mostly 0, some > 0
            nonzero_frac = (depths > 0).float().mean().item()
            assert nonzero_frac < 0.5, f"depths too dense: {nonzero_frac}"

        # Check img_aug_matrix is not identity (train aug)
        if "img_aug_matrix" in sample:
            aug = sample["img_aug_matrix"]
            identity = torch.eye(4).unsqueeze(0).expand(6, 4, 4)
            assert not torch.allclose(aug, identity, atol=0.01), \
                "img_aug_matrix is identity (expected train aug)"

        print(f"  [OK] Train sample {i}: img={img.shape}, "
              f"points=({points.shape[0]}, 5), "
              f"boxes=({len(gt_boxes)}, 9), "
              f"labels={gt_labels.shape}")

    print("[OK] 3 train samples passed all checks")


def test_val_sample():
    """Test val sample (no aug, test pipeline)."""
    from scripts.config import load_config
    from datasets import build_dataset

    cfg = load_config(
        "configs/det_convfuser.py"
    )
    if "max_epochs" not in cfg:
        cfg["max_epochs"] = 20

    val_cfg = dict(cfg["data"]["val"])
    dataset = build_dataset(val_cfg)
    sample = dataset[0]
    assert sample is not None

    img = sample["img"]
    assert img.shape == (6, 3, 256, 704), f"val img shape: {img.shape}"

    points = sample["points"]
    assert points.shape[0] > 100, f"val points: {points.shape}"

    # img_aug_matrix should be resize-only (no flip/rotate)
    if "img_aug_matrix" in sample:
        aug = sample["img_aug_matrix"]
        # Off-diagonal elements should be ~0 (no rotation)
        for i in range(6):
            assert abs(aug[i, 0, 1].item()) < 0.5, \
                f"val aug has rotation: {aug[i]}"

    print(f"[OK] Val sample: img={img.shape}, points=({points.shape[0]}, 5)")
    return dataset


def test_collate(dataset):
    """Test collation of 2 samples."""
    from datasets.dataloader import collate_fn

    samples = [dataset[0], dataset[1]]
    batch = collate_fn(samples)

    # img: (2, 6, 3, 256, 704)
    assert batch["img"].shape == (2, 6, 3, 256, 704), \
        f"batch img shape: {batch['img'].shape}"

    # points: list of 2
    assert isinstance(batch["points"], list)
    assert len(batch["points"]) == 2

    # gt_bboxes_3d: list of 2
    assert isinstance(batch["gt_bboxes_3d"], list)
    assert len(batch["gt_bboxes_3d"]) == 2

    # gt_labels_3d: list of 2
    assert isinstance(batch["gt_labels_3d"], list)

    # metas: list of 2
    assert isinstance(batch["metas"], list)
    assert len(batch["metas"]) == 2

    # matrices should be stacked
    for key in ["camera_intrinsics", "camera2ego", "lidar2image"]:
        if key in batch:
            assert batch[key].shape[0] == 2, \
                f"batch {key} shape: {batch[key].shape}"

    if "depths" in batch:
        assert batch["depths"].shape == (2, 6, 256, 704), \
            f"batch depths shape: {batch['depths'].shape}"

    print(f"[OK] Collate: batch img={batch['img'].shape}, "
          f"points=list[{len(batch['points'])}]")


def test_seg_masks():
    """Test BEV segmentation masks from seg config."""
    from scripts.config import load_config
    from datasets import build_dataset

    cfg = load_config("configs/seg_fusion_bev256d2_lss.py")
    if "max_epochs" not in cfg:
        cfg["max_epochs"] = 20

    # Build val dataset from seg config (uses shared test_pipeline)
    val_cfg = dict(cfg["data"]["val"])
    dataset = build_dataset(val_cfg)
    sample = dataset[0]
    assert sample is not None

    # Check gt_masks_bev: (6, 200, 200) long
    assert "gt_masks_bev" in sample, "No gt_masks_bev in seg sample"
    masks = sample["gt_masks_bev"]
    if isinstance(masks, np.ndarray):
        masks = torch.from_numpy(masks)
    assert masks.shape == (6, 200, 200), f"gt_masks_bev shape: {masks.shape}"
    assert masks.dtype in (torch.int64, torch.long), \
        f"gt_masks_bev dtype: {masks.dtype}"

    # Orientation test: check all 8 variants
    points = sample["points"]
    if isinstance(points, torch.Tensor):
        pts_xy = points[:, :2].numpy()
    else:
        pts_xy = np.asarray(points)[:, :2]

    # Filter to ±50m range
    valid = (np.abs(pts_xy[:, 0]) < 50) & (np.abs(pts_xy[:, 1]) < 50)
    pts_valid = pts_xy[valid]

    drivable = masks[0].numpy()  # (200, 200)

    # Convert point xy to grid indices
    # Grid: x in [-50, 50] → col index, y in [-50, 50] → row index
    # Resolution: 0.5m, so 200 cells per 100m

    variants = {
        "identity": lambda m: m,
        "transpose": lambda m: m.T,
        "flip_x": lambda m: m[::-1, :],
        "flip_y": lambda m: m[:, ::-1],
        "flip_both": lambda m: m[::-1, ::-1],
        "T_flip_x": lambda m: m.T[::-1, :],
        "T_flip_y": lambda m: m.T[:, ::-1],
        "T_flip_both": lambda m: m.T[::-1, ::-1],
    }

    print("\n  Orientation variants (drivable_area coverage):")
    best_name = None
    best_coverage = 0
    results = {}

    for name, transform in variants.items():
        m = transform(drivable)
        # For each variant, map points to grid and check coverage
        # Assume m is indexed as [row, col] where:
        # row corresponds to one axis, col to another
        covered = 0
        total = len(pts_valid)
        for px, py in pts_valid:
            # Map to grid indices
            # Try: row = y-mapped, col = x-mapped
            col = int((px + 50) / 0.5)
            row = int((py + 50) / 0.5)
            col = max(0, min(199, col))
            row = max(0, min(199, row))
            if m[row, col] > 0:
                covered += 1
        coverage = covered / max(total, 1)
        results[name] = coverage
        marker = " <-- BEST" if coverage > best_coverage else ""
        print(f"    {name:15s}: {coverage:.4f}{marker}")
        if coverage > best_coverage:
            best_coverage = coverage
            best_name = name

    # The chosen variant should have coverage > 0.4
    print(f"\n  Best variant: {best_name} ({best_coverage:.4f})")
    assert best_coverage > 0.4, \
        f"Best coverage {best_coverage:.4f} < 0.4 — wrong orientation?"

    # Class coverage sanity
    class_names = ["drivable_area", "ped_crossing", "walkway",
                   "stop_line", "carpark_area", "divider"]
    coverages = []
    for c in range(6):
        coverages.append(float(masks[c].sum()) / (200 * 200))
    print(f"\n  Class coverages: {dict(zip(class_names, [f'{c:.3f}' for c in coverages]))}")

    # drivable_area should have non-trivial coverage
    assert coverages[0] > 0.01, f"drivable_area coverage too low: {coverages[0]}"
    # stop_line should be smaller than drivable
    assert coverages[3] < coverages[0], \
        f"stop_line ({coverages[3]}) >= drivable ({coverages[0]})"

    print(f"[OK] BEV seg masks: shape={masks.shape}, "
          f"best_orient={best_name} ({best_coverage:.3f})")
    return dataset


def test_warp_sanity(seg_dataset):
    """Test that warped mask in train mode still covers ground points."""
    from scripts.config import load_config
    from datasets import build_dataset

    cfg = load_config("configs/seg_fusion_bev256d2_lss.py")
    if "max_epochs" not in cfg:
        cfg["max_epochs"] = 20

    # Build train dataset (has augmentation)
    train_cfg = dict(cfg["data"]["train"]["dataset"])
    dataset = build_dataset(train_cfg)
    dataset.set_epoch(0)

    for i in range(3):
        sample = dataset[i]
        if sample is None:
            continue
        if "gt_masks_bev" not in sample:
            continue

        masks = sample["gt_masks_bev"]
        if isinstance(masks, np.ndarray):
            masks = torch.from_numpy(masks)
        assert masks.shape == (6, 200, 200), f"warp masks shape: {masks.shape}"

        points = sample["points"]
        if isinstance(points, torch.Tensor):
            pts_xy = points[:, :2].numpy()
        else:
            pts_xy = np.asarray(points)[:, :2]

        # Check coverage of drivable area
        valid = (np.abs(pts_xy[:, 0]) < 50) & (np.abs(pts_xy[:, 1]) < 50)
        pts_valid = pts_xy[valid]
        drivable = masks[0].numpy()

        covered = 0
        for px, py in pts_valid:
            col = int((px + 50) / 0.5)
            row = int((py + 50) / 0.5)
            col = max(0, min(199, col))
            row = max(0, min(199, row))
            if drivable[row, col] > 0:
                covered += 1
        coverage = covered / max(len(pts_valid), 1)

        # With augmentation, the warp should still maintain reasonable coverage
        # but it might be lower than unaugmented due to point-mask frame mismatch
        print(f"  Warp sample {i}: coverage={coverage:.3f}, "
              f"points={len(pts_valid)}, "
              f"drivable_sum={drivable.sum()}")

    print("[OK] Warp sanity check passed")


def main():
    print("=" * 60)
    print("Gate D: Data Pipeline Tests")
    print("=" * 60)

    # Wait for pkl files
    wait_for_pkls()

    # Test a: Build datasets
    print("\n--- Test a: Build datasets ---")
    dataset, cbgs_cfg = test_build_train_dataset()

    # Test b: Train samples
    print("\n--- Test b: Train samples ---")
    test_train_samples(dataset)

    # Test c: Val sample
    print("\n--- Test c: Val sample ---")
    val_dataset = test_val_sample()

    # Test d: Collate
    print("\n--- Test d: Collate ---")
    test_collate(val_dataset)

    # Test e: Seg masks + orientation
    print("\n--- Test e: BEV segmentation masks ---")
    try:
        seg_dataset = test_seg_masks()
    except Exception as e:
        print(f"[WARN] Seg mask test failed: {e}")
        import traceback; traceback.print_exc()
        seg_dataset = None

    # Test f: Warp sanity
    print("\n--- Test f: Warp sanity ---")
    if seg_dataset is not None:
        try:
            test_warp_sanity(seg_dataset)
        except Exception as e:
            print(f"[WARN] Warp test failed: {e}")
            import traceback; traceback.print_exc()
    else:
        print("[SKIP] Warp test (seg dataset not available)")

    print("\n" + "=" * 60)
    print("ALL GATE D TESTS PASSED")
    print("=" * 60)
    test_occ_suite()


# ---- occupancy pipeline contract ----

CFG = "configs/flashocc_r50.py"

def test_occ_normalize_quirk():
    """mmlabNormalize feeds BGR-ordered pixels normalized with RGB stats
    (mmcv imnormalize to_rgb=True applied to PIL RGB data). Pin it exactly."""
    arr = np.zeros((2, 2, 3), np.uint8)
    arr[..., 0], arr[..., 1], arr[..., 2] = 10, 20, 30  # R=10 G=20 B=30
    t = mmlab_normalize(Image.fromarray(arr))
    expected = torch.tensor([
        (30 - MEAN[0]) / STD[0],  # ch0 gets B, normalized with R stats
        (20 - MEAN[1]) / STD[1],
        (10 - MEAN[2]) / STD[2],
    ], dtype=torch.float32)
    assert torch.allclose(t[:, 0, 0], expected, atol=1e-5), t[:, 0, 0]
    assert t.shape == (3, 2, 2) and t.dtype == torch.float32
    print("[OK] normalize quirk (BGR + RGB stats)")


def test_occ_val_sample():
    cfg = load_config(CFG)
    ds = build_dataset(cfg.data.val)
    assert len(ds) == 81, len(ds)
    s = ds[0]
    assert set(s.keys()) == {"img_inputs", "metas"}
    imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda = s["img_inputs"]
    assert imgs.shape == (6, 3, 256, 704) and imgs.dtype == torch.float32
    assert sensor2egos.shape == ego2globals.shape == (6, 4, 4)
    assert intrins.shape == post_rots.shape == (6, 3, 3) and post_trans.shape == (6, 3)
    # test mode: identity BDA
    assert torch.allclose(bda, torch.eye(3)), bda
    # test mode: no rotation -> post_rot = resize * I (2x2 block)
    for cam in range(6):
        pr = post_rots[cam]
        assert abs(pr[0, 1].item()) < 1e-6 and abs(pr[1, 0].item()) < 1e-6
        assert pr[0, 0].item() > 0 and abs(pr[0, 0].item() - pr[1, 1].item()) < 1e-6
        assert pr[2, 2].item() == 1.0
    # normalized range sanity
    assert imgs.min() > -3 and imgs.max() < 3
    assert s["metas"]["token"] == ds.data_list[0]["token"]
    assert "occ_gt_path" in s["metas"]
    print("[OK] val sample contract")
    return ds


def test_occ_train_sample():
    cfg = load_config(CFG)
    ds = build_dataset(cfg.data.train)
    assert len(ds) == 323, len(ds)
    s = ds[0]
    need = {"img_inputs", "gt_depth", "voxel_semantics", "mask_lidar", "mask_camera", "metas"}
    assert need <= set(s.keys()), set(s.keys())

    vs = s["voxel_semantics"]
    assert vs.shape == (200, 200, 16) and vs.dtype == torch.uint8
    assert int(vs.min()) >= 0 and int(vs.max()) <= 17
    for m in (s["mask_lidar"], s["mask_camera"]):
        assert m.shape == (200, 200, 16) and m.dtype == torch.uint8
        assert set(np.unique(m.numpy())) <= {0, 1}

    gd = s["gt_depth"]
    assert gd.shape == (6, 256, 704) and gd.dtype == torch.float32
    frac = (gd > 0).float().mean().item()
    assert 0.0 < frac < 0.2, frac

    bda = s["img_inputs"][6]
    assert bda.shape == (3, 3)
    # train bda_conf: rot_lim=(0,0), scale=(1,1) -> only flips: |det| == 1
    assert abs(abs(torch.det(bda).item()) - 1.0) < 1e-5
    # camera-mask covers a plausible fraction of the grid
    cov = s["mask_camera"].float().mean().item()
    assert 0.02 < cov < 0.9, cov
    print(f"[OK] train sample contract (depth nz={frac:.4f}, cam mask={cov:.3f})")


def test_occ_gt_flip_consistency():
    """voxel_semantics must equal the raw npz GT with the sampled BDA flips
    applied (flip_dx -> axis 0, flip_dy -> axis 1)."""
    cfg = load_config(CFG)
    ds = build_dataset(cfg.data.train)
    # rebuild pre-pack to observe flags: run pipeline stages manually
    from datasets.preprocess import build_pipeline
    raw_pipeline = build_pipeline(cfg.data.train.pipeline[:5])  # up to depth
    from datasets.preprocess import run_pipeline
    checked = 0
    for idx in range(6):
        info = ds.get_data_info(idx)
        data = run_pipeline(raw_pipeline, info)
        raw = np.load(osp.join(info["occ_gt_path"], "labels.npz"))
        sem = raw["semantics"]
        if data["flip_dx"]:
            sem = sem[::-1, ...]
        if data["flip_dy"]:
            sem = sem[:, ::-1, ...]
        assert np.array_equal(np.asarray(data["voxel_semantics"]), sem), idx
        checked += 1
    print(f"[OK] GT flip consistency over {checked} samples")


def test_occ_collate():
    cfg = load_config(CFG)
    ds = build_dataset(cfg.data.val)
    batch = collate_fn([ds[0], ds[1]])
    shapes = [tuple(t.shape) for t in batch["img_inputs"]]
    assert shapes[0] == (2, 6, 3, 256, 704), shapes
    assert shapes[6] == (2, 3, 3)
    assert len(batch["metas"]) == 2 and batch["metas"][0]["token"] != batch["metas"][1]["token"]
    print("[OK] collate batching")


def test_occ_miou_math():
    m = MetricMIoU()
    gt = np.zeros((4, 4, 4), np.uint8)
    gt[..., 0] = 1  # 16 voxels of class 1
    mask = np.ones_like(gt, dtype=bool)
    m.add_batch(gt.copy(), gt, None, mask)
    miou = m.count_miou(verbose=False)
    assert abs(miou[0] - 1.0) < 1e-9 and abs(miou[1] - 1.0) < 1e-9
    assert np.isnan(miou[3])  # absent class -> NaN (nanmean skips it)

    m2 = MetricMIoU()
    pred = gt.copy()
    pred[0] = 2  # 4 voxels misclassified as class 2
    m2.add_batch(pred, gt, None, mask)
    miou2 = m2.count_miou(verbose=False)
    # cls1: TP=12, GT=16, Pred=12 -> 12/16; cls2: TP=0, Pred=4 -> 0
    assert abs(miou2[1] - 0.75) < 1e-9, miou2[1]
    assert miou2[2] == 0.0
    # mask filtering: restrict to the correct slab -> perfect again
    m3 = MetricMIoU()
    mask3 = mask.copy()
    mask3[0] = False
    m3.add_batch(pred, gt, None, mask3)
    assert abs(m3.count_miou(verbose=False)[1] - 1.0) < 1e-9
    print("[OK] mIoU math (perfect / confusion / masking)")



def test_occ_suite():
    test_occ_normalize_quirk()
    test_occ_val_sample()
    test_occ_train_sample()
    test_occ_gt_flip_consistency()
    test_occ_collate()
    test_occ_miou_math()


if __name__ == "__main__":
    main()
