"""Standalone verification tests for SwinTransformer port.

Run from repo root (/home/wika/wk_robot/aigorithm/bevfusion):
    python tests/test_swin.py

Tests:
  1. Parameter key/shape matching against bevfusion-det.pth checkpoint
  2. Forward pass shapes with input (2, 3, 256, 704)
  3. Forward under torch.autocast cuda fp16
"""

import os
import sys

# Make sure the repo root is importable
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
from models.backbones.swin import SwinTransformer


def make_model(**kwargs):
    """Build a SwinTransformer with the BEVFusion default config."""
    defaults = dict(
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=[1, 2, 3],
        with_cp=False,
    )
    defaults.update(kwargs)
    return SwinTransformer(**defaults)


# -----------------------------------------------------------------------
# Test 1: Parameter key/shape matching
# -----------------------------------------------------------------------

def test_checkpoint_keys():
    """Verify state_dict keys/shapes match the bevfusion-det checkpoint exactly."""
    ckpt_path = os.path.join(REPO_ROOT, "ckpts", "bevfusion-det.pth")
    if not os.path.exists(ckpt_path):
        print(f"[SKIP] test_checkpoint_keys: checkpoint not found at {ckpt_path}")
        return True

    m = make_model()
    mine = {k: tuple(v.shape) for k, v in m.state_dict().items()}

    ck = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    prefix = "encoders.camera.backbone."
    ref = {
        k.replace(prefix, ""): tuple(v.shape)
        for k, v in ck.items()
        if k.startswith(prefix)
    }

    missing = sorted(set(ref) - set(mine))
    unexpected = sorted(set(mine) - set(ref))
    common = set(ref) & set(mine)
    mismatched = sorted(k for k in common if ref[k] != mine[k])

    print(f"  ref keys: {len(ref)}, mine keys: {len(mine)}")
    print(f"  missing ({len(missing)}): {missing[:10]}")
    print(f"  unexpected ({len(unexpected)}): {unexpected[:10]}")
    print(f"  shape-mismatch ({len(mismatched)}): {mismatched[:10]}")

    ok = (not missing) and (not unexpected) and (not mismatched)
    if ok:
        print("[OK] test_checkpoint_keys")
    else:
        print("[FAIL] test_checkpoint_keys")
    return ok


# -----------------------------------------------------------------------
# Test 2: Forward pass shapes
# -----------------------------------------------------------------------

def test_forward_shapes():
    """Forward with input (2, 3, 256, 704) → outputs [(2,192,32,88), (2,384,16,44), (2,768,8,22)].

    Note: The task description expected [(2,192,64,176),...] but that would
    require 192 channels at H/4 resolution, which doesn't exist in standard
    Swin-Tiny. The correct output for out_indices=[1,2,3] is:
      Stage 1: (2, 192, 32, 88)   — H/8, W/8
      Stage 2: (2, 384, 16, 44)   — H/16, W/16
      Stage 3: (2, 768, 8, 22)    — H/32, W/32
    """
    m = make_model()
    m.eval()

    x = torch.randn(2, 3, 256, 704)
    with torch.no_grad():
        outs = m(x)

    expected = [
        (2, 192, 32, 88),
        (2, 384, 16, 44),
        (2, 768, 8, 22),
    ]

    ok = True
    for i, (out, exp) in enumerate(zip(outs, expected)):
        if tuple(out.shape) != exp:
            print(f"  output[{i}] shape: {tuple(out.shape)}, expected {exp}")
            ok = False
        else:
            print(f"  output[{i}] shape: {tuple(out.shape)} OK")

    if ok:
        print("[OK] test_forward_shapes")
    else:
        print("[FAIL] test_forward_shapes")
    return ok


# -----------------------------------------------------------------------
# Test 3: Forward under autocast fp16
# -----------------------------------------------------------------------

def test_forward_autocast():
    """Forward under torch.autocast(device_type='cuda', dtype=float16)."""
    if not torch.cuda.is_available():
        print("[SKIP] test_forward_autocast: CUDA not available")
        return True

    m = make_model().cuda()
    m.eval()

    x = torch.randn(2, 3, 256, 704, device="cuda")
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outs = m(x)

    expected = [
        (2, 192, 32, 88),
        (2, 384, 16, 44),
        (2, 768, 8, 22),
    ]

    ok = True
    for i, (out, exp) in enumerate(zip(outs, expected)):
        if tuple(out.shape) != exp:
            print(f"  output[{i}] shape: {tuple(out.shape)}, expected {exp}")
            ok = False
        else:
            print(f"  output[{i}] shape: {tuple(out.shape)} OK")

    if ok:
        print("[OK] test_forward_autocast")
    else:
        print("[FAIL] test_forward_autocast")
    return ok


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

if __name__ == "__main__":
    results = []

    print("=" * 60)
    print("Test 1: Checkpoint key/shape matching")
    print("=" * 60)
    results.append(test_checkpoint_keys())

    print()
    print("=" * 60)
    print("Test 2: Forward pass shapes")
    print("=" * 60)
    results.append(test_forward_shapes())

    print()
    print("=" * 60)
    print("Test 3: Forward under autocast fp16")
    print("=" * 60)
    results.append(test_forward_autocast())

    print()
    print("=" * 60)
    if all(results):
        print("ALL TESTS PASSED")
        sys.exit(0)
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)
