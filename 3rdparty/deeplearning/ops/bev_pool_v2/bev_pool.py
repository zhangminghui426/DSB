"""bev_pool_v2 — pillar-based BEV pooling CUDA op (LSS view transform).

Ported from the FlashOCC plugin (Phigent Robotics, arXiv:2211.17111); the only
BEV pool in this repo. fp32-only kernel; built by the top-level setup.py.

Three entry points:
- bev_pool_v2: BEVDepth-style API (depth x feat multiplied inside the kernel,
  ranks precomputed by the caller) — used by the occupancy view transforms.
- bev_pool_v2_sum: depth-already-folded API (BEVFusion-style call convention:
  (N, C) feats + (N, 4) coords) expressed as a degenerate v2 configuration —
  used by the det/seg view transforms.
- bev_pool_v2_reference: pure-torch (index_add_) equivalent, numeric reference
  for tests and the eager side of the ONNX export path.
"""

import torch

from . import bev_pool_v2_ext

__all__ = ["bev_pool_v2", "bev_pool_v2_sum", "bev_pool_v2_reference",
           "scatter_pool", "bev_pool_plugin_fn"]


class QuickCumsumCuda(torch.autograd.Function):
    """BEVPoolv2: out[rank_bev] += depth[rank_depth] * feat[rank_feat] over
    sorted pillar intervals."""

    @staticmethod
    def forward(ctx, depth, feat, ranks_depth, ranks_feat, ranks_bev,
                bev_feat_shape, interval_starts, interval_lengths):
        ranks_bev = ranks_bev.int()
        depth = depth.contiguous().float()   # (B, N, D, fH, fW)
        feat = feat.contiguous().float()     # (B, N, fH, fW, C)
        ranks_depth = ranks_depth.contiguous().int()
        ranks_feat = ranks_feat.contiguous().int()
        interval_lengths = interval_lengths.contiguous().int()
        interval_starts = interval_starts.contiguous().int()

        out = feat.new_zeros(bev_feat_shape)  # (B, Dz, Dy, Dx, C)

        bev_pool_v2_ext.bev_pool_v2_forward(
            depth, feat, out,
            ranks_depth, ranks_feat, ranks_bev,
            interval_lengths, interval_starts,
        )

        ctx.save_for_backward(ranks_bev, depth, feat, ranks_feat, ranks_depth)
        return out

    @staticmethod
    def backward(ctx, out_grad):
        ranks_bev, depth, feat, ranks_feat, ranks_depth = ctx.saved_tensors

        order = ranks_feat.argsort()
        ranks_feat, ranks_depth, ranks_bev = (
            ranks_feat[order], ranks_depth[order], ranks_bev[order]
        )
        kept = torch.ones(ranks_bev.shape[0], device=ranks_bev.device, dtype=torch.bool)
        kept[1:] = ranks_feat[1:] != ranks_feat[:-1]
        interval_starts_bp = torch.where(kept)[0].int()
        interval_lengths_bp = torch.zeros_like(interval_starts_bp)
        interval_lengths_bp[:-1] = interval_starts_bp[1:] - interval_starts_bp[:-1]
        interval_lengths_bp[-1] = ranks_bev.shape[0] - interval_starts_bp[-1]

        depth = depth.contiguous()
        feat = feat.contiguous()
        ranks_depth = ranks_depth.contiguous()
        ranks_feat = ranks_feat.contiguous()
        ranks_bev = ranks_bev.contiguous()
        interval_lengths_bp = interval_lengths_bp.contiguous()
        interval_starts_bp = interval_starts_bp.contiguous()

        depth_grad = depth.new_zeros(depth.shape)
        feat_grad = feat.new_zeros(feat.shape)
        out_grad = out_grad.contiguous()
        bev_pool_v2_ext.bev_pool_v2_backward(
            out_grad, depth_grad, feat_grad,
            depth, feat,
            ranks_depth, ranks_feat, ranks_bev,
            interval_lengths_bp, interval_starts_bp,
        )
        return depth_grad, feat_grad, None, None, None, None, None, None


def bev_pool_v2(depth, feat, ranks_depth, ranks_feat, ranks_bev,
                bev_feat_shape, interval_starts, interval_lengths):
    """CUDA pillar pooling.

    Args:
        depth: (B, N, D, fH, fW)   feat: (B, N, fH, fW, C)
        ranks_depth / ranks_feat / ranks_bev: (P,) flat indices
        bev_feat_shape: (B, Dz, Dy, Dx, C)
        interval_starts / interval_lengths: (N_pillar,)

    Returns:
        (B, C, Dz, Dy, Dx)
    """
    x = QuickCumsumCuda.apply(
        depth, feat, ranks_depth, ranks_feat, ranks_bev,
        bev_feat_shape, interval_starts, interval_lengths,
    )  # (B, Dz, Dy, Dx, C)
    return x.permute(0, 4, 1, 2, 3).contiguous()


def bev_pool_v2_sum(feats, coords, B, D, H, W):
    """BEVPoolv2 with depth already folded into the features.

    BEVFusion-style call convention: sum the (N, C) rows per (x, y, z, batch)
    BEV cell. Expressed as a degenerate v2 configuration — depth = ones, so
    the kernel's depth * feat reduces to the identity; sorting/intervals are
    computed here (the v2 contract puts them on the caller).

    Args:
        feats: (N, C) float tensor of features.
        coords: (N, 4) int tensor, columns = (x, y, z, batch).
        B, D, H, W: batch / z bins / bev x / bev y.

    Returns:
        (B, C, D, H, W) pooled tensor (same layout as the legacy bev_pool).
    """
    assert feats.shape[0] == coords.shape[0]
    n, c = feats.shape
    if n == 0:
        return feats.new_zeros(B, c, D, H, W)
    # kernel indexes feat in int32 as ranks_feat * C + c
    assert n * c < 2 ** 31, "P*C overflows the kernel's int32 feat indexing"

    # v2 axis order (Dz, Dy, Dx) = (z, x, y) = (D, H, W) — the BEVFusion grid
    # convention (transposed vs the occupancy (z, y, x) one); ranks_bev
    # indexes out.view(-1, C) of the (B, D, H, W, C) kernel output and
    # reproduces the legacy kernel offset b*(d*h*w)+z*(h*w)+x*w+y exactly.
    ranks_bev = (coords[:, 3] * (D * H * W) + coords[:, 2] * (H * W)
                 + coords[:, 0] * W + coords[:, 1])
    order = ranks_bev.argsort()
    ranks_bev = ranks_bev[order]
    # ranks_feat must stay a permutation (unique rows): backward re-groups
    # intervals by ranks_feat and singleton intervals are what make the
    # folded gradient bitwise identical to the legacy broadcast.
    ranks_feat = order.int()   # row index into feat.reshape(-1, C)
    ranks_depth = order.int()  # row index into depth.reshape(-1) (all ones)

    kept = torch.ones(n, device=feats.device, dtype=torch.bool)
    kept[1:] = ranks_bev[1:] != ranks_bev[:-1]
    interval_starts = torch.where(kept)[0].int()
    interval_lengths = torch.zeros_like(interval_starts)
    interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
    interval_lengths[-1] = n - interval_starts[-1]

    depth = feats.new_ones(1, n, 1, 1, 1)
    feat = feats.reshape(1, n, 1, 1, c)
    # (B, Dz=z, Dy=x, Dx=y, C) -> permute inside bev_pool_v2 -> (B, C, D, H, W)
    return bev_pool_v2(depth, feat, ranks_depth, ranks_feat, ranks_bev,
                       (B, D, H, W, c), interval_starts, interval_lengths)


def bev_pool_v2_reference(depth, feat, ranks_depth, ranks_feat, ranks_bev,
                          bev_feat_shape, interval_starts, interval_lengths):
    """Pure-torch equivalent (index_add_) — numeric reference for tests.

    Same inputs/outputs as bev_pool_v2; interval_* are unused (index_add_
    accumulates all points directly).
    """
    depth = depth.contiguous().float()
    feat = feat.contiguous().float()
    num_channels = feat.shape[-1]
    depth_flat = depth.reshape(-1)[ranks_depth.long()]            # (P,)
    feat_flat = feat.reshape(-1, num_channels)[ranks_feat.long()]  # (P, C)
    out = feat.new_zeros(bev_feat_shape)                           # (B,Dz,Dy,Dx,C)
    out.view(-1, num_channels).index_add_(
        0, ranks_bev.long(), depth_flat[:, None] * feat_flat
    )
    return out.permute(0, 4, 1, 2, 3).contiguous()


def scatter_pool(x, geom, B, D, H, W):
    """Segment scatter-add into the (B, D, H, W, C) grid (torch reference for
    the BevPool TRT plugin; also the ONNX-export-time eager implementation).
    geom columns = (x, y, z, batch)."""
    C = x.shape[1]
    flat = (
        geom[:, 3] * (D * H * W)
        + geom[:, 2] * (H * W)
        + geom[:, 0] * W
        + geom[:, 1]
    )
    out = torch.zeros(B * D * H * W, C, device=x.device)
    out.scatter_add_(0, flat.unsqueeze(1).expand(-1, C), x)
    return out.view(B, D, H, W, C)


try:
    torch.ops.bf.bev_pool  # noqa: B018  (probe; raises if not defined yet)
except AttributeError:
    torch.library.define(
        "bf::bev_pool",
        "(Tensor x, Tensor geom, int B, int D, int H, int W) -> Tensor",
    )
    torch.library.impl(
        "bf::bev_pool", "CompositeExplicitAutograd", scatter_pool
    )

    def bev_pool_symbolic(g, x, geom, B, D, H, W):
        # custom ONNX op resolved by deploy/plugin/libbf_bevpool_plugin.so
        # (attribute names match the plugin creator's PluginFields)
        return g.op("bf.plugins::BevPool", x, geom, B_i=B, D_i=D, H_i=H, W_i=W)

    torch.onnx.register_custom_op_symbolic("bf::bev_pool", bev_pool_symbolic, 17)


def bev_pool_plugin_fn(x, geom, B, D, H, W):
    """Export-time bev_pool: torch.library op whose ONNX symbolic emits the
    bf.plugins::BevPool custom op; eager execution runs scatter_pool."""
    return torch.ops.bf.bev_pool(x, geom, B, D, H, W)
