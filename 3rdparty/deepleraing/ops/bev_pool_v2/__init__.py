"""BEV pooling v2 — scatter-pool camera features into BEV.

Pure PyTorch implementation (scatter_add_) for CPU/ONNX export;
CUDA kernel version is optional (ops/bev_pool_v2/bev_pool_v2_cuda.py).
"""

import torch


def bev_pool_v2(
    depth_net,  # (B, N, D, fH, fW)
    feat,       # (B, N, C, fH, fW)
    geom_feats, # (B, X, 3) — (x, y, z) BEV coords in voxel grid
    ranks,      # (B, X) — rank indices
    interval,   # scalar — voxel size in BEV
    b,          # batch size
    d,          # depth size
    h,          # BEV height
    w,          # BEV width
):
    """Scatter-pool camera features into BEV grid.

    Pure PyTorch version using scatter_add_ (ONNX-export friendly).
    """
    B, N, D, fH, fW = depth_net.shape
    B, N, C, fH2, fW2 = feat.shape

    # Combine depth distribution with image features
    # depth_net: (B, N, D, fH, fW)
    # feat: (B, N, C, fH, fW)
    x = depth_net.unsqueeze(2) * feat.unsqueeze(3)  # (B, N, C, D, fH, fW)
    x = x.view(B, N * C * D, fH, fW)

    # Scatter into BEV grid
    out = torch.zeros(b, C, d, h, w, device=feat.device, dtype=feat.dtype)

    for batch_idx in range(b):
        batch_geom = geom_feats[batch_idx]  # (X, 3)
        batch_ranks = ranks[batch_idx]       # (X,)

        # Flatten spatial dimensions
        x_flat = x[batch_idx].view(-1, fH * fW)  # (N*C*D, fH*fW)

        # Scatter add
        idx = batch_ranks.long()
        out_flat = out[batch_idx].view(-1, h * w)  # (C*d, h*w)
        out_flat.index_add_(1, idx, x_flat[:, idx])

    return out.view(b, C, d, h, w)


class _BEVPoolv2(torch.autograd.Function):
    """Autograd wrapper for bev_pool_v2 (matches CUDA API)."""

    @staticmethod
    def forward(ctx, depth, feat, ranks_depth, ranks_feat, ranks_bev,
                interval_starts, interval_lengths, d, h, w):
        # Simplified: use pure pytorch scatter
        B, N, D, fH, fW = depth.shape
        B, N, C, _, _ = feat.shape

        out = torch.zeros(B, C, d, h, w, device=depth.device, dtype=depth.dtype)
        x = depth.unsqueeze(2) * feat.unsqueeze(3)  # (B, N, C, D, fH, fW)
        x = x.view(B, N * C * D, -1)

        for b in range(B):
            rd = ranks_depth[b].long()
            rf = ranks_feat[b].long()
            rb = ranks_bev[b].long()

            # Gather features at valid positions
            feat_gathered = x[b, :, rf]  # (N*C*D, X)
            out_flat = out[b].view(-1, h * w)
            out_flat.index_add_(1, rb, feat_gathered)

        ctx.save_for_backward(depth, feat, ranks_depth, ranks_feat, ranks_bev)
        ctx.d, ctx.h, ctx.w = d, h, w
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # Simplified backward — for training use CUDA version
        return None, None, None, None, None, None, None, None, None, None


def bev_pool_v2_func(depth, feat, ranks_depth, ranks_feat, ranks_bev,
                      interval_starts, interval_lengths, d, h, w):
    """Functional API matching CUDA bev_pool_v2 signature."""
    return _BEVPoolv2.apply(
        depth, feat, ranks_depth, ranks_feat, ranks_bev,
        interval_starts, interval_lengths, d, h, w
    )
