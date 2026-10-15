"""Swin Transformer backbone (Swin-T configuration used by BEVFusion).

Parameter names match the official checkpoint exactly (~187 backbone keys):
patch_embed.{projection,norm}, stages.N.blocks.M.{norm1,attn.w_msa,ffn,norm2},
stages.N.downsample.{norm,reduction}, norm{1,2,3}. Keep the structure.
"""

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def to_2tuple(x):
    if isinstance(x, (list, tuple)):
        return tuple(x)
    return (x, x)


# ---------------------------------------------------------------------------
# DropPath (stochastic depth)
# ---------------------------------------------------------------------------

def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Standard stochastic-depth drop_path.

    keep_prob = 1 - drop_prob
    shape = (B,) + (1,) * ndim_rest
    random_tensor ~ Uniform(0,1); mask = floor(random_tensor + keep_prob)
    output = x / keep_prob * mask   (so E[output] == x when training)
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
    mask = torch.floor(random_tensor + keep_prob)
    return x / keep_prob * mask


class DropPath(nn.Module):
    """Stochastic depth per-sample drop path. drop_prob=0 → identity."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self):
        return f"drop_prob={self.drop_prob:.3f}"


# ---------------------------------------------------------------------------
# FFN — nested-Sequential key layout matches the official checkpoint
# ---------------------------------------------------------------------------

class FFN(nn.Module):
    """Feed-forward network used in Swin blocks.

    The nested Sequential layout is load-bearing for checkpoint keys:
    layers.0.0.{weight,bias} (first Linear) and layers.1.{weight,bias}
    (second Linear) — do not flatten.
    """

    def __init__(self, embed_dims, feedforward_channels, ffn_drop=0.0, drop_path=0.0):
        super().__init__()
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.layers = nn.Sequential(
            nn.Sequential(
                nn.Linear(embed_dims, feedforward_channels),
                nn.Dropout(ffn_drop),
                nn.GELU(),
            ),
            nn.Linear(feedforward_channels, embed_dims),
            nn.Dropout(ffn_drop),
        )
        self.dropout_layer = DropPath(drop_path)

    def forward(self, x, identity=None):
        out = self.layers(x)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)


# ---------------------------------------------------------------------------
# PatchEmbed — image → patch embedding via Conv2d
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """Image to Patch Embedding.

    Uses a Conv2d (kernel=patch_size, stride=patch_size) to project the
    image into patch tokens, followed by an optional norm layer.

    Output: ``(B, H*W, embed_dims)`` and ``hw_shape=(H, W)``.
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dims: int = 768,
        kernel_size: int = 16,
        stride: int | None = None,
        padding: str = "corner",
        dilation: int = 1,
        bias: bool = True,
        norm: bool = True,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        if stride is None:
            stride = kernel_size

        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)

        if isinstance(padding, str):
            # AdaptivePadding — "corner" pads bottom-right; "same" pads evenly.
            self.adap_padding = AdaptivePadding(
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
            )
            padding_val = 0
        else:
            self.adap_padding = None
            padding_val = padding

        padding_val = to_2tuple(padding_val)

        self.projection = nn.Conv2d(
            in_channels, embed_dims, kernel_size,
            stride=stride, padding=padding_val, dilation=dilation, bias=bias,
        )
        self.norm = nn.LayerNorm(embed_dims) if norm else None

    def forward(self, x):
        if self.adap_padding is not None:
            x = self.adap_padding(x)
        x = self.projection(x)
        out_size = (x.shape[2], x.shape[3])
        x = x.flatten(2).transpose(1, 2)  # B, H*W, C
        if self.norm is not None:
            x = self.norm(x)
        return x, out_size


class AdaptivePadding(nn.Module):
    """Pads input so it is evenly divisible by stride (corner or same mode)."""

    def __init__(self, kernel_size=1, stride=1, dilation=1, padding="corner"):
        super().__init__()
        assert padding in ("same", "corner")
        self.kernel_size = to_2tuple(kernel_size)
        self.stride = to_2tuple(stride)
        self.dilation = to_2tuple(dilation)
        self.padding = padding

    def get_pad_shape(self, input_shape):
        input_h, input_w = input_shape
        kh, kw = self.kernel_size
        sh, sw = self.stride
        dh, dw = self.dilation
        output_h = math.ceil(input_h / sh)
        output_w = math.ceil(input_w / sw)
        pad_h = max((output_h - 1) * sh + (kh - 1) * dh + 1 - input_h, 0)
        pad_w = max((output_w - 1) * sw + (kw - 1) * dw + 1 - input_w, 0)
        return pad_h, pad_w

    def forward(self, x):
        pad_h, pad_w = self.get_pad_shape(x.size()[-2:])
        if pad_h > 0 or pad_w > 0:
            if self.padding == "corner":
                x = F.pad(x, [0, pad_w, 0, pad_h])
            else:  # same
                x = F.pad(
                    x,
                    [
                        pad_w // 2,
                        pad_w - pad_w // 2,
                        pad_h // 2,
                        pad_h - pad_h // 2,
                    ],
                )
        return x


# ---------------------------------------------------------------------------
# PatchMerging — 2× downsampling between stages
# ---------------------------------------------------------------------------

class PatchMerging(nn.Module):
    """Merge 2×2 neighbouring patches via Unfold → Norm → Linear.

    Using ``nn.Unfold`` is ~25 % faster than the original Swin
    implementation but requires the ``correct_unfold_*_order`` weight
    conversion when loading the original Microsoft checkpoint.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 2,
        stride: int | None = None,
        padding: str = "corner",
        dilation: int = 1,
        bias: bool = False,
        norm: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if stride is None:
            stride = kernel_size

        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)

        if isinstance(padding, str):
            self.adap_padding = AdaptivePadding(
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
            )
            padding_val = 0
        else:
            self.adap_padding = None
            padding_val = padding

        padding_val = to_2tuple(padding_val)
        self.sampler = nn.Unfold(
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding_val,
            stride=stride,
        )

        sample_dim = kernel_size[0] * kernel_size[1] * in_channels

        self.norm = nn.LayerNorm(sample_dim) if norm else None

        self.reduction = nn.Linear(sample_dim, out_channels, bias=bias)

    def forward(self, x, input_size):
        """
        Args:
            x: (B, H*W, C_in)
            input_size: (H, W)
        Returns:
            x: (B, H'*W', C_out), output_size: (H', W')
        """
        B, L, C = x.shape
        H, W = input_size
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C).permute(0, 3, 1, 2)  # B, C, H, W

        if self.adap_padding is not None:
            x = self.adap_padding(x)
            H, W = x.shape[-2:]

        x = self.sampler(x)  # B, 4*C, H/2*W/2

        out_h = (
            H
            + 2 * self.sampler.padding[0]
            - self.sampler.dilation[0] * (self.sampler.kernel_size[0] - 1)
            - 1
        ) // self.sampler.stride[0] + 1
        out_w = (
            W
            + 2 * self.sampler.padding[1]
            - self.sampler.dilation[1] * (self.sampler.kernel_size[1] - 1)
            - 1
        ) // self.sampler.stride[1] + 1

        output_size = (out_h, out_w)
        x = x.transpose(1, 2)  # B, H/2*W/2, 4*C
        if self.norm is not None:
            x = self.norm(x)
        x = self.reduction(x)
        return x, output_size


# ---------------------------------------------------------------------------
# WindowMSA — window multi-head self-attention with relative position bias
# ---------------------------------------------------------------------------

class WindowMSA(nn.Module):
    """Window-based multi-head self-attention with relative position bias."""

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        window_size: tuple[int, int],
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        attn_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.window_size = window_size  # (Wh, Ww)
        self.num_heads = num_heads
        head_embed_dims = embed_dims // num_heads
        self.scale = qk_scale or head_embed_dims ** -0.5

        # Relative position bias table: (2*Wh-1)*(2*Ww-1), nH
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * window_size[0] - 1) * (2 * window_size[1] - 1),
                num_heads,
            )
        )

        # Compute relative position index (buffer — no gradient)
        Wh, Ww = self.window_size
        rel_index_coords = self.double_step_seq(2 * Ww - 1, Wh, 1, Ww)
        rel_position_index = rel_index_coords + rel_index_coords.T
        rel_position_index = rel_position_index.flip(1).contiguous()
        self.register_buffer("relative_position_index", rel_position_index)

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_rate)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: (num_windows*B, N, C)
            mask: (num_windows, N, N) or None
        """
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )  # Wh*Ww, Wh*Ww, nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        ).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @staticmethod
    def double_step_seq(step1, len1, step2, len2):
        seq1 = torch.arange(0, step1 * len1, step1)
        seq2 = torch.arange(0, step2 * len2, step2)
        return (seq1[:, None] + seq2[None, :]).reshape(1, -1)


# ---------------------------------------------------------------------------
# ShiftWindowMSA — shifted window multi-head self-attention
# ---------------------------------------------------------------------------

class ShiftWindowMSA(nn.Module):
    """Shifted Window Multihead Self-Attention.

    Wraps ``WindowMSA`` with optional cyclic shift and attention masking.
    """

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        window_size: int,
        shift_size: int = 0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        attn_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        assert 0 <= self.shift_size < self.window_size

        self.w_msa = WindowMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=to_2tuple(window_size),
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=proj_drop_rate,
        )

        self.drop = DropPath(drop_path)

    def forward(self, query, hw_shape):
        B, L, C = query.shape
        H, W = hw_shape
        assert L == H * W, "input feature has wrong size"
        query = query.view(B, H, W, C)

        # Pad feature maps to multiples of window size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        query = F.pad(query, (0, 0, 0, pad_r, 0, pad_b))
        H_pad, W_pad = query.shape[1], query.shape[2]

        # Cyclic shift
        if self.shift_size > 0:
            shifted_query = torch.roll(
                query,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2),
            )

            # Calculate attention mask for SW-MSA
            img_mask = torch.zeros((1, H_pad, W_pad, 1), device=query.device)
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            # nW, window_size, window_size, 1
            mask_windows = self.window_partition(img_mask)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(
                attn_mask != 0, float(-100.0)
            ).masked_fill(attn_mask == 0, float(0.0))
        else:
            shifted_query = query
            attn_mask = None

        # nW*B, window_size, window_size, C
        query_windows = self.window_partition(shifted_query)
        # nW*B, window_size*window_size, C
        query_windows = query_windows.view(-1, self.window_size ** 2, C)

        # W-MSA / SW-MSA
        attn_windows = self.w_msa(query_windows, mask=attn_mask)

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        # B H' W' C
        shifted_x = self.window_reverse(attn_windows, H_pad, W_pad)
        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2),
            )
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)
        x = self.drop(x)
        return x

    def window_reverse(self, windows, H, W):
        """(num_windows*B, ws, ws, C) → (B, H, W, C)"""
        window_size = self.window_size
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(
            B, H // window_size, W // window_size, window_size, window_size, -1
        )
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def window_partition(self, x):
        """(B, H, W, C) → (num_windows*B, ws, ws, C)"""
        B, H, W, C = x.shape
        window_size = self.window_size
        x = x.view(
            B, H // window_size, window_size, W // window_size, window_size, C
        )
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, window_size, window_size, C)
        return windows


# ---------------------------------------------------------------------------
# SwinBlock
# ---------------------------------------------------------------------------

class SwinBlock(nn.Module):
    """A single Swin Transformer block: Norm1 → Attn → Norm2 → FFN."""

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        feedforward_channels: int,
        window_size: int = 7,
        shift: bool = False,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        with_cp: bool = False,
    ):
        super().__init__()
        self.with_cp = with_cp

        self.norm1 = nn.LayerNorm(embed_dims)
        self.attn = ShiftWindowMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=window_size // 2 if shift else 0,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=drop_rate,
            drop_path=drop_path_rate,
        )

        self.norm2 = nn.LayerNorm(embed_dims)
        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            ffn_drop=drop_rate,
            drop_path=drop_path_rate,
        )

    def forward(self, x, hw_shape):
        def inner_forward(x):
            identity = x
            x = self.norm1(x)
            x = self.attn(x, hw_shape)
            x = x + identity

            identity = x
            x = self.norm2(x)
            x = self.ffn(x, identity=identity)
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(inner_forward, x, use_reentrant=False)
        else:
            x = inner_forward(x)

        return x


# ---------------------------------------------------------------------------
# SwinBlockSequence — one Swin stage
# ---------------------------------------------------------------------------

class SwinBlockSequence(nn.Module):
    """One Swin stage: sequence of blocks + optional downsample."""

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        feedforward_channels: int,
        depth: int,
        window_size: int = 7,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float | list[float] = 0.0,
        downsample: nn.Module | None = None,
        with_cp: bool = False,
    ):
        super().__init__()

        if isinstance(drop_path_rate, list):
            drop_path_rates = drop_path_rate
            assert len(drop_path_rates) == depth
        else:
            drop_path_rates = [drop_path_rate] * depth

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = SwinBlock(
                embed_dims=embed_dims,
                num_heads=num_heads,
                feedforward_channels=feedforward_channels,
                window_size=window_size,
                shift=False if i % 2 == 0 else True,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rates[i],
                with_cp=with_cp,
            )
            self.blocks.append(block)

        self.downsample = downsample

    def forward(self, x, hw_shape):
        for block in self.blocks:
            x = block(x, hw_shape)

        if self.downsample is not None:
            x_down, down_hw_shape = self.downsample(x, hw_shape)
            return x_down, down_hw_shape, x, hw_shape
        else:
            return x, hw_shape, x, hw_shape


# ---------------------------------------------------------------------------
# SwinTransformer — main class
# ---------------------------------------------------------------------------

class SwinTransformer(nn.Module):
    """Swin Transformer backbone.

    out_indices selects which stages are output (each through its own
    norm{i} LayerNorm); with_cp enables per-block gradient checkpointing
    (8GB-GPU memory saving). init_weights() does the scratch init; official
    BEVFusion checkpoints are loaded by the trainer.
    """

    def __init__(
        self,
        pretrain_img_size: int = 224,
        in_channels: int = 3,
        embed_dims: int = 96,
        patch_size: int = 4,
        window_size: int = 7,
        mlp_ratio: int = 4,
        depths: tuple[int, ...] = (2, 2, 6, 2),
        num_heads: tuple[int, ...] = (3, 6, 12, 24),
        strides: tuple[int, ...] = (4, 2, 2, 2),
        out_indices: tuple[int, ...] = (0, 1, 2, 3),
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        patch_norm: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        with_cp: bool = False,
        frozen_stages: int = -1,
    ):
        super().__init__()
        self.frozen_stages = frozen_stages

        num_layers = len(depths)
        self.out_indices = out_indices

        assert strides[0] == patch_size, "Use non-overlapping patch embed."

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dims=embed_dims,
            kernel_size=patch_size,
            stride=strides[0],
            norm=patch_norm,
        )

        self.drop_after_pos = nn.Dropout(p=drop_rate)

        # Stochastic depth decay rule
        total_depth = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]

        self.stages = nn.ModuleList()
        in_channels = embed_dims
        for i in range(num_layers):
            if i < num_layers - 1:
                downsample = PatchMerging(
                    in_channels=in_channels,
                    out_channels=2 * in_channels,
                    stride=strides[i + 1],
                    norm=patch_norm,
                )
            else:
                downsample = None

            stage = SwinBlockSequence(
                embed_dims=in_channels,
                num_heads=num_heads[i],
                feedforward_channels=mlp_ratio * in_channels,
                depth=depths[i],
                window_size=window_size,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                downsample=downsample,
                with_cp=with_cp,
            )
            self.stages.append(stage)
            if downsample is not None:
                in_channels = downsample.out_channels

        self.num_features = [int(embed_dims * 2 ** i) for i in range(num_layers)]
        # Add a norm layer for each output
        for i in out_indices:
            layer = nn.LayerNorm(self.num_features[i])
            layer_name = f"norm{i}"
            self.add_module(layer_name, layer)

    # -----------------------------------------------------------------------
    # Train / freeze
    # -----------------------------------------------------------------------

    def train(self, mode=True):
        super().train(mode)
        self.freeze_stages()

    def freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            self.drop_after_pos.eval()

        for i in range(1, self.frozen_stages + 1):
            if (i - 1) in self.out_indices:
                norm_layer = getattr(self, f"norm{i-1}")
                norm_layer.eval()
                for param in norm_layer.parameters():
                    param.requires_grad = False
            m = self.stages[i - 1]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    # -----------------------------------------------------------------------
    # Weight initialization
    # -----------------------------------------------------------------------

    def init_weights(self):
        """Scratch init — official BEVFusion checkpoints are loaded by the
        trainer separately. trunc_normal(0.02) on Linear weights (bias 0),
        constant(1, 0) on LayerNorm. relative_position_bias_table keeps its
        zero init (matches this repo's historical from-scratch behavior).
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    # -----------------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------------

    def forward(self, x):
        x, hw_shape = self.patch_embed(x)
        x = self.drop_after_pos(x)

        outs = []
        for i, stage in enumerate(self.stages):
            x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            if i in self.out_indices:
                norm_layer = getattr(self, f"norm{i}")
                out = norm_layer(out)
                out = out.view(-1, *out_hw_shape, self.num_features[i]).permute(
                    0, 3, 1, 2
                ).contiguous()
                outs.append(out)

        return tuple(outs)
