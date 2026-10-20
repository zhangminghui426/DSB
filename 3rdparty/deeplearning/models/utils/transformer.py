"""Transformer building blocks for the TransFusion head.

Standard PyTorch: attention is nn.MultiheadAttention (parameter names
in_proj_weight/in_proj_bias/out_proj.* match the official checkpoint).
The decoder layer keeps the custom (query, key, query_pos, key_pos) forward
signature and posembed slots required by TransFusionHead.
"""

from torch import nn

from models.layers import ConvBN1d

__all__ = ["PositionEmbeddingLearned", "TransformerDecoderLayer", "FFN"]


class PositionEmbeddingLearned(nn.Module):
    """Learned absolute position embedding: (B, P, C) -> (B, num_pos_feats, P)."""

    def __init__(self, input_channel, num_pos_feats=288):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Conv1d(input_channel, num_pos_feats, kernel_size=1),
            nn.BatchNorm1d(num_pos_feats),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_pos_feats, num_pos_feats, kernel_size=1),
        )

    def forward(self, xyz):
        xyz = xyz.transpose(1, 2).contiguous()
        return self.position_embedding_head(xyz)


class TransformerDecoderLayer(nn.Module):
    """Post-norm decoder layer: optional self-attn + cross-attn + linear FFN,
    learned position embeddings added to query/key.

    Parameter names (self_attn/multihead_attn/linear1/linear2/norm1-3) match
    the official checkpoint; keep them.
    """

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        self_posembed=None,
        cross_posembed=None,
        cross_only=False,
    ):
        super().__init__()
        self.cross_only = cross_only
        if not self.cross_only:
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"activation should be relu/gelu, not {activation}")
        self.self_posembed = self_posembed
        self.cross_posembed = cross_posembed

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, query, key, query_pos, key_pos, attn_mask=None):
        """query/key: (B, C, P); query_pos/key_pos: (B, P, 2/3)."""
        # NxCxP -> PxNxC for nn.MultiheadAttention
        query_pos_embed = (
            self.self_posembed(query_pos).permute(2, 0, 1)
            if self.self_posembed is not None
            else None
        )
        key_pos_embed = (
            self.cross_posembed(key_pos).permute(2, 0, 1)
            if self.cross_posembed is not None
            else None
        )
        query = query.permute(2, 0, 1)
        key = key.permute(2, 0, 1)

        if not self.cross_only:
            q = k = v = self.with_pos_embed(query, query_pos_embed)
            query2 = self.self_attn(q, k, value=v)[0]
            query = query + self.dropout1(query2)
            query = self.norm1(query)

        query2 = self.multihead_attn(
            query=self.with_pos_embed(query, query_pos_embed),
            key=self.with_pos_embed(key, key_pos_embed),
            value=self.with_pos_embed(key, key_pos_embed),
            attn_mask=attn_mask,
        )[0]
        query = query + self.dropout2(query2)
        query = self.norm2(query)

        query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
        query = query + self.dropout3(query2)
        query = self.norm3(query)

        return query.permute(1, 2, 0)  # PxNxC -> NxCxP


class FFN(nn.Module):
    """Per-head 1D prediction branches for TransFusionHead.

    heads: {name: (out_channels, num_conv)}; each branch is
    Sequential(ConvBN1d x (num_conv-1), Conv1d). Branches are registered
    under their bare head names — those names ARE state_dict keys
    (e.g. `center.0.conv.weight`); do not wrap them in a named container.
    """

    def __init__(self, in_channels, heads, head_conv=64, final_kernel=1):
        super().__init__()
        self.head_names = list(heads)
        for name, (classes, num_conv) in heads.items():
            layers = []
            c_in = in_channels
            for _ in range(num_conv - 1):
                layers.append(
                    ConvBN1d(c_in, head_conv, final_kernel, padding=final_kernel // 2)
                )
                c_in = head_conv
            layers.append(
                nn.Conv1d(c_in, classes, final_kernel, stride=1,
                          padding=final_kernel // 2, bias=True)
            )
            setattr(self, name, nn.Sequential(*layers))

    def forward(self, x):
        return {name: getattr(self, name)(x) for name in self.head_names}
