"""Sparse lidar backbone (voxelnet-style sparse conv encoder, spconv 2.x).

Naming is load-bearing for the official checkpoints: conv_input/conv_out are
SparseSequential (.0 conv, .1 bn); stages register as encoder_layer{1..4};
SparseBasicBlock uses conv1/bn1/conv2/bn2.

spconv 2.x stores conv weights as (Cout, K, D, H, Cin) while the spconv-1.x
checkpoints use (K, D, H, Cin, Cout) — run tools/remap_pretrained.py first.

Coordinate convention (this fork): voxelization emits (x, y, z) cells and
sparse_shape is [X, Y, Z] (e.g. [1440, 1440, 41]); dense() returns
(N, C, X, Y, Z) and permute(0,1,4,2,3)+view folds Z into channels, yielding
BEV features ordered (N, C*Z, X, Y) — matching the camera branch's BEV grid.
"""

import spconv.pytorch as spconv
from torch import nn

__all__ = ["SparseEncoder", "SparseBasicBlock", "make_sparse_convmodule"]


def make_sparse_convmodule(
    in_channels,
    out_channels,
    kernel_size,
    indice_key,
    stride=1,
    padding=0,
    conv_type="SubMConv3d",
    bn_eps=1e-3,
    bn_momentum=0.01,
):
    """SparseSequential(conv, bn, relu) — the .0/.1 slots are ckpt keys."""
    conv_cls = getattr(spconv, conv_type)
    if conv_type in ("SparseInverseConv3d", "SparseInverseConv2d", "SparseInverseConv1d"):
        conv = conv_cls(
            in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key
        )
    else:
        conv = conv_cls(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=False, indice_key=indice_key,
        )
    return spconv.SparseSequential(
        conv,
        nn.BatchNorm1d(out_channels, eps=bn_eps, momentum=bn_momentum),
        nn.ReLU(inplace=True),
    )


class SparseBasicBlock(spconv.SparseModule):
    """Residual block with submanifold sparse convs.

    Attribute names conv1/bn1/conv2/bn2 match the official checkpoint keys.
    """

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 bn_eps=1e-3, bn_momentum=0.01):
        super().__init__()
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, 3, stride=stride, padding=1, bias=False, indice_key=None
        )
        self.bn1 = nn.BatchNorm1d(planes, eps=bn_eps, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = spconv.SubMConv3d(
            planes, planes, 3, stride=stride, padding=1, bias=False, indice_key=None
        )
        self.bn2 = nn.BatchNorm1d(planes, eps=bn_eps, momentum=bn_momentum)
        self.downsample = downsample

    def forward(self, x):
        identity = x.features
        assert x.features.dim() == 2, f"x.features.dim()={x.features.dim()}"

        out = self.conv1(x)
        out = out.replace_feature(self.bn1(out.features))
        out = out.replace_feature(self.relu(out.features))

        out = self.conv2(out)
        out = out.replace_feature(self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out.replace_feature(out.features + identity)
        out = out.replace_feature(self.relu(out.features))
        return out


class SparseEncoder(nn.Module):
    """Voxels -> conv_input -> 4 encoder stages -> conv_out -> BEV features.

    Args:
        in_channels: input feature dims (5: x,y,z,intensity,dt).
        sparse_shape: [X, Y, Z] voxel grid.
        base_channels / output_channels: conv_input / conv_out channels.
        encoder_channels: per-stage channel tuples.
        encoder_paddings: per-stage conv paddings.
        block_type: 'basicblock' (residual) or 'conv_module' (plain).
    """

    def __init__(
        self,
        in_channels,
        sparse_shape,
        base_channels=16,
        output_channels=128,
        encoder_channels=((16,), (32, 32, 32), (64, 64, 64), (64, 64, 64)),
        encoder_paddings=((1,), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1, 1)),
        block_type="conv_module",
        bn_eps=1e-3,
        bn_momentum=0.01,
    ):
        super().__init__()
        assert block_type in ("conv_module", "basicblock")
        self.sparse_shape = sparse_shape
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.output_channels = output_channels
        self.encoder_channels = encoder_channels
        self.encoder_paddings = encoder_paddings
        self.stage_num = len(self.encoder_channels)
        self.bn_eps = bn_eps
        self.bn_momentum = bn_momentum
        # spconv initializes all weights on its own

        self.conv_input = make_sparse_convmodule(
            in_channels, base_channels, 3, padding=1, indice_key="subm1",
            conv_type="SubMConv3d", bn_eps=bn_eps, bn_momentum=bn_momentum,
        )
        encoder_out_channels = self.make_encoder_layers(base_channels, block_type)
        self.conv_out = make_sparse_convmodule(
            encoder_out_channels, output_channels, kernel_size=(1, 1, 3),
            stride=(1, 1, 2), padding=0, indice_key="spconv_down2",
            conv_type="SparseConv3d", bn_eps=bn_eps, bn_momentum=bn_momentum,
        )

    def forward(self, voxel_features, coors, batch_size, **kwargs):
        """(N, C) float feats + (N, 4) int coords (batch,x,y,z) -> (N, C*Z, X', Y')."""
        coors = coors.int()
        input_sp_tensor = spconv.SparseConvTensor(
            voxel_features, coors, self.sparse_shape, batch_size
        )
        x = self.conv_input(input_sp_tensor)

        encode_features = []
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
            encode_features.append(x)

        # fold Z into channels for the detection head
        out = self.conv_out(encode_features[-1])
        spatial_features = out.dense()
        N, C, H, W, D = spatial_features.shape
        spatial_features = spatial_features.permute(0, 1, 4, 2, 3).contiguous()
        return spatial_features.view(N, C * D, H, W)

    def make_encoder_layers(self, in_channels, block_type="conv_module"):
        """Build the 4 stages as encoder_layers; returns final out channels."""
        assert block_type in ("conv_module", "basicblock")
        self.encoder_layers = spconv.SparseSequential()
        bn_kwargs = dict(bn_eps=self.bn_eps, bn_momentum=self.bn_momentum)

        for i, blocks in enumerate(self.encoder_channels):
            blocks_list = []
            for j, out_channels in enumerate(tuple(blocks)):
                padding = tuple(self.encoder_paddings[i])[j]
                # each stage starts with a strided spconv layer, except the first
                if i != 0 and j == 0 and block_type == "conv_module":
                    blocks_list.append(
                        make_sparse_convmodule(
                            in_channels, out_channels, 3, stride=2, padding=padding,
                            indice_key=f"spconv{i + 1}", conv_type="SparseConv3d",
                            **bn_kwargs,
                        )
                    )
                elif block_type == "basicblock":
                    if j == len(blocks) - 1 and i != len(self.encoder_channels) - 1:
                        blocks_list.append(
                            make_sparse_convmodule(
                                in_channels, out_channels, 3, stride=2, padding=padding,
                                indice_key=f"spconv{i + 1}", conv_type="SparseConv3d",
                                **bn_kwargs,
                            )
                        )
                    else:
                        blocks_list.append(
                            SparseBasicBlock(out_channels, out_channels, **bn_kwargs)
                        )
                else:
                    blocks_list.append(
                        make_sparse_convmodule(
                            in_channels, out_channels, 3, padding=padding,
                            indice_key=f"subm{i + 1}", conv_type="SubMConv3d",
                            **bn_kwargs,
                        )
                    )
                in_channels = out_channels
            stage_name = f"encoder_layer{i + 1}"
            self.encoder_layers.add_module(
                stage_name, spconv.SparseSequential(*blocks_list)
            )
        return out_channels
