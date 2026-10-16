"""Image backbone: torchvision ResNet-50 feature extractor.

state_dict naming matches mmdet.ResNet(style='pytorch') used by the official
FlashOCC checkpoints: conv1/bn1, layer{1..4}.{block}.{conv1,bn1,conv2,bn2,
conv3,bn3}, downsample.0/.1. fc is replaced by Identity so no fc.* keys
appear.
"""

import logging

import torch.utils.checkpoint as cp
import torchvision
from torch import nn

logger = logging.getLogger(__name__)

__all__ = ["ResNetBackbone"]


class ResNetBackbone(torchvision.models.ResNet):
    """ResNet-50 returning multi-scale features at out_indices (0-based
    over layer1..layer4). with_cp enables per-stage gradient checkpointing
    (8GB-GPU memory saving).
    """

    def __init__(self, out_indices=(2, 3), with_cp=False, pretrained=False):
        super().__init__(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3])
        self.out_indices = tuple(out_indices)
        self.with_cp = with_cp
        self.fc = nn.Identity()  # key-compatible drop of the classifier
        if pretrained:
            self.load_torchvision_pretrained()

    def load_torchvision_pretrained(self):
        """ImageNet pretrain for from-scratch training; offline -> scratch."""
        try:
            state = torchvision.models.resnet50(
                weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1
            ).state_dict()
        except Exception as e:  # offline / no cache
            logger.warning(
                "torchvision pretrained unavailable (%s); init from scratch", e
            )
            return
        state.pop("fc.weight", None)
        state.pop("fc.bias", None)
        msg = self.load_state_dict(state, strict=False)
        logger.info(
            "loaded torchvision resnet50 pretrain: missing=%d unexpected=%d",
            len(msg.missing_keys), len(msg.unexpected_keys),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer in enumerate((self.layer1, self.layer2, self.layer3, self.layer4)):
            if self.with_cp and self.training:
                x = cp.checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)
