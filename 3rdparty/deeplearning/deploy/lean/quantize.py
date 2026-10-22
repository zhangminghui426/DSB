"""Quantization stubs for the CUDA-BEVFusion ONNX exporter (lean).

The reference implementation (NVIDIA qat/lean) monkey-patches
SparseConvolutionQunat/QuantAdd modules and TensorQuantizer knobs during
export. Our checkpoints are NOT PTQ-quantized, so this module only keeps the
class names / context-manager interfaces alive for `lean.exptool` and
`lean.funcs`; the quantization paths are never entered.
"""

from torch import nn


class QuantAdd(nn.Module):
    """No-op placeholder (reference PTQ residual-add wrapper)."""

    def forward(self, a, b):
        return a + b


class SparseConvolutionQunat:
    """No-op placeholder (reference PTQ sparse-conv wrapper)."""

    def forward(self, *args):
        raise NotImplementedError("PTQ quantized export is not supported")


class _NoopQuantization:
    def __init__(self, model):
        self.model = model

    def apply(self, disabled=True):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        return False


class disable_quantization(_NoopQuantization):
    pass


class enable_quantization(_NoopQuantization):
    pass
