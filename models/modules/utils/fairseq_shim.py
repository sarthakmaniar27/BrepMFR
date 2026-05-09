"""Minimal fairseq replacements for PyTorch 2.7+ (avoids unmaintained fairseq wheels)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def quant_noise(module: nn.Module, p: float = 0.0, qn_block_size: int = 8) -> nn.Module:
    return module


class FairseqDropout(nn.Dropout):
    def __init__(self, p: float, module_name: str = "", broadcast_dim: int = 0):
        super().__init__(p, inplace=False)


class LayerNorm(nn.LayerNorm):
    def __init__(
        self,
        normalized_shape,
        export: bool = False,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ):
        super().__init__(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)


def get_activation_fn(name: str):
    if name == "gelu":
        return F.gelu
    if name == "relu":
        return F.relu
    if name == "tanh":
        return torch.tanh
    raise RuntimeError(f"Unknown activation_fn: {name}")


def softmax(x, dim: int, onnx_trace: bool = False):
    return F.softmax(x, dim=dim, dtype=torch.float32)


class LayerDropModuleList(nn.ModuleList):
    def __init__(self, p: float, modules=None):
        if modules is None:
            modules = []
        super().__init__(modules)
        self.p = float(p)

    def __iter__(self):
        if not self.training or self.p <= 0.0:
            yield from super().__iter__()
            return
        probs = torch.empty(len(self)).uniform_()
        for i, m in enumerate(super().__iter__()):
            if probs[i].item() > self.p:
                yield m
