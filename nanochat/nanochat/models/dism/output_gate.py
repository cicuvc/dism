"""Explicit, checkpoint-described mixer output variant from the short screen."""
import torch
from torch import nn
from torch.nn import functional as F


class SigmoidInputLinear(nn.Linear):
    """Keep original parameter names, identities and initialization draw order."""
    def __init__(self, original):
        nn.Module.__init__(self)
        self.in_features = original.in_features
        self.out_features = original.out_features
        self.weight = original.weight
        self.bias = original.bias

    def forward(self, x):
        return F.linear(x.sigmoid(), self.weight, self.bias)


class LayerNormRawGate(nn.Module):
    explicit_output_gate = True

    def __init__(self, original, scale):
        super().__init__()
        self.weight = original.weight
        self.eps = original.eps
        self.scale = float(scale)

    def forward(self, x, gate):
        centered = x.float() - x.float().mean(-1, keepdim=True)
        normalized = centered * torch.rsqrt(centered.square().mean(-1, keepdim=True) + self.eps)
        return (normalized * self.weight.float() * (gate.float() * self.scale)).to(x.dtype)


def install_layernorm_sigmoid(model, scales):
    for layer, scale in zip(model.layers, scales, strict=True):
        attn = layer.attn
        attn.g_proj_up = SigmoidInputLinear(attn.g_proj_up)
        attn.norm = LayerNormRawGate(attn.norm, scale)
