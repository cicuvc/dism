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
    center_output = True

    def __init__(self, original, scale):
        super().__init__()
        self.weight = original.weight
        self.eps = original.eps
        self.scale = float(scale)
        # Runtime-only switch for diagnostics; no checkpoint/config state.
        self.enable_fused_output = True

    def fused_output(self, raw, x, down, up):
        if self.enable_fused_output and not torch.compiler.is_compiling():
            from flash_dism.kernels.mixer_output import fused_mixer_output, supported
            if supported(x, raw, down.weight, down.bias, up.weight, up.bias, self.weight):
                return fused_mixer_output(x, raw, down.weight, down.bias,
                                          up.weight, up.bias, self.weight,
                                          self.scale, self.eps, self.center_output)
        gate = up(down(x)).unflatten(-1, raw.shape[-2:])
        return self(raw, gate)

    def forward(self, x, gate):
        centered = x.float() - x.float().mean(-1, keepdim=True)
        normalized = centered * torch.rsqrt(centered.square().mean(-1, keepdim=True) + self.eps)
        return (normalized * self.weight.float() * (gate.float() * self.scale)).to(x.dtype)


class RMSNormRawGate(LayerNormRawGate):
    center_output = False

    def forward(self, x, gate):
        value = x.float()
        normalized = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.eps)
        return (normalized * self.weight.float() * (gate.float() * self.scale)).to(x.dtype)


def install_layernorm_sigmoid(model, scales):
    for layer, scale in zip(model.layers, scales, strict=True):
        attn = layer.attn
        attn.g_proj_up = SigmoidInputLinear(attn.g_proj_up)
        norm_type = LayerNormRawGate if model.config.mixer_output == 'layernorm_sigmoid' else RMSNormRawGate
        attn.norm = norm_type(attn.norm, scale)
