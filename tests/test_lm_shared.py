"""Shared-attention structure tests; CPU substitutes do not validate CUDA math."""
from dataclasses import replace
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from dism_v2 import lm_model as lm


def test_shared_parameter_budget():
    old = lm.LMConfig(softcap=30.)
    new = replace(old, architecture='hybrid_shared')
    assert lm.parameter_count(old) == 49_678_876
    assert lm.parameter_count(new) == 45_792_796
    with torch.device('meta'):
        shared, baseline = lm.DecoderLM(new), lm.DecoderLM(replace(old, architecture='swa_only'))
    count = lambda m: sum(p.numel() for p in m.parameters())
    assert count(shared.blocks[0]) - count(baseline.blocks[0]) == 285_476
    assert shared.blocks[0].swa is None
    assert shared.embedding.weight is not shared.lm_head.weight
    assert not any('qkv' in name for name, _ in shared.named_parameters())
    assert lm.matched_swa_config(new).ffn_hidden == 1408
    groups = lm.parameter_groups(shared, .01)
    no_decay = {id(p) for p in groups[1]['params']}
    for name, p in shared.named_parameters():
        assert (id(p) in no_decay) == (p.ndim <= 1 or getattr(p, '_no_weight_decay', False)), name


def test_shared_routing_and_gradients(monkeypatch):
    from dism_v2 import autograd
    from fla.modules import ShortConvolution
    # CPU causal depthwise implementation for connectivity, not CUDA accuracy.
    def conv(self, x):
        out = F.conv1d(x.transpose(1, 2), self.weight, self.bias,
                       padding=3, groups=x.shape[-1])[..., :x.shape[1]]
        return F.silu(out).transpose(1, 2), None
    monkeypatch.setattr(ShortConvolution, 'forward', conv)
    outputs, inputs, calls = {}, {}, {}
    def dism(q, k, v, tau, qvoc, kvoc, **kwargs):
        assert kwargs['hard_prob'] == .37 and kwargs['direction'] == 'random'
        outputs['dism'] = q + 2*k + v + tau[None, :, None, None] + qvoc.mean() + kvoc.mean()
        return outputs['dism']
    def swa(q, k, v, **kwargs):
        assert kwargs == dict(causal=True, window_size=(127, 0), dropout_p=0.)
        outputs['swa'] = q + k + 2*v
        return outputs['swa']
    monkeypatch.setattr(autograd, 'voc_dism', dism)
    monkeypatch.setattr(lm, 'flash_attn_func', swa)
    model = lm.SharedDismAttention(lm.LMConfig(architecture='hybrid_shared', context=32))
    class Norm(nn.Module):
        def forward(self, x, gate):
            outputs['norm_input'] = x
            return x * gate.sigmoid()
    model.norm = Norm()
    for name in ('q_proj', 'k_proj', 'v_proj'):
        def hook(module, args, out, name=name):
            calls[name] = calls.get(name, 0) + 1
        getattr(model, name).register_forward_hook(hook)
    for name in ('qd_conv', 'kd_conv', 'v_dism', 'q_swa_conv', 'k_swa_conv', 'v_swa_conv'):
        def hook(module, args, name=name):
            inputs[name] = args[0]
        getattr(model, name).register_forward_pre_hook(hook)
    x = torch.randn(2, 17, 256, requires_grad=True)
    result = model(x, .37, None)
    assert calls == dict(q_proj=1, k_proj=1, v_proj=1)
    for old, new in [('qd_conv', 'q_swa_conv'), ('kd_conv', 'k_swa_conv'), ('v_dism', 'v_swa_conv')]:
        assert inputs[old] is inputs[new]
    expected = (outputs['dism'].transpose(1, 2) + outputs['swa']).reshape(2, 17, 256)
    torch.testing.assert_close(outputs['norm_input'], expected, atol=0, rtol=0)
    result.square().mean().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum() > 0, name


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA smoke test')
@pytest.mark.parametrize('probability', [0., .37, 1.])
def test_shared_cuda_smoke(probability):
    torch.manual_seed(81)
    model = lm.DecoderLM(lm.LMConfig(architecture='hybrid_shared', layers=1,
                                    context=256, softcap=30.)).cuda()
    x = torch.randint(50257, (1, 129), device='cuda')
    y = torch.randint(50257, x.shape, device='cuda')
    rng = torch.Generator(device='cuda').manual_seed(37)
    before = rng.get_state()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss = model(x, y, probability, rng)
        rng.set_state(before)
        replay = model(x, y, probability, rng)
    torch.testing.assert_close(loss, replay, atol=0, rtol=0)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
