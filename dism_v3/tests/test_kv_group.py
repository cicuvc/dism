"""kv_group_size: group heads so every kv_group_size queries share one kv head."""
import json
from pathlib import Path
import types

import pytest
import torch

from flash_dism.module import DismAttention

ROOT = Path(__file__).resolve().parents[2]
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA driver unavailable')


def build(**overrides):
    from nanochat.models.dism import DismConfig
    from nanochat.models.dism.model import DismLM
    config = json.loads((ROOT / 'nanochat/configs/sequence_true_72m_valres.json').read_text())['config']
    config.update(overrides)
    torch.manual_seed(1729)
    model = DismLM(DismConfig(**config), attention_type='hybrid_gdn')
    model.init_weights()
    return model


def test_group1_is_bitwise_the_default():
    one, default = build(kv_group_size=1), build()
    a, b = one.state_dict(), default.state_dict()
    assert a.keys() == b.keys()
    assert all(a[k].shape == b[k].shape and torch.equal(a[k], b[k]) for k in a)
    assert sum(p.numel() for p in one.parameters()) == 72035808
    attn = one.layers[6].attn
    assert attn.kv_heads == attn.heads == 6
    assert attn.k_conv.weight is attn.gdn_k_conv.weight
    assert attn.q_conv.weight is attn.gdn_q_conv.weight
    print('kv_group_size=1: state_dict bitwise identical to default, total=72035808')


def test_group2_shapes_and_parameter_budget():
    model = build(kv_group_size=2)
    assert sum(p.numel() for p in model.parameters()) == 71215584
    # 6 layers x (-98304 k_vocab - 1536 k/v depthwise - 36864 sk_proj).
    assert 72035808 - 71215584 == 6 * 136704
    attn = model.layers[6].attn
    assert (attn.heads, attn.kv_heads) == (6, 3)
    assert attn.k_conv.weight.shape == (192, 384)
    assert attn.v_conv.weight.shape == (192, 384)
    assert attn.sk_proj.weight.shape == (96, 384)
    assert attn.q_conv.weight.shape == (384, 384)
    assert attn.sq_proj.weight.shape == (192, 384)
    assert tuple(attn.q_vocab.shape) == (6, 512, 64)
    assert tuple(attn.k_vocab.shape) == (3, 512, 64)
    # The key linear can no longer alias GDN's full-width key projection; the
    # query tie and GDN's own widths are untouched.
    assert attn.q_conv.weight is attn.gdn_q_conv.weight
    assert attn.k_conv.weight is not attn.gdn_k_conv.weight
    assert attn.gdn_k_conv.weight.shape == (384, 384)
    assert attn.gdn_v_conv.weight.shape == (384, 384)
    # Every other pure-GDN layer keeps its own full-width branch.
    assert getattr(model.layers[0].attn, 'kv_heads', 0) == 0
    print('kv_group_size=2: 3 kv heads / 6 query heads, total=71215584 (-820224)')


@pytest.mark.parametrize('bad', [0, 3, True, 2.0])
def test_invalid_group_size(bad):
    with pytest.raises(ValueError, match='kv_group_size'):
        DismAttention(64, 2, kv_group_size=bad)


def test_mutually_exclusive_with_query_channels():
    with pytest.raises(ValueError, match='mutually exclusive'):
        DismAttention(64, 2, gqa_query_channels=2, kv_group_size=2)


def test_head_count_must_divide():
    with pytest.raises(ValueError, match='divisible'):
        DismAttention(64, 3, kv_group_size=2)


@pytest.mark.parametrize('axis,shape', [(0, (2,)), (0, (2, 64, 32)), (2, (2, 256, 2, 32))])
def test_expansion_factor_gradient(axis, shape):
    x = torch.randn(shape, requires_grad=True)
    out = DismAttention._expand_gqa(x, axis, 3)
    assert out.is_contiguous()
    assert out.shape[axis] == 3 * shape[axis]
    for g in range(3):
        assert torch.equal(out.select(axis, g), x.select(axis, 0))
    out.sum().backward()
    assert torch.equal(x.grad, torch.full_like(x, 3))


@CUDA
@pytest.mark.parametrize('fixed_true', [False, True])
@pytest.mark.parametrize('value_residual', [False, True])
def test_pair_share_invariant_and_backward(fixed_true, value_residual, monkeypatch):
    """Exactly one kv head is read by each group of query heads."""
    import flash_dism.compiler as compiler
    from flash_dism.hybrid_gdn import DismGdnAttention
    import flash_dism
    m = DismGdnAttention(64, 4, head_dim=32, value_dim=32, readout_dim=16,
                         vocab_size=64, value_residual=value_residual,
                         kv_group_size=2).cuda()
    captured = {}
    target, name = (compiler, 'voc_forward') if fixed_true else (flash_dism, 'voc_dism')
    original = getattr(target, name)

    def record(*args, **kwargs):
        q, k, sq, sk, v, eq, ek, tau = args[:8]
        # Slots [0,0,1,1]: the kv side repeats in groups, the query side does not.
        # value_residual mixes in GDN's per-head v_first and gate *above* the
        # shared DISM value, so only the un-mixed path keeps v paired per group.
        for shared in ((k, sk, ek) if value_residual else (k, sk, v, ek)):
            axis = 2 if shared.ndim == 4 else 0
            assert torch.equal(shared.select(axis, 0), shared.select(axis, 1))
            assert torch.equal(shared.select(axis, 2), shared.select(axis, 3))
            assert not torch.equal(shared.select(axis, 0), shared.select(axis, 2))
        assert not torch.equal(q[:, :, 0], q[:, :, 1])
        assert not torch.equal(sq[:, :, 0], sq[:, :, 1])
        assert q.shape[2] == k.shape[2] == 4
        assert eq.shape[0] == 4 and ek.shape[0] == 4
        assert tau.shape == (4,)
        result = original(*args, **kwargs)
        captured['raw'] = result[0] if fixed_true else result
        return result

    combine = m._combine_cuda

    def check_combine(output, x, v, *args, **kwargs):
        assert v.shape == (2, 256, 4, 32)
        assert torch.equal(output, captured['raw'])
        return combine(output, x, v, *args, **kwargs)

    monkeypatch.setattr(target, name, record)
    monkeypatch.setattr(m, '_combine_cuda', check_combine)
    x = torch.randn(2, 256, 64, device='cuda', requires_grad=True)
    v_first = torch.randn(2, 256, 4, 32, device='cuda', dtype=torch.bfloat16)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        out = m(x, hard_prob=0., direction=torch.ones(2, 4, device='cuda', dtype=torch.bool),
                direction_true=fixed_true, v_first=v_first)[0]
    out.float().square().mean().backward()
    assert out.shape == x.shape and torch.isfinite(out).all()
    for name, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    assert m.k_conv.weight.grad.abs().sum() > 0
    assert m.gdn_k_conv.weight.grad.abs().sum() > 0
    assert m.k_vocab.grad.abs().sum() > 0
    print('G=2 invariant: [k,sk,v,ek] paired per group, q/sq independent, 4 head slots')


@CUDA
def test_group2_full_model_forward_backward():
    model = build(kv_group_size=2).cuda()
    ids = torch.arange(256, device='cuda').unsqueeze(0) % 32000
    cu = torch.tensor([0, 256], dtype=torch.int32, device='cuda')
    out = model(ids, cu_seqlens=cu, hard_mode=False)
    out.float().square().mean().backward()
    assert torch.isfinite(out).all()
    grads = {n: p.grad for n, p in model.named_parameters() if p.requires_grad}
    for n in ('layers.6.attn.k_conv.weight', 'layers.6.attn.k_vocab', 'layers.6.attn.v_conv.weight'):
        g = grads[n]
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0, n
    print('G=2 72m forward+backward: finite, k_conv/k_vocab/v_conv gradients nonzero')


@CUDA
def test_group2_hard_row_is_finite():
    """The hard path reads the shared key through every query slot."""
    model = build(kv_group_size=2).cuda()
    ids = torch.arange(256, device='cuda').unsqueeze(0) % 32000
    cu = torch.tensor([0, 256], dtype=torch.int32, device='cuda')
    out = model(ids, cu_seqlens=cu, hard_mode=True)
    assert torch.isfinite(out).all()
    print('G=2 hard path: finite')


def test_inference_rejected():
    from flash_dism.decoding import forward_torch
    from nanochat.hard_inference import HardInferenceEngine
    m = DismAttention(64, 2, kv_group_size=2).eval()
    x = torch.zeros(1, 1, 64)
    with pytest.raises(NotImplementedError, match='shared-KV'):
        m(x)
    with pytest.raises(NotImplementedError, match='shared-KV'):
        forward_torch(m, x, None)
    with pytest.raises(NotImplementedError, match='shared-KV'):
        HardInferenceEngine(types.SimpleNamespace(config=types.SimpleNamespace(
            gqa_query_channels=1, kv_group_size=2)), None, capacity=256)
