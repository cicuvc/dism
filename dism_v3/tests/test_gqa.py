"""GQA adapter regressions. Baseline is the pre-GQA commit, never an installed tree."""
import json
import linecache
from pathlib import Path
import subprocess
import sys
import types

import pytest
import torch

from flash_dism.module import DismAttention

ROOT = Path(__file__).resolve().parents[2]
BASE = 'a95d69c'
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA driver unavailable')


def head_module(path, name, replacements=()):
    source = subprocess.check_output(['git', 'show', f'{BASE}:{path}'], cwd=ROOT, text=True)
    for old, new in replacements:
        source = source.replace(old, new)
    module = types.ModuleType(name)
    module.__package__ = name.rpartition('.')[0]
    sys.modules[name] = module
    filename = f'{BASE}:{path}'
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    exec(compile(source, filename, 'exec'), module.__dict__)
    return module


@pytest.fixture(scope='module')
def baseline():
    head_module('dism_v3/python/flash_dism/module.py', 'flash_dism._gqa_baseline_module')
    head_module('dism_v3/python/flash_dism/hybrid_gdn.py', 'flash_dism._gqa_baseline_hybrid',
                [('from .module import', 'from ._gqa_baseline_module import')])
    head_module('dism_v3/python/flash_dism/modeling_dism.py', 'flash_dism._gqa_baseline_modeling',
                [('from .module import', 'from ._gqa_baseline_module import'),
                 ('from .hybrid_gdn import', 'from ._gqa_baseline_hybrid import')])
    return head_module('nanochat/nanochat/models/dism/model.py',
                       'nanochat.models.dism._gqa_baseline_model',
                       [('from flash_dism.modeling_dism import',
                         'from flash_dism._gqa_baseline_modeling import')]).DismLM


def build(channels=1, cls=None):
    from nanochat.models.dism import DismConfig
    from nanochat.models.dism.model import DismLM
    config = json.loads((ROOT / 'nanochat/configs/sequence_true_72m_valres.json').read_text())['config']
    config['gqa_query_channels'] = channels
    torch.manual_seed(1729)
    model = (cls or DismLM)(DismConfig(**config), attention_type='hybrid_gdn')
    model.init_weights()
    return model


def test_head_state_identity_and_counts(baseline):
    old, current = build(cls=baseline), build()
    a, b = old.state_dict(), current.state_dict()
    assert a.keys() == b.keys()
    assert all(a[k].shape == b[k].shape and torch.equal(a[k], b[k]) for k in a)
    assert sum(p.numel() for p in current.parameters()) == 72035808
    print(f'G1: keys={len(a)}, every shape/value identical to {BASE}, total=72035808')
    del old, a, b
    two = build(2)
    added = {n: p.numel() for n, p in two.named_parameters() if '.gqa_' in n}
    expected = {'gqa_q_conv.weight': 147456, 'gqa_q_conv.conv_weight': 1536,
                'gqa_sq_proj.weight': 73728, 'gqa_mix': 12}
    assert len(added) == 24
    for i in range(6, 12):
        for name, count in expected.items():
            assert added[f'layers.{i}.attn.{name}'] == count
    assert sum(added.values()) == 1336392
    assert sum(p.numel() for p in two.parameters()) == 73372200
    for one_layer, two_layer in zip(current.layers, two.layers):
        one, other = one_layer.attn, two_layer.attn
        if getattr(one, 'is_pure_gdn', False):
            assert {k: p.shape for k, p in one.named_parameters()} == {k: p.shape for k, p in other.named_parameters()}
            continue
        assert other.gdn_q_conv.weight is other.q_conv.weight
        assert other.gdn_k_conv.weight is other.k_conv.weight
        assert torch.equal(other.gqa_mix, torch.full((6, 2), .5))
        for name, p in one.named_parameters():
            assert dict(other.named_parameters())[name].shape == p.shape
    attn = two.layers[6].attn
    counts = {name: sum(p.numel() for p in getattr(attn, name).parameters())
              for name in ('q_conv', 'k_conv', 'v_conv', 'gdn_q_conv', 'gdn_k_conv', 'gdn_v_conv')}
    counts.update({name: getattr(attn, name).numel() for name in ('q_vocab', 'k_vocab')})
    counts['gdn_named_including_qk_aliases'] = sum(p.numel() for n, p in attn.named_parameters(remove_duplicate=False) if n.startswith('gdn_'))
    print(f'G2: total=73372200 delta={sum(added.values())} per_layer={expected}; unchanged={counts}')


@CUDA
def test_head_fixed_seed_forward(baseline):
    old, current = build(cls=baseline).cuda(), build().cuda()
    ids = torch.arange(256, device='cuda').unsqueeze(0) % 32000
    cu = torch.tensor([0, 256], dtype=torch.int32, device='cuda')
    old.eval(); current.eval()
    old.eval_hard = current.eval_hard = False
    with torch.no_grad():
        torch.manual_seed(73)
        a = old(ids, cu_seqlens=cu)
        torch.manual_seed(73)
        b = current(ids, cu_seqlens=cu)
    assert torch.equal(a, b)
    print('G1 full 72m model forward: bitwise equal, B=1 N=256')


@pytest.mark.parametrize('axis,shape', [(0, (2,)), (0, (2, 64, 32)),
                                       (1, (2, 2)), (2, (2, 256, 2, 32))])
def test_shared_expansion_gradient(axis, shape):
    x = torch.randn(shape, requires_grad=True)
    out = DismAttention._expand_gqa(x, axis)
    assert out.is_contiguous()
    assert torch.equal(out.select(axis, 0), x.select(axis, 0))
    assert torch.equal(out.select(axis, 1), x.select(axis, 0))
    out.sum().backward()
    assert torch.equal(x.grad, torch.full_like(x, 2))


def test_mix_after_normalization_and_gradient():
    m = DismAttention(64, 2, head_dim=32, value_dim=32, readout_dim=16,
                      vocab_size=64, gqa_query_channels=2)
    # These are normalized core outputs, including a channel with no matches.
    out = torch.randn(2, 256, 2, 2, 32)
    out[:, :, :, 1] = 0
    with torch.no_grad():
        m.gqa_mix.copy_(torch.tensor([[.25, .75], [.6, .4]]))
    actual = m._mix_gqa(out.flatten(2, 3))
    manual = out[:, :, :, 0] * m.gqa_mix[:, 0, None] + out[:, :, :, 1] * m.gqa_mix[:, 1, None]
    assert torch.equal(actual, manual)
    z = torch.randn_like(actual)
    (actual * z).sum().backward()
    expected = (out * z.unsqueeze(3)).sum((0, 1, 4))
    torch.testing.assert_close(m.gqa_mix.grad, expected)
    assert torch.count_nonzero(m.gqa_mix.grad[:, 1]) == 0
    print('post-core mix: exact hand equality; unmatched channel=0; gradient=sum <Z,O_g>')


@pytest.mark.parametrize('bad', [0, 3, True, 2.0])
def test_invalid_channels(bad):
    with pytest.raises(ValueError, match='gqa_query_channels'):
        DismAttention(64, 2, gqa_query_channels=bad)


def test_inference_rejected():
    from flash_dism.decoding import forward_torch
    from nanochat.hard_inference import HardInferenceEngine
    from flash_dism.inference import HardDismDecoder, HardDismPrefill
    with pytest.raises(NotImplementedError, match='GQA-DISM'):
        HardDismDecoder(1, 2, 16, 32, 256, [1., 1.], gqa_query_channels=2)
    with pytest.raises(NotImplementedError, match='GQA-DISM'):
        HardDismPrefill(gqa_query_channels=2)
    m = DismAttention(64, 2, gqa_query_channels=2).eval()
    x = torch.zeros(1, 1, 64)
    with pytest.raises(NotImplementedError, match='GQA-DISM'):
        m(x)
    with pytest.raises(NotImplementedError, match='GQA-DISM'):
        forward_torch(m, x, None)
    with pytest.raises(NotImplementedError, match='GQA-DISM'):
        HardInferenceEngine(types.SimpleNamespace(config=types.SimpleNamespace(gqa_query_channels=2)), None, capacity=256)


@CUDA
@pytest.mark.parametrize('fixed_true', [False, True])
def test_gqa_small_hybrid_backward(fixed_true, monkeypatch):
    from flash_dism.hybrid_gdn import DismGdnAttention
    m = DismGdnAttention(64, 2, head_dim=32, value_dim=32, readout_dim=16,
                         vocab_size=64, value_residual=True, gqa_query_channels=2).cuda()
    import flash_dism
    import flash_dism.compiler as compiler
    target, name = (compiler, 'voc_forward') if fixed_true else (flash_dism, 'voc_dism')
    original = getattr(target, name)
    captured = {}

    def record(*args, **kwargs):
        q, k, sq, sk, v, eq, ek, tau = args[:8]
        for shared, axis in ((k, 2), (sk, 2), (v, 2), (eq, 0), (ek, 0), (tau, 0)):
            assert shared.is_contiguous()
            assert torch.equal(shared.select(axis, 0), shared.select(axis, 1))
            assert torch.equal(shared.select(axis, 2), shared.select(axis, 3))
        assert not torch.equal(q[:, :, 0], q[:, :, 1])
        assert not torch.equal(sq[:, :, 0], sq[:, :, 1])
        result = original(*args, **kwargs)
        captured['raw'] = result[0] if fixed_true else result
        return result

    combine = m._combine_cuda

    def check_combine(output, x, v, *args, **kwargs):
        assert v.shape == (2, 256, 2, 32)
        raw = captured['raw'].reshape(2, 256, 2, 2, 32)
        expected = .5 * raw[:, :, :, 0].float() + .5 * raw[:, :, :, 1].float()
        assert torch.equal(output, expected)
        return combine(output, x, v, *args, **kwargs)

    monkeypatch.setattr(target, name, record)
    monkeypatch.setattr(m, '_combine_cuda', check_combine)
    x = torch.randn(2, 256, 64, device='cuda', requires_grad=True)
    v_first = torch.randn(2, 256, 2, 32, device='cuda', dtype=torch.bfloat16)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        out = m(x, hard_prob=0., direction=torch.ones(2, 2, device='cuda', dtype=torch.bool),
                direction_true=fixed_true, v_first=v_first)[0]
    out.float().square().mean().backward()
    assert out.shape == x.shape and torch.isfinite(out).all()
    for n, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), n
        if 'gqa_' in n:
            assert p.grad.abs().sum() > 0, n


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=CUDA)])
def test_independent_labels_and_unmatched_core_channel(device):
    from flash_dism.reference.dism_v3_ref import dism_wrapper
    n = 8 if device == 'cpu' else 256
    dtype = torch.float32 if device == 'cpu' else torch.bfloat16
    m = DismAttention(64, 2, head_dim=32, value_dim=32, readout_dim=16,
                      vocab_size=64, gqa_query_channels=2).to(device)
    table = torch.zeros(64, 32, device=device)
    table[0, 0], table[1, 0] = 1, -1
    q0 = torch.zeros(1, n, 2, 32, device=device, dtype=dtype)
    q0[..., 0] = 2
    q = torch.stack((q0, -q0), dim=3).flatten(2, 3)
    k = m._expand_gqa(q0, 2)
    sq = sk = torch.full((1, n, 4, 16), .25, device=device, dtype=dtype)
    v = torch.ones(1, n, 4, 32, device=device, dtype=dtype)
    tau = torch.ones(4, device=device)
    direction = hard = torch.ones(1, 4, device=device, dtype=torch.bool)
    if device == 'cpu':
        idx_q = torch.einsum('bnhd,vd->bnhv', q, table).argmax(-1).transpose(1, 2)
        output = dism_wrapper(q, k, sq, sk, table, table, v, tau,
                              direction=direction, hard=hard[..., None].expand(1, 4, n))
    else:
        from flash_dism import voc_dism
        from flash_dism.emb_kernel import interp_embeddings
        eq = table.unsqueeze(0).expand(4, -1, -1).contiguous().to(dtype)
        idx_q = interp_embeddings(q, k, eq, eq, 1., tau)[-1]
        output = voc_dism(q, k, sq, sk, v, eq, eq, tau, direction=direction, hard=hard)
    assert torch.all(idx_q[:, 0::2] == 0) and torch.all(idx_q[:, 1::2] == 1)
    channels = output.reshape(1, n, 2, 2, 32)
    assert torch.count_nonzero(channels[:, :, :, 1]) == 0
    assert torch.all(channels[:, :, :, 0] > 0)
    mixed = m._mix_gqa(output)
    manual = .5 * channels[:, :, :, 0].float() + .5 * channels[:, :, :, 1].float()
    assert torch.equal(mixed, manual)
    mixed.sum().backward()
    torch.testing.assert_close(m.gqa_mix.grad, channels.float().sum((0, 1, 4)))
    print(f'{device}: idx_q=[0,1] per head; unmatched output exactly 0; mixed=0.5*O0; '
          f'O0_first={channels[0,0,0,0,0].item():.8f}')


def test_g1_cpu_forward_identity(baseline):
    old_cls = sys.modules['flash_dism._gqa_baseline_module'].DismAttention
    kwargs = dict(head_dim=32, value_dim=32, readout_dim=16, vocab_size=64)
    torch.manual_seed(93)
    old = old_cls(64, 2, **kwargs).eval()
    torch.manual_seed(93)
    current = DismAttention(64, 2, **kwargs).eval()
    x = torch.randn(2, 8, 64)
    direction = torch.ones(2, 2, dtype=torch.bool)
    old_decode = head_module('dism_v3/python/flash_dism/decoding.py',
                             'flash_dism._gqa_baseline_decoding').forward_torch
    from flash_dism.decoding import forward_torch
    with torch.no_grad():
        a = old_decode(old, x, None, direction=direction, hard_prob=1.)[0]
        b = forward_torch(current, x, None, direction=direction, hard_prob=1.)[0]
    assert torch.equal(a, b)
    print('G1 direct CPU decoding forward: bitwise equal, B=2 N=8 H=2; max_abs_diff=0')


@CUDA
def test_g2_full_model_smoke():
    model = build(2).cuda()
    ids = torch.arange(256, device='cuda').unsqueeze(0) % 32000
    cu = torch.tensor([0, 256], dtype=torch.int32, device='cuda')
    out = model(ids, cu_seqlens=cu, hard_mode=False)
    out.float().square().mean().backward()
    assert torch.isfinite(out).all()
    for name, p in model.named_parameters():
        if '.gqa_' in name:
            assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, name
