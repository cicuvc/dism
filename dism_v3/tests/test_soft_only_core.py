"""Sequence-hard is an explicit skip, not an implementation of hard attention."""
import pytest
import torch
import cu_flash_dism as cu

from flash_dism.reference.dism_v3_ref import dism_ref, dism_ref_backward
from flash_dism.varlen import VarlenLayout
from gradient_acceptance import assert_gradient


def fixture(packed, direction, mode, r=32, d=64, dv=64):
    torch.manual_seed(417)
    b, n, h = (1, 1024, 2) if packed else (2, 512, 2)
    vectors = [(torch.randn(b, n, h, c, device='cuda') * .3).bfloat16()
               for c in (d, d, r, r, dv)]
    lse = [torch.full((b, n, h), 3., device='cuda') for _ in range(2)]
    labels = [torch.randint(64, (b, h, n), device='cuda', dtype=torch.int32)
              for _ in range(2)]
    hard = torch.tensor([[False, True], [True, False]], device='cuda')
    if mode != 'mixed':
        hard.fill_(mode == 'hard')
    raw = (*vectors, *lse, *labels, torch.full((b, h), direction, device='cuda'),
           hard, torch.full((h,), .5, device='cuda'))
    # Two training rows, multiple documents in row 0, an empty document, and
    # one longer document in row 1. Flags are not sampled per document.
    offsets = [0, 256, 256, 512, 1024]
    table = VarlenLayout.from_cu_seqlens(
        torch.tensor(offsets, device='cuda', dtype=torch.int32), n).table if packed else None
    return raw, table, offsets


def run(raw, table, dout, ctas=1):
    out, norm, state = cu.core_forward(raw, table, 0 if table is not None else ctas,
                                       True, False, False)
    grads, _ = cu.core_backward(state['operands'], state['vertical'], out,
                               state['normalizer'], dout, table,
                               0 if table is not None else ctas, False, False)
    return out, norm, grads


@pytest.mark.parametrize('packed', [False, True])
@pytest.mark.parametrize('direction', [False, True])
@pytest.mark.parametrize('mode', ['soft', 'mixed', 'hard'])
def test_soft_oracle_and_hard_zero(packed, direction, mode):
    raw, table, offsets = fixture(packed, direction, mode)
    q, k, sq, sk, v, lq, lk, iq, ik, direct, hard, tau = raw
    dout = torch.randn_like(v) * .1
    out, norm, grads = run(raw, table, dout)
    expected_out = torch.zeros_like(out, dtype=torch.float64)
    expected = {name: torch.zeros_like(value, dtype=torch.float64)
                for name, value in grads.items()}
    docs = [(0, a, z, a // 512) for a, z in zip(offsets, offsets[1:]) if z > a] if packed else [
        (row, 0, 512, row) for row in range(2)]
    for b, a, z, row in docs:
        soft_tokens = torch.zeros((1, 2, z-a), device='cuda', dtype=torch.bool)
        args = [x[b:b+1, a:z].double() for x in (q, k, sq, sk, lq, lk)]
        args += [iq[b:b+1, :, a:z], ik[b:b+1, :, a:z], direct[b:b+1], soft_tokens,
                 v[b:b+1, a:z].double(), tau.double()]
        keep = (~hard[row])[None, None, :, None]
        expected_out[b:b+1, a:z] = dism_ref(*args) * keep
        derivative = dout[b:b+1, a:z].double() * keep
        oracle = dism_ref_backward(*args, derivative)
        for name, value in oracle.items():
            if name == 'rtau':
                expected[name] += value
            else:
                expected[name][b:b+1, a:z] = value
        torch.testing.assert_close(out[b:b+1, a:z, hard[row]],
                                   torch.zeros_like(out[b:b+1, a:z, hard[row]]), rtol=0, atol=0)
        assert torch.count_nonzero(norm[b, hard[row], a:z]) == 0
        for name, value in grads.items():
            if name != 'rtau':
                assert torch.count_nonzero(value[b:b+1, a:z, hard[row]]) == 0, name
    torch.testing.assert_close(out.double(), expected_out, atol=.008, rtol=.025)
    assert (out.double()-expected_out).norm()/expected_out.norm().clamp_min(1e-8) < .01
    for name, value in grads.items():
        assert_gradient(value, expected[name], name)


@pytest.mark.parametrize('packed', [False, True])
def test_token_masks_rejected(packed):
    raw, table, _ = fixture(packed, True, 'soft')
    raw = list(raw)
    raw[10] = torch.zeros(raw[0].size(0), 2, raw[0].size(1), device='cuda', dtype=torch.bool)
    with pytest.raises(RuntimeError, match='token-level masks'):
        cu.core_forward(raw, table, 0, False, False, False)


def test_document_cannot_cross_training_rows():
    raw, _, _ = fixture(True, True, 'mixed')
    table = VarlenLayout.from_cu_seqlens(
        torch.tensor([0, 768, 1024], device='cuda', dtype=torch.int32), 1024).table
    with pytest.raises(RuntimeError, match='cross a hard training-row'):
        cu.core_forward(raw, table, 0, False, False, False)


def test_persistent_skip_transitions_and_replay():
    raw, table, _ = fixture(False, True, 'mixed')
    dout = torch.randn_like(raw[4]) * .1
    before = raw[10].clone()
    expected, norm, grads = run(raw, table, dout, 1)
    for ctas in (2, 0, 1):
        actual, normalizer, got = run(raw, table, dout, ctas)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(normalizer, norm, atol=0, rtol=0)
        for name in grads:
            torch.testing.assert_close(got[name], grads[name], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(raw[10], before, atol=0, rtol=0)


@pytest.mark.parametrize('packed', [False, True])
def test_all_hard_never_reads_nan_scores(packed):
    raw, table, _ = fixture(packed, True, 'hard')
    for x in raw[:7]:
        x.fill_(float('nan'))
    out, norm, gradients = run(raw, table, torch.zeros_like(raw[4]))
    for x in (out, norm, *gradients.values()):
        assert torch.isfinite(x).all() and torch.count_nonzero(x) == 0


@pytest.mark.parametrize('packed', [False, True])
@pytest.mark.parametrize('direction', [False, True])
def test_runtime_skip_safety(packed, direction):
    """Small sanitizer target without unrelated FP64 oracle GPU launches."""
    raw, table, _ = fixture(packed, direction, 'mixed')
    out, norm, gradients = run(raw, table, torch.randn_like(raw[4]))
    for x in (out, norm, *gradients.values()):
        assert torch.isfinite(x).all()


@pytest.mark.parametrize('config', sorted(cu.configs))
def test_each_compiled_config_skips_hard(config):
    raw, table, _ = fixture(False, True, 'hard', *config)
    out, norm, gradients = run(raw, table, torch.zeros_like(raw[4]))
    for x in (out, norm, *gradients.values()):
        assert torch.count_nonzero(x) == 0
