"""Component probes and summary regression. No chunk-passing/output kernel."""
import math

import pytest
import torch
import torch.nn.functional as F

import cu_flash_dism
from flash_dism.summary import summarize


def affine_oracle(logm, valid_rows=None):
    """Sequential log2 affine composition, arbitrary incoming top/left state.

    logm: [...,rows,cols]. Return a,b of the bottom edge, not just b=W.
    Missing top/left transforms are identities; masked scores are zero maps.
    """
    a = torch.zeros_like(logm[..., 0, :])
    b = torch.full_like(a, -torch.inf)
    for row in range(logm.shape[-2]):
        pa = F.pad(a[..., :-1], (1, 0), value=0.)
        pb = F.pad(b[..., :-1], (1, 0), value=-torch.inf)
        m = logm[..., row, :]
        a = pa + m
        b = torch.logaddexp2(pb + m, m)
        if valid_rows is not None:
            valid = valid_rows[..., row, None]
            a = torch.where(valid, a, pa)
            b = torch.where(valid, b, pb)
    return torch.stack((a, b), -1)


@pytest.mark.parametrize('columns', [64, 128, 320])
def test_scan_probe(columns):
    torch.manual_seed(21)
    values = torch.randn(32, columns, device='cuda') * .5 - .2
    got = cu_flash_dism.scan_probe(values)
    expected = affine_oracle(values.double()).float()
    assert_finite_summary(got, expected, first_atol=2e-5)


def test_pipe_probe():
    got = cu_flash_dism.pipe_probe(torch.empty(0, device='cuda'), 101).cpu()
    for task in range(101):
        for tile in range(7):
            expected = (task * 7 + tile) * 4 if tile <= task % 7 else -2
            assert (got[task, :, tile] == expected).all(), (task, tile, got[task, :, tile])


@pytest.mark.parametrize('q_start', [0, 32, 64, 256])
def test_causal_mask_does_not_affect_valid_diagonals(q_start):
    """Same finite scan, masked versus unmasked, including K-slot crossings."""
    torch.manual_seed(83)
    values = torch.randn(32, 320, device='cuda') * .5 - .2
    values[torch.rand_like(values) < .2] = -1e6  # Hard mismatches remain masked.
    rows = q_start + torch.arange(32, device='cuda')
    columns = torch.arange(320, device='cuda')
    noncausal = columns[None, :] > rows[:, None]
    # Deliberately strong upper-triangle values expose any layout contamination.
    values[noncausal] = 100.
    masked = values.masked_fill(noncausal, -1e6)
    probe = cu_flash_dism.scan_probe
    valid = columns <= q_start + 31
    torch.testing.assert_close(probe(values)[valid], probe(masked)[valid], atol=0, rtol=0)


def assert_finite_summary(got, expected, first_atol=2e-4):
    """Keep the mathematical oracle; classify finite zero-map approximations.

    The affine first component is only addition, so it retains the strict
    tolerance. Only the softplus component uses the existing tanh tolerance.
    """
    reachable = torch.isfinite(expected)
    assert torch.isfinite(got).all()
    assert (got[~reachable] < -1e5).all()
    for component, tolerance in enumerate((first_atol, .03)):
        mask = reachable[..., component]
        torch.testing.assert_close(got[..., component][mask], expected[..., component][mask],
                                   atol=tolerance, rtol=2e-5)


def assert_causal_summary(got, expected):
    """Only bottom-row j<=i is in the production summary contract."""
    assert got.shape == expected.shape
    rows = (torch.arange(got.shape[-3], device=got.device) + 1) * 32 - 1
    columns = torch.arange(got.shape[-2], device=got.device)
    valid = columns[None, :] <= rows[:, None]
    assert_finite_summary(got[..., valid, :], expected[..., valid, :])


@pytest.mark.parametrize('all_masked', [False, True])
def test_finite_masked_scan(all_masked):
    torch.manual_seed(37)
    values = torch.randn(32, 320, device='cuda') * .5 - .2
    masked = torch.rand_like(values) < .2
    if all_masked:
        masked.fill_(True)
    oracle = affine_oracle(values.double().masked_fill(masked, -torch.inf)).float()
    finite = values.masked_fill(masked, -1e6)
    probe = cu_flash_dism.scan_probe
    assert_finite_summary(probe(finite), oracle)


def test_rv_load_probe():
    source = torch.arange(32, device='cuda', dtype=torch.float32)
    for direct in (False, True):
        torch.testing.assert_close(cu_flash_dism.rv_load_probe(source, direct), source, atol=0, rtol=0)


def make_inputs(n, mode, direction_mode):
    torch.manual_seed(123)
    b, h, d = 2, 3, cu_flash_dism.summary_key_dim()
    q, k = [(.1 * torch.randn(b, n, h, d, device='cuda')).bfloat16() for _ in range(2)]
    lq, lk = [torch.randn(b, n, h, device='cuda') * .2 + 2 for _ in range(2)]
    iq, ik = [torch.randint(0, 7, (b, h, n), device='cuda', dtype=torch.int32) for _ in range(2)]
    hard = torch.rand(b, h, n, device='cuda') < {'soft': 0., 'mixed': .5, 'hard': 1.}[mode]
    direction = torch.rand(b, h, device='cuda') < .5
    if direction_mode != 'mixed':
        direction.fill_(direction_mode == 'query')
    tau = torch.tensor([.1, .7, math.log(d)], device='cuda')
    return q, k, lq, lk, iq, ik, direction, hard, tau


def summarize_inputs(inputs, **kwargs):
    # Fixture/oracle keep raw LSE. Absorb tau explicitly OUTSIDE the kernel.
    q, k, lq, lk, *metadata = inputs
    tau = metadata[-1][None, None, :]
    return pack_summary(summarize(q, k, lq - tau, lk - tau, *metadata, **kwargs))


def pack_summary(result):
    """Only tests materialize AoS to compare with the unchanged affine oracle."""
    assert isinstance(result, tuple) and len(result) == 2
    a, b = result
    assert a.shape == b.shape
    assert a.dtype == b.dtype == torch.float32
    assert a.is_contiguous() and b.is_contiguous()
    assert a.stride(-1) == b.stride(-1) == 1
    if a.numel():
        assert a.untyped_storage().data_ptr() != b.untyped_storage().data_ptr()
    return torch.stack((a, b), dim=-1)


@pytest.mark.parametrize('direction', ['query', 'key'])
def test_soft_uses_preabsorbed_lse(direction):
    inputs = make_inputs(512, 'soft', direction)
    q, k, lq, lk, iq, ik, selected_direction, hard, tau = inputs
    lq, lk = lq - tau[None, None, :], lk - tau[None, None, :]
    got = pack_summary(summarize(q, k, lq, lk, iq, ik, selected_direction, hard, tau, ctas=1))
    # With LSE held fixed, rtau must have NO effect on soft scores. This also
    # rejects accidental second absorption inside the Python wrapper.
    changed = pack_summary(summarize(q, k, lq, lk, iq, ik, selected_direction, hard, tau + .5, ctas=1))
    torch.testing.assert_close(got, changed, atol=0, rtol=0)
    assert_causal_summary(got, summary_oracle(inputs, got.shape[-2]))


def summary_oracle(inputs, padded):
    q, k, lq, lk, iq, ik, direction, hard, tau = inputs
    n = q.shape[1]
    summary_rows = (n - 1) // 32 * 32
    if summary_rows == 0:
        return torch.empty(q.shape[0], q.shape[2], 0, padded, 2, device=q.device)
    score = torch.einsum('bnhd,bmhd->bhnm', q.double(), k.double())
    bias = torch.where(direction[:, :, None, None],
                       lq.transpose(1, 2)[:, :, :, None], lk.transpose(1, 2)[:, :, None, :])
    hard_score = torch.where(iq[..., None] == ik[..., None, :], 0., -torch.inf)
    score = torch.where(hard[..., None], hard_score, score - bias) + tau[None, :, None, None]
    score *= math.log2(math.e)
    causal = torch.ones(n, n, device='cuda', dtype=torch.bool).tril()
    score.masked_fill_(~causal, -torch.inf)
    score = F.pad(score, (0, padded - n, 0, (n + 31) // 32 * 32 - n), value=-torch.inf)
    outputs = []
    for start in range(0, summary_rows, 32):
        valid = torch.arange(start, start + 32, device='cuda') < n
        outputs.append(affine_oracle(score[..., start:start + 32, :], valid))
    return torch.stack(outputs, 2).float()


@pytest.mark.parametrize('n', [256,512,768])
@pytest.mark.parametrize('mode', ['soft', 'mixed', 'hard'])
@pytest.mark.parametrize('direction', ['query', 'key', 'mixed'])
def test_summary(n, mode, direction):
    inputs = make_inputs(n, mode, direction)
    # A single CTA forces repeated ring-wrap and workload transitions.
    got = summarize_inputs(inputs, ctas=1)
    assert got.shape[2] == (n - 1) // 32
    expected = summary_oracle(inputs, got.shape[-2])
    assert_causal_summary(got, expected)


@pytest.mark.parametrize('n', [33, 64, 97, 129, 161, 193, 225, 255, 500, 768])
def test_partial_cta_warp_checkpoints(n):
    """Former partial-workload inputs must now fail, except aligned768."""
    if n%256:
        with pytest.raises(RuntimeError,match="256-token aligned"):
            summarize_inputs(make_inputs(n,'mixed','mixed'))
        return
    inputs = make_inputs(n, 'mixed', 'mixed')
    got = summarize_inputs(inputs, ctas=1)
    assert got.shape[2] == (n - 1) // 32
    assert_causal_summary(got, summary_oracle(inputs, got.shape[-2]))


@pytest.mark.parametrize('n', [256,512,768])
def test_finite_summary(n):
    inputs = make_inputs(n, 'mixed', 'mixed')
    got = summarize_inputs(inputs, ctas=1)
    assert_causal_summary(got, summary_oracle(inputs, got.shape[-2]))


def test_dispatch_matches_persistent():
    inputs = make_inputs(768, 'mixed', 'mixed')
    torch.testing.assert_close(summarize_inputs(inputs), summarize_inputs(inputs, ctas=1), atol=0, rtol=0)


@pytest.mark.parametrize('direction', ['query', 'key'])
@pytest.mark.parametrize('single_head', [False, True])
def test_last_channel_tma_mapping(direction, single_head):
    """Exercise the last channel/swizzle panel across batches, heads and tiles."""
    inputs = list(make_inputs(768, 'soft', direction))
    if single_head:
        for i in range(4):
            inputs[i] = inputs[i][:, :, :1].contiguous()
        for i in (4, 5, 6, 7):
            inputs[i] = inputs[i][:, :1].contiguous()
        inputs[8] = inputs[8][:1].contiguous()
    q, k = inputs[:2]
    q.zero_()
    k.zero_()
    q[..., -1] = .5
    k[..., -1] = torch.arange(k.shape[1], device=k.device)[None, :, None] / 1024
    got = summarize_inputs(inputs, ctas=1)
    assert_causal_summary(got, summary_oracle(inputs, got.shape[-2]))


def test_unregistered_dimension_rejected():
    inputs = list(make_inputs(512, 'soft', 'query'))
    wrong = 128
    for index in (0, 1):
        shape = (*inputs[index].shape[:-1], wrong)
        inputs[index] = torch.zeros(shape, device='cuda', dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match='unsupported'):
        summarize_inputs(inputs)


@pytest.mark.parametrize('matched', [False, True])
def test_degenerate_hard_chain(matched):
    inputs = list(make_inputs(1024, 'hard', 'mixed'))
    inputs[4].zero_()
    inputs[5].fill_(0 if matched else 1)
    got = summarize_inputs(inputs, ctas=1)
    assert_causal_summary(got, summary_oracle(inputs, got.shape[-2]))


@pytest.mark.parametrize('query_lse', [False, True])
def test_triton_interpolation(query_lse):
    # v2's output names follow interpolation destinations: out_q/LSE_q/idx_q
    # come from K logits, and out_k/LSE_k/idx_k from Q logits. Do not infer
    # source ownership from tuple variable names.
    from dism_v2.emb_kernel import emb_fwd_wrapper
    inputs = list(make_inputs(512, 'mixed', 'query' if query_lse else 'key'))
    q, k = inputs[:2]
    h = q.shape[2]
    vocab_q, vocab_k = [torch.randn(h, 128, q.shape[-1], device='cuda').bfloat16() * .2 for _ in range(2)]
    oq, ok, lse_from_k, lse_from_q, _, _, label_k, label_q = emb_fwd_wrapper(
        q.transpose(1, 2), k.transpose(1, 2), vocab_q, vocab_k, sm_scale=1.)
    inputs[:6] = [q if query_lse else ok.transpose(1, 2),
                  oq.transpose(1, 2) if query_lse else k,
                  lse_from_q.transpose(1, 2), lse_from_k.transpose(1, 2),
                  label_q, label_k]
    got = summarize_inputs(inputs, ctas=1)
    # Isolate core error by using the SAME BF16 interpolation outputs as oracle.
    assert_causal_summary(got, summary_oracle(inputs, got.shape[-2]))


@pytest.mark.parametrize('n', [768,1536])
def test_sanitizer_smoke(n):
    """Tail + two/five persistent workloads and repeated three-slot reuse."""
    inputs = list(make_inputs(n, 'mixed', 'query'))
    for i in range(4):
        inputs[i] = inputs[i][:1, :, :1].contiguous()
    for i in (4, 5, 6, 7):
        inputs[i] = inputs[i][:1, :1].contiguous()
    inputs[8] = inputs[8][:1].contiguous()
    for direction in (False, True):
        inputs[6].fill_(direction)
        got = summarize_inputs(inputs, ctas=1)
        assert_causal_summary(got, summary_oracle(inputs, got.shape[-2]))
