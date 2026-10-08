"""Evaluation-only hard DISM with a finite suffix-order horizon.

Cap C keeps every matching edge, replacing sum(exp(m*tau), m=1..L)
by the same sum up to min(L,C). It does NOT discard long matching edges.
Dense intermediates are intentional for this diagnostic, not production.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _maximum(a, b):
    return tl.maximum(a, b)


@triton.jit
def _lengths(Q, K, OUT, START, T: tl.constexpr,
             N: tl.constexpr, BLOCK: tl.constexpr):
    h = tl.program_id(0)
    lag = tl.program_id(1)
    j = tl.arange(0, BLOCK)
    i = j + lag
    valid = i < N
    q = tl.load(Q + h * T + START + i, valid, other=-1)
    k = tl.load(K + h * T + START + j, valid, other=-2)
    last_mismatch = tl.associative_scan(tl.where((q != k) | ~valid, j + 1, 0), 0, _maximum)
    tl.store(OUT + (h * N + i) * N + j, j + 1 - last_mismatch, valid)


def lengths(iq, ik, start, end):
    assert iq.is_contiguous() and ik.is_contiguous()
    assert iq.shape == ik.shape and iq.shape[0] == 1
    _, heads, total = iq.shape
    n = end - start
    result = torch.zeros((heads, n, n), device=iq.device, dtype=torch.int16)
    _lengths[(heads, n)](iq, ik, result, start, total, n, triton.next_power_of_2(n))
    return result


def probabilities(run_lengths, tau, cap=0, distance='all'):
    """Normalized matching weights; denominator retains the score-0 fallback."""
    n = run_lengths.shape[-1]
    length = run_lengths.float()
    if distance == 'window128':
        pos = torch.arange(n, device=length.device)
        length = length.masked_fill((pos[:, None] - pos[None, :]) > 128, 0)
    if cap:
        shortened = length.clamp_max(cap)
        if distance != 'all':
            pos = torch.arange(n, device=length.device)
            far = (pos[:, None] - pos[None, :]) > 128
            selected = far if distance == 'far' else ~far
            shortened = torch.where(selected, shortened, length)
        length = shortened
    t = tau.float()[:, None, None]
    # Stable geometric sum for nonnegative tau, including tau=0 exactly.
    safe_t = t.clamp_min(torch.finfo(torch.float32).tiny)
    logw = length * t + torch.log(-torch.expm1(-length * safe_t)) - torch.log(-torch.expm1(-safe_t))
    logw = torch.where(t == 0, length.log(), logw)
    logw = logw.masked_fill(length == 0, -torch.inf)
    logz = torch.logaddexp(torch.logsumexp(logw, dim=-1), torch.zeros_like(logw[:, :, 0]))
    return torch.exp(logw - logz[..., None])


@torch.no_grad()
def hard_output(sq, sk, v, iq, ik, tau, boundaries, cap=0, distance='all'):
    assert (tau >= 0).all()
    result = torch.empty_like(v)
    # The enclosing model uses BF16 autocast; disable it for diagnostic GEMMs.
    with torch.autocast('cuda', enabled=False):
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            if start == end:
                continue
            run = lengths(iq, ik, start, end)
            p = probabilities(run, tau, cap, distance)
            qs = sq[0, start:end].float().transpose(0, 1)
            ks = sk[0, start:end].float().transpose(0, 1)
            vs = v[0, start:end].float().transpose(0, 1)
            readout = torch.bmm(qs, ks.transpose(1, 2))
            output = torch.bmm(p * readout, vs)
            result[0, start:end] = output.transpose(0, 1).to(v.dtype)
    return result


def test():
    torch.manual_seed(391)
    torch.backends.cuda.matmul.allow_tf32 = False
    for same in (False, True):
        n, h, r, dv = 47, 3, 7, 11
        iq = torch.randint(0, 3, (1, h, n), dtype=torch.int32, device='cuda')
        ik = iq.clone() if same else torch.randint_like(iq, 0, 3)
        if same:
            iq.zero_(); ik.zero_()
        sq, sk = [torch.randn((1, n, h, r), device='cuda') for _ in range(2)]
        v = torch.randn((1, n, h, dv), device='cuda')
        tau = torch.tensor([0., 1e-7, 4.2], device='cuda')
        cu = [0, 17, 17, 47]
        for cap in (0, 1, 2, 4):
            out = hard_output(sq, sk, v, iq, ik, tau, cu, cap)
            ref = torch.empty_like(out, dtype=torch.float64)
            for start, end in zip(cu[:-1], cu[1:]):
                if start == end: continue
                size = end - start
                run = lengths(iq, ik, start, end)
                expected = torch.zeros_like(run)
                weights = torch.zeros_like(run, dtype=torch.float64)
                for i in range(size):
                    for j in range(i + 1):
                        prior = expected[:, i-1, j-1] if i and j else 0
                        expected[:, i, j] = (iq[0, :, start+i] == ik[0, :, start+j]) * (prior + 1)
                assert torch.equal(run, expected)
                # Independent explicit sum over suffix orders, not geometric formula.
                for order in range(1, (cap or size) + 1):
                    weights += (expected >= order) * torch.exp(order * tau.double())[:, None, None]
                p = weights / (1 + weights.sum(-1, keepdim=True))
                s = torch.einsum('bthr,bshr->hts', sq[:, start:end].double(), sk[:, start:end].double())
                ref[:, start:end] = torch.bmm(p * s, v[0, start:end].double().transpose(0, 1)).transpose(0, 1)
            torch.testing.assert_close(out.double(), ref, atol=7e-5, rtol=1e-4)
    # Exercise near/far masking and ensure nonselected edges retain raw weights.
    run = torch.ones((1, 193, 193), device='cuda', dtype=torch.int16).tril().mul_(8)
    for distance in ('near', 'far'):
        p = probabilities(run, torch.tensor([.3], device='cuda'), 1, distance)
        assert torch.isfinite(p).all() and (p.sum(-1) <= 1.000001).all()
        pos = torch.arange(193, device='cuda')
        far = (pos[:, None] - pos[None, :]) > 128
        chosen = far if distance == 'far' else ~far
        limit = torch.where(chosen, run.clamp_max(1), run)
        weights = torch.zeros_like(run, dtype=torch.float64)
        for order in range(1, 9):
            weights += (limit >= order) * torch.exp(torch.tensor(order, device='cuda') * torch.tensor(.3, device='cuda').double())
        reference = weights / (1 + weights.sum(-1, keepdim=True))
        torch.testing.assert_close(p.double(), reference, atol=1e-7, rtol=1e-5)
    empty = probabilities(torch.zeros_like(run), torch.tensor([.3], device='cuda'), 1)
    assert torch.count_nonzero(empty) == 0
    window = probabilities(run, torch.tensor([.3], device='cuda'), distance='window128')
    assert torch.count_nonzero(window[:, far]) == 0
    window_run = run.masked_fill(far, 0)
    torch.testing.assert_close(window, probabilities(window_run, torch.tensor([.3], device='cuda')), atol=0, rtol=0)
    torch.testing.assert_close(probabilities(run, torch.tensor([.3], device='cuda'), 193),
                               probabilities(run, torch.tensor([.3], device='cuda')),
                               atol=0, rtol=0)
    print('MATCH_LENGTH_CORE_TEST_PASS', flush=True)


if __name__ == '__main__':
    test()
