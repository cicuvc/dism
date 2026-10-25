"""Independent suffix enumeration and numerical derivatives for the v4 oracle."""
import pytest
import torch

from flash_dism_v4 import dism_v4_from_embeddings, dism_v4_ref


def sample(batch=1, length=5, heads=2, vocab=4, dv=3):
    generator = torch.Generator().manual_seed(241)
    def rand(*shape):
        return torch.randn(*shape, dtype=torch.float64, generator=generator)
    return (rand(batch, length, heads, vocab), rand(batch, length, heads, vocab),
            rand(batch, length, heads, dv), rand(heads) * .2)


def enumerate_suffixes(q, k, v, tau, *, key_mode="hard", sq=None, sk=None, query_reset=None):
    """Enumerate each modified Q suffix directly, not by the production recurrence."""
    p, pk = q.softmax(-1), k.softmax(-1)
    iq, ik = q.argmax(-1), k.argmax(-1)
    batch, length, heads, vocab = q.shape
    batches = []
    for b in range(batch):
        rows = []
        for i in range(length):
            head_outputs = []
            start = 0
            if query_reset is not None:
                starts = query_reset[b, :i + 1].nonzero().flatten().tolist()
                start = starts[-1] if starts else 0
            for h in range(heads):
                candidates = []
                for c in range(vocab):
                    numerator = v.new_zeros(v.shape[-1])
                    denominator = v.new_ones(())
                    for j in range(i + 1):
                        if key_mode == "hard":
                            # Explicit suffix of Q[:i] + [c] against K[:j+1].
                            query = iq[b, :i, h].tolist() + [c]
                            matched = 0
                            while matched <= min(i - start, j) and query[i - matched] == int(ik[b, j - matched, h]):
                                matched += 1
                            powers = torch.arange(1, matched + 1, dtype=tau.dtype, device=tau.device)
                            weight = (tau[h] * powers).exp().sum()
                        else:
                            # Only the current K compatibility is soft. History
                            # still compares literal argmax strings.
                            matched = 0
                            while matched < min(i - start, j) and int(iq[b, i - 1 - matched, h]) == int(ik[b, j - 1 - matched, h]):
                                matched += 1
                            powers = torch.arange(1, matched + 2, dtype=tau.dtype, device=tau.device)
                            weight = (tau[h] * powers).exp().sum() * pk[b, j, h, c]
                        signed = 1. if sq is None else (sq[b, i, h] * sk[b, j, h]).sum()
                        numerator = numerator + weight * signed * v[b, j, h]
                        denominator = denominator + weight
                    candidates.append(numerator / denominator)
                head_outputs.append(p[b, i, h] @ torch.stack(candidates))
            rows.append(torch.stack(head_outputs))
        batches.append(torch.stack(rows))
    return torch.stack(batches)


@pytest.mark.parametrize("mode", ["hard", "soft"])
@pytest.mark.parametrize("tau_value", [-.7, 0., .6])
@pytest.mark.parametrize("readout", [False, True])
def test_suffix_enumeration_and_gradients(mode, tau_value, readout):
    q, k, v, tau = [x.requires_grad_() for x in sample()]
    tau = torch.full_like(tau, tau_value, requires_grad=True)
    sq, sk = (None, None)
    if readout:
        sq = torch.linspace(-1., 1., 20, dtype=q.dtype).reshape(1, 5, 2, 2).requires_grad_()
        sk = sq.detach().flip(1).clone().requires_grad_()
    kwargs = dict(key_mode=mode, sq=sq, sk=sk)
    actual = dism_v4_ref(q, k, v, tau, **kwargs)
    expected = enumerate_suffixes(q, k, v, tau, **kwargs)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-11)
    inputs = [q, k, v, tau] + ([] if sq is None else [sq, sk])
    cotangent = torch.linspace(-.8, 1.2, actual.numel(), dtype=q.dtype).reshape_as(actual)
    found = torch.autograd.grad((actual * cotangent).sum(), inputs, allow_unused=True)
    wanted = torch.autograd.grad((expected * cotangent).sum(), inputs, allow_unused=True)
    for x, y in zip(found, wanted):
        if x is None or y is None:
            assert x is None and y is None
        else:
            torch.testing.assert_close(x, y, atol=1e-11, rtol=1e-10)


@pytest.mark.parametrize("mode", ["hard", "soft", "jensen"])
def test_embedding_gradcheck(mode):
    q, k, v, tau = sample(length=3, heads=1, vocab=2, dv=2)
    eq = torch.tensor([[[1.2, -.3], [.2, .9], [-.7, -.5]]], dtype=q.dtype)
    ek = torch.tensor([[[.8, .2], [-.4, 1.1], [.3, -.6]]], dtype=q.dtype)
    sq = torch.tensor([[[[.4, -.2]], [[.1, .6]], [[-.5, .3]]]], dtype=q.dtype)
    sk = sq.flip(1).clone()
    inputs = tuple(x.requires_grad_() for x in (q, k, eq, ek, v, tau, sq, sk))
    def run(q, k, eq, ek, v, tau, sq, sk):
        return dism_v4_from_embeddings(q, k, eq, ek, v, tau, key_mode=mode, sq=sq, sk=sk)
    assert torch.autograd.gradcheck(run, inputs, eps=1e-6, atol=2e-6, rtol=2e-4)


def test_independent_normalization_not_ratio_of_mixtures():
    q = torch.tensor([[[[9., 1.]]]], dtype=torch.float64).log()
    k = torch.tensor([[[[1000., -1000.]]]], dtype=q.dtype)
    v = torch.ones(1, 1, 1, 1, dtype=q.dtype)
    tau = torch.tensor([10.], dtype=q.dtype).log()
    actual = dism_v4_ref(q, k, v, tau).item()
    assert actual == pytest.approx(.9 * 10 / 11, abs=1e-14)
    assert abs(actual - 9 / 10) > .05


def test_hard_k_has_no_gradient_but_current_q_does():
    q = torch.tensor([[[[.4, -.2]]]], dtype=torch.float64, requires_grad=True)
    k = torch.tensor([[[[1., -1.]]]], dtype=q.dtype, requires_grad=True)
    v = torch.ones(1, 1, 1, 1, dtype=q.dtype, requires_grad=True)
    tau = torch.zeros(1, dtype=q.dtype, requires_grad=True)
    dq, dk, dv, dt = torch.autograd.grad(dism_v4_ref(q, k, v, tau).sum(), (q, k, v, tau), allow_unused=True)
    assert dk is None
    assert dq.abs().sum() > 0 and dv.abs().sum() > 0 and dt.abs().sum() > 0


@pytest.mark.parametrize("mode", ["hard", "soft", "jensen"])
def test_causal_outputs_and_gradients(mode):
    q, k, v, tau = [x.requires_grad_() for x in sample()]
    full = dism_v4_ref(q, k, v, tau, key_mode=mode)
    short = dism_v4_ref(q[:, :3], k[:, :3], v[:, :3], tau, key_mode=mode)
    torch.testing.assert_close(full[:, :3], short, atol=0, rtol=0)
    for gradient in torch.autograd.grad(full[:, :3].sum(), (q, k, v), allow_unused=True):
        if gradient is not None:
            assert torch.count_nonzero(gradient[:, 3:]) == 0


def test_packed_segments_outputs_and_gradients():
    q, k, v, tau = [x.requires_grad_() for x in sample(length=6)]
    cu = torch.tensor([0, 2, 6], dtype=torch.int32)
    packed = dism_v4_ref(q, k, v, tau, key_mode="soft", cu_seqlens=cu)
    separate = torch.cat([dism_v4_ref(q[:, a:b], k[:, a:b], v[:, a:b], tau, key_mode="soft") for a, b in [(0, 2), (2, 6)]], dim=1)
    torch.testing.assert_close(packed, separate, atol=0, rtol=0)
    x = torch.autograd.grad(packed.square().sum(), (q, k, v, tau))
    y = torch.autograd.grad(separate.square().sum(), (q, k, v, tau))
    for a, b in zip(x, y):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_query_reset_keeps_old_keys_readable():
    q = torch.zeros(1, 3, 1, 1, dtype=torch.float64)
    v = torch.tensor([10., 20., 3.], dtype=q.dtype).reshape(1, 3, 1, 1)
    tau = torch.zeros(1, dtype=q.dtype)
    reset = torch.tensor([[False, False, True]])
    out = dism_v4_ref(q, q, v, tau, query_reset=reset)
    expected = enumerate_suffixes(q, q, v, tau, query_reset=reset)
    torch.testing.assert_close(out, expected, atol=1e-12, rtol=1e-12)
    assert out[0, 2, 0, 0].item() == pytest.approx(33 / 4)
    isolated = dism_v4_ref(q[:, 2:], q[:, 2:], v[:, 2:], tau)
    assert isolated.item() == pytest.approx(3 / 2)


@pytest.mark.parametrize("mode", ["hard", "soft"])
def test_batched_heads_and_resets(mode):
    q, k, v, tau = sample(batch=2)
    reset = torch.tensor([[False, False, True, False, False], [True, False, False, True, False]])
    out = dism_v4_ref(q, k, v, tau, key_mode=mode, query_reset=reset)
    expected = enumerate_suffixes(q, k, v, tau, key_mode=mode, query_reset=reset)
    torch.testing.assert_close(out, expected, atol=1e-12, rtol=1e-11)


def test_hard_k_limit_and_jensen_coefficient_bound():
    q, k, _, tau = sample(heads=1)
    v = torch.eye(5, dtype=q.dtype)[None, :, None, :]
    exact = dism_v4_ref(q, k, v, tau, key_mode="soft")
    bound = dism_v4_ref(q, k, v, tau, key_mode="jensen")
    assert bool((bound <= exact + 1e-13).all())
    assert bool((exact >= 0).all()) and bool((exact.sum(-1) <= 1 + 1e-13).all())
    peaked = torch.full_like(k, -1000.).scatter(-1, k.argmax(-1, keepdim=True), 1000.)
    hard = dism_v4_ref(q, peaked, v, tau)
    for mode in ("soft", "jensen"):
        torch.testing.assert_close(dism_v4_ref(q, peaked, v, tau, key_mode=mode), hard, atol=1e-12, rtol=1e-12)


def test_one_hot_query_reduces_to_single_hard_read():
    q, k, v, tau = sample(heads=1)
    peaked = torch.full_like(q, -1000.).scatter(-1, q.argmax(-1, keepdim=True), 1000.)
    out = dism_v4_ref(peaked, k, v, tau)
    expected = enumerate_suffixes(peaked, k, v, tau)
    torch.testing.assert_close(out, expected, atol=1e-12, rtol=1e-12)


def test_empty_candidates_and_extreme_chain_scales_have_finite_backward():
    q = torch.tensor([[[[1000., -1000.]], [[1000., -1000.]], [[1000., -1000.]], [[0., 0.]]]], dtype=torch.float64, requires_grad=True)
    k = torch.tensor([[[[1000., -1000.]], [[1000., -1000.]], [[1000., -1000.]], [[-1000., 1000.]]]], dtype=q.dtype)
    v = torch.tensor([[[[0., 0.]], [[0., 0.]], [[1., 0.]], [[0., 1.]]]], dtype=q.dtype, requires_grad=True)
    tau = torch.tensor([1000.], dtype=q.dtype, requires_grad=True)
    out = dism_v4_ref(q, k, v, tau)
    # Candidate masses are about exp(3000) and exp(4000), yet both matter.
    torch.testing.assert_close(out[0, -1, 0], torch.tensor([.5, .5], dtype=q.dtype), atol=1e-12, rtol=0)
    for gradient in torch.autograd.grad(out.square().sum(), (q, v, tau)):
        assert bool(torch.isfinite(gradient).all())


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_dtype_promotion_and_rng(dtype):
    data = [x.to(dtype).requires_grad_() for x in sample(heads=1)]
    before = torch.get_rng_state().clone()
    actual = dism_v4_ref(*data, key_mode="soft")
    expected = dism_v4_ref(*(x.double() for x in data), key_mode="soft")
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual.double(), expected, atol=2e-6, rtol=2e-5)
    actual.sum().backward()
    assert all(x.grad is not None and bool(torch.isfinite(x.grad).all()) for x in data)
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize("kwargs", [
    {"key_mode": "unknown"},
    {"cu_seqlens": torch.tensor([1, 5])},
    {"cu_seqlens": torch.tensor([0, 2, 2, 5])},
    {"cu_seqlens": torch.tensor([0., 5.])},
    {"query_reset": torch.zeros(1, 5)},
    {"sq": torch.zeros(1, 5, 2, 2)},
])
def test_invalid_contracts(kwargs):
    with pytest.raises(ValueError):
        dism_v4_ref(*sample(), **kwargs)


def test_nonfinite_logits_rejected():
    q, k, v, tau = sample()
    q[0, 0, 0, 0] = -torch.inf
    with pytest.raises(ValueError, match="finite"):
        dism_v4_ref(q, k, v, tau)
