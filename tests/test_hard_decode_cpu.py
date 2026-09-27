import math
import pytest
import torch

from dism_v2.hard_decode_cpu import HardDismCPUCache, hard_labels_cpu
from dism_v2.dism_ref import dism_recurrence, normalize_with_zero_fallback, voc_dism_ref


def dense(q, k, v, tau):
    matches = q.unsqueeze(-1) == k.unsqueeze(-2)
    log_m = torch.where(matches, tau.double()[None, :, None, None], -torch.inf)
    scores = dism_recurrence(log_m)
    return normalize_with_zero_fallback(scores, v.double()), scores


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize('dv', [32, 64, 128])
@pytest.mark.parametrize('n', [1, 17, 65])
def test_dense_recurrence(dtype, dv, n):
    torch.manual_seed(17)
    q, k = [torch.randint(7, (2, 3, n), dtype=torch.int32) for _ in range(2)]
    v = torch.randn(2, 3, n, dv).to(dtype)
    tau = torch.tensor([-3., 0., math.log(64)])
    expected, scores = dense(q, k, v, tau)
    cache = HardDismCPUCache(tau, batch_size=2, value_dim=dv, output_dtype=torch.float64)
    rng = torch.get_rng_state()
    out = []
    for i in range(n):
        out.append(cache.step(q[:, :, i], k[:, :, i], v[:, :, i]))
        for bh, cursors in enumerate(cache.active_cursors):
            b, h = divmod(bh, 3)
            row = scores[b, h, i]
            assert set(cursors) == set(torch.where(torch.isfinite(row))[0].tolist())
            for j, score in cursors.items():
                assert score == pytest.approx(row[j].item(), rel=1e-9, abs=1e-8)
    torch.testing.assert_close(torch.stack(out, 2), expected, atol=1e-8, rtol=1e-8)
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize('d', [32, 64, 128])
@pytest.mark.parametrize('dv', [32, 64, 128])
def test_qkv_full_reference(d, dv):
    torch.manual_seed(5)
    q, k = [torch.randn(1, 2, 13, d).bfloat16() for _ in range(2)]
    v = torch.randn(1, 2, 13, dv)
    qvoc, kvoc = [torch.randn(2, 11, d).bfloat16() for _ in range(2)]
    tau = torch.tensor([0., math.log(d)])
    cache = HardDismCPUCache(tau, value_dim=dv)
    out = torch.stack([cache.step_qkv(q[:, :, i], k[:, :, i], v[:, :, i], qvoc, kvoc)
                       for i in range(13)], 2)
    for direction in ('q_from_k', 'k_from_q'):
        expected = voc_dism_ref(q, k, v, tau, qvoc, kvoc,
                               hard_prob=1., direction=direction)
        torch.testing.assert_close(out, expected, atol=3e-6, rtol=3e-6)


def test_new_starts_dead_cursors_and_fallback():
    cache = HardDismCPUCache(torch.tensor([-2.]), value_dim=1, output_dtype=torch.float64)
    def step(q, k, v):
        return cache.step(torch.tensor([[q]]), torch.tensor([[k]]), torch.tensor([[[float(v)]]]))
    assert step(9, 1, 2).item() == 0  # No active cursor at all.
    first = step(1, 2, 8).item()  # New match at an OLD key, despite no prior cursor.
    assert first == pytest.approx(2 / (1 + math.exp(2)))
    assert cache.active_cursors == ({0: -2.},)  # Negative finite states are retained.
    step(2, 7, 4)
    assert cache.active_cursors[0][1] == pytest.approx(-2 + math.log1p(math.exp(-2)))
    assert step(99, 99, 1).item() == pytest.approx(1 / (1 + math.exp(2)))
    assert set(cache.active_cursors[0]) == {3}


@pytest.mark.parametrize('tau', [-1000., 0., 10000.])
def test_long_chain_stability(tau):
    n = 1024
    cache = HardDismCPUCache(torch.tensor([tau]), value_dim=1, output_dtype=torch.float64)
    labels = torch.arange(n).reshape(1, 1, n)
    out = cache.prefill(labels, labels, torch.ones(1, 1, n, 1))
    assert torch.isfinite(out).all()
    assert len(cache.active_cursors[0]) == 1
    if tau == 0:
        expected = torch.arange(1, n + 1, dtype=torch.float64) / torch.arange(2, n + 2, dtype=torch.float64)
        torch.testing.assert_close(out.flatten(), expected, atol=1e-12, rtol=1e-12)
    elif tau > 0:
        assert out.min() == 1
    else:
        assert out.max() == 0  # Exponents below FP64 representability.


def test_chunks_clone_reset_and_owned_values():
    torch.manual_seed(19)
    labels = torch.randint(3, (1, 1, 29))
    keys = torch.randint(3, labels.shape)
    values = torch.randn(1, 1, 29, 3)
    cache = HardDismCPUCache(torch.tensor([1.]), value_dim=3)
    first = cache.prefill(labels[:, :, :11], keys[:, :, :11], values[:, :, :11])
    fork = cache.clone()
    rest = cache.prefill(labels[:, :, 11:], keys[:, :, 11:], values[:, :, 11:])
    other = fork.prefill(labels[:, :, 11:], keys[:, :, 11:], values[:, :, 11:])
    torch.testing.assert_close(rest, other, atol=0, rtol=0)
    expected, _ = dense(labels, keys, values, torch.tensor([1.]))
    torch.testing.assert_close(torch.cat((first, rest), 2).double(), expected, atol=2e-7, rtol=2e-7)
    cache.reset()
    assert cache.length == 0 and fork.length == 29
    label = torch.tensor([[2]])
    value = torch.ones(1, 1, 3)
    cache.step(label, label, value)
    value.zero_()
    output = cache.step(label, torch.tensor([[3]]), value)
    torch.testing.assert_close(output, torch.full_like(output, 1 / (1 + math.exp(-1))))


def test_contracts_and_tied_argmax():
    cache = HardDismCPUCache(torch.tensor([1.]), value_dim=2)
    label = torch.tensor([[1]])
    with pytest.raises(ValueError):
        cache.step(label.float(), label, torch.ones(1, 1, 2))
    with pytest.raises(ValueError):
        cache.step(label, label, torch.full((1, 1, 2), float('nan')))
    assert cache.length == 0
    assert cache.prefill(torch.empty(1, 1, 0, dtype=torch.int64),
                         torch.empty(1, 1, 0, dtype=torch.int64), torch.empty(1, 1, 0, 2)).shape == (1, 1, 0, 2)
    x, vocab = torch.ones(1, 1, 4), torch.ones(1, 3, 4)
    q, k = hard_labels_cpu(x, x, vocab, vocab)
    assert q.item() == k.item() == 0


def test_all_repeated_labels():
    labels = torch.zeros(1, 1, 257, dtype=torch.int64)
    values = torch.linspace(-1, 1, 257).reshape(1, 1, 257, 1)
    tau = torch.tensor([math.log(128)])
    cache = HardDismCPUCache(tau, value_dim=1, output_dtype=torch.float64)
    actual = cache.prefill(labels, labels, values)
    expected, _ = dense(labels, labels, values, tau)
    torch.testing.assert_close(actual, expected, atol=1e-8, rtol=1e-8)
    assert len(cache.active_cursors[0]) == 257
