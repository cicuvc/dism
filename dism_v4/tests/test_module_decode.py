import pytest
import torch
from torch.nn import functional as F
from flash_dism import DismAttention, DismCache as Cache
from flash_dism.reference.dism_v3_ref import dism_ref


def model_at(device='cpu', **kwargs):
    torch.manual_seed(7)
    return DismAttention(32, 2, head_dim=32, value_dim=32,
                         readout_dim=16, vocab_size=16, layer_idx=0, **kwargs).to(device).eval()


@pytest.mark.parametrize('normalize', [False, True])
def test_readout_activation_gradient(normalize):
    model = model_at(readout_l2_norm=normalize).double()
    projected = torch.randn(1, 2, 32, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(model._activate_readout, (projected,))
    actual = model._activate_readout(projected)
    expected = F.silu(projected).reshape(1, 2, 2, 16)
    if normalize:
        expected = F.normalize(expected, dim=-1, eps=1e-6)
        torch.testing.assert_close(actual.norm(dim=-1), torch.ones(1, 2, 2, dtype=torch.float64))
    torch.testing.assert_close(actual, expected)
    zeros = torch.zeros_like(projected, requires_grad=True)
    model._activate_readout(zeros).sum().backward()
    assert torch.isfinite(zeros.grad).all()


def dense_reference(model, x, direction, hard):
    def conv(layer):
        projected = F.linear(x, layer.weight)
        result = F.conv1d(projected.transpose(1, 2), layer.conv_weight.flip(0).T[:, None],
                          padding=layer.kernel_size-1, groups=layer.out_channels)
        return F.silu(result[:, :, :x.shape[1]].transpose(1, 2)).reshape(x.shape[0], x.shape[1], 2, -1)
    q, k, v = map(conv, (model.q_conv, model.k_conv, model.v_conv))
    def readout(layer):
        result = F.silu(F.linear(x, layer.weight)).reshape(
            x.shape[0], x.shape[1], model.heads, model.readout_dim)
        return F.normalize(result, dim=-1, eps=1e-6) if model.readout_l2_norm else result
    sq, sk = readout(model.sq_proj), readout(model.sk_proj)
    qs = torch.einsum('bnhd,hvd->bnhv', q, model.q_vocab)
    ks = torch.einsum('bnhd,hvd->bnhv', k, model.k_vocab)
    qfk = torch.einsum('bnhv,hvd->bnhd', ks.softmax(-1), model.q_vocab)
    kfq = torch.einsum('bnhv,hvd->bnhd', qs.softmax(-1), model.k_vocab)
    choose = direction[:, None, :, None]
    output = dism_ref(torch.where(choose, q, kfq), torch.where(choose, qfk, k), sq, sk,
                      qs.logsumexp(-1), ks.logsumexp(-1), qs.argmax(-1).transpose(1, 2),
                      ks.argmax(-1).transpose(1, 2), direction, hard, v,
                      F.softplus(model.log_sel_tau))
    output = output * torch.rsqrt(output.square().mean(-1, keepdim=True) + model.norm.eps)
    gate = model.g_proj_up(model.g_proj_down(x)).reshape_as(output)
    output = output * model.norm.weight * F.silu(gate)
    return model.o_proj(output.flatten(2))


@pytest.mark.parametrize('mode', ['soft', 'mixed', 'hard'])
@pytest.mark.parametrize('conv_size', [1, 4])
@pytest.mark.parametrize('readout_l2_norm', [False, True])
def test_cached_module_dense(mode, conv_size, readout_l2_norm):
    model = model_at(conv_size=conv_size, readout_l2_norm=readout_l2_norm).double()
    x = torch.randn(2, 9, 32, dtype=torch.float64)
    direction = torch.tensor([[True, False], [False, True]])
    hard = torch.rand(2, 2, 9) < {'soft': 0., 'mixed': .5, 'hard': 1.}[mode]
    expected = dense_reference(model, x, direction, hard)
    cache, parts = Cache(), []
    for start, end in ((0, 5), (5, 6), (6, 9)):
        output, attn, returned = model(x[:, start:end], past_key_values=cache,
            use_cache=True, output_attentions=True,
            direction=direction if start == 0 else None, hard=hard[:, :, start:end])
        assert returned is cache and attn is None and not output.requires_grad
        assert cache.get_seq_length() == end
        parts.append(output)
    torch.testing.assert_close(torch.cat(parts, 1), expected, atol=1e-10, rtol=1e-10)
    assert cache[0]['conv_state'][0].shape == (2, conv_size-1, 64)


def test_cache_read_only_and_validation():
    model = model_at()
    x = torch.randn(1, 3, 32)
    cache = Cache()
    model(x, past_key_values=cache, use_cache=True)
    snapshot = tuple(t.clone() for t in cache[0]['recurrent_state'])
    first = model(x[:, :1], past_key_values=cache, use_cache=False)[0]
    replay = model(x[:, :1], past_key_values=cache, use_cache=False)[0]
    torch.testing.assert_close(first, replay, atol=0, rtol=0)
    assert cache.get_seq_length() == 3
    for a, b in zip(snapshot, cache[0]['recurrent_state']):
        torch.testing.assert_close(a, b)
    with pytest.raises(ValueError, match='direction and rtau'):
        model(x[:, :1], past_key_values=cache, direction=~snapshot[6])
    model.train()
    with pytest.raises(ValueError, match='eval'):
        model(x, past_key_values=cache, use_cache=True)
    model.eval().layer_idx = None
    with pytest.raises(ValueError, match='layer_idx'):
        model(x, past_key_values=cache, use_cache=True)


def test_shared_layers_and_beam_reordering():
    first, second = model_at(), model_at()
    second.layer_idx = 1
    cache = Cache()
    x = torch.randn(2, 4, 32)
    direction = torch.tensor([[True, False], [False, True]])
    a = first(x, direction=direction, past_key_values=cache, use_cache=True)[0]
    second(a, direction=direction, past_key_values=cache, use_cache=True)
    assert cache.get_seq_length(0) == cache.get_seq_length(1) == 4
    token = torch.randn(2, 1, 32)
    expected = first(token, past_key_values=cache)[0]
    # Duplicate and reorder beams; per-head tau is batch-major in the cache too.
    beams = torch.tensor([1, 1, 0])
    cache.reorder_cache(beams)
    actual = first(token[beams], past_key_values=cache)[0]
    torch.testing.assert_close(actual, expected[beams], atol=1e-6, rtol=1e-5)
    for layer in (0, 1):
        assert all(t.shape[0] == 3 for t in cache[layer]['recurrent_state'])


def test_unsupported_cached_padding_is_explicit():
    model = model_at()
    x = torch.randn(1, 3, 32)
    cache = Cache()
    with pytest.raises(NotImplementedError, match='unpadded'):
        model(x, attention_mask=torch.tensor([[0, 1, 1]]), past_key_values=cache, use_cache=True)
    assert len(cache) == 0
    output, attention, result = model(x, attention_mask=torch.ones(1, 3), use_cache=True)
    assert output.shape == x.shape and attention is None and result is None


@pytest.mark.parametrize('tensor_seed', [False, True])
def test_decode_hard_seed(tensor_seed):
    model = model_at()
    x = torch.randn(1, 9, 32)
    direction = torch.tensor([[True, False]])
    generator = torch.Generator().manual_seed(7)
    before = generator.get_state()
    seed = torch.tensor(123, dtype=torch.int32) if tensor_seed else 123
    from flash_dism.module import _salt_hard_seed
    expected = torch.rand(1, 2, 9, generator=torch.Generator().manual_seed(_salt_hard_seed(123, 0))) < .4
    a = model(x, direction=direction, hard_prob=.4, hard_seed=seed, generator=generator)[0]
    b = model(x, direction=direction, hard=expected)[0]
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert torch.equal(before, generator.get_state())


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA')
def test_cuda_prefill_soft_agreement():
    model = model_at('cuda')
    x = torch.randn(1, 256, 32, device='cuda')
    direction = torch.tensor([[True, False]], device='cuda')
    with torch.autocast('cuda', dtype=torch.bfloat16):
        cuda = model(x, direction=direction, hard_prob=0.)[0]
        ref = model(x, direction=direction, hard_prob=0., past_key_values=Cache(), use_cache=True)[0]
    cosine = F.cosine_similarity(cuda.float().flatten(), ref.float().flatten(), dim=0)
    assert cosine > .999
    assert (cuda.float()-ref.float()).norm()/ref.float().norm() < .03


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA')
def test_training_padding_mask():
    model = model_at('cuda').train()
    x = torch.randn(2, 512, 32, device='cuda', requires_grad=True)
    mask = torch.ones(2, 512, device='cuda', dtype=torch.bool)
    mask[0, :256] = False
    direction = torch.ones(2, 2, device='cuda', dtype=torch.bool)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        output = model(x, attention_mask=mask, direction=direction, hard_prob=0.)[0]
        a = model(x[:1, 256:], direction=direction[:1], hard_prob=0.)[0]
        b = model(x[1:], direction=direction[:1], hard_prob=0.)[0]
    assert torch.count_nonzero(output[0, :256]) == 0
    torch.testing.assert_close(output[:1, 256:], a, atol=.005, rtol=.02)
    torch.testing.assert_close(output[1:], b, atol=.005, rtol=.02)
    output.float().square().sum().backward()
    assert torch.count_nonzero(x.grad[0, :256]) == 0
