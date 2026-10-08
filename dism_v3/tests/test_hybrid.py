import pytest
import torch
from torch.nn import functional as F
from flash_dism import DismSwaAttention, DismAttention, DismCache


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA')
@pytest.mark.parametrize('value_dim', [32, 64])
@pytest.mark.parametrize('packed', [False, True])
def test_hybrid_gradients_and_shared_v(value_dim, packed):
    torch.manual_seed(51)
    model = DismSwaAttention(64, 2, head_dim=32, value_dim=value_dim,
                             readout_dim=16, vocab_size=64, window_size=128).cuda()
    x = torch.randn(1, 768 if packed else 256, 64, device='cuda', requires_grad=True)
    kwargs = dict(hard_prob=.5, hard_seed=93, direction=torch.tensor([[True, False]], device='cuda'))
    if packed:
        kwargs['cu_seqlens'] = torch.tensor([0, 256, 256, 768], device='cuda', dtype=torch.int32)
    calls = []
    handle = model.v_conv.register_forward_hook(lambda m, args, result: calls.append(result))
    with torch.autocast('cuda', dtype=torch.bfloat16):
        output = model(x, **kwargs)[0]
    handle.remove()
    assert len(calls) == 1 and model.norm.weight.shape == (value_dim,)
    output.float().square().mean().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum() > 0, name
    assert torch.isfinite(x.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA')
def test_flash_swa_vs_torch_window_and_v_gradient():
    torch.manual_seed(19)
    model = DismSwaAttention(64, 2, value_dim=32, window_size=8).cuda()
    x = torch.randn(1, 32, 64, device='cuda')
    v = torch.randn(1, 32, 2, 32, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        output = model._swa_cuda(x, v)
        cos, sin = model._rotary_tables(32, x.device, dimension=32)
        q = model.swa_q_proj(x, cos, sin)
        k = model.swa_k_proj(x, cos, sin)
    scores = torch.einsum('bqhd,bkhd->bhqk', q.float(), k.float()) / 32**.5
    pos = torch.arange(32, device='cuda')
    valid = (pos[None, :] <= pos[:, None]) & (pos[None, :] > pos[:, None]-8)
    probs = scores.masked_fill(~valid, -torch.inf).softmax(-1)
    ref_v = v.detach().float().requires_grad_()
    reference = torch.einsum('bhqk,bkhd->bqhd', probs, ref_v)
    torch.testing.assert_close(output.float(), reference, atol=.01, rtol=.02)
    grad = torch.randn_like(output)
    actual_grad = torch.autograd.grad(output, v, grad)[0]
    reference_grad = torch.autograd.grad(reference, ref_v, grad.float())[0]
    torch.testing.assert_close(actual_grad.float(), reference_grad, atol=.02, rtol=.02)


@pytest.mark.parametrize('window', [1, 4, 128])
def test_hybrid_decode_chunks(window):
    torch.manual_seed(8)
    model = DismSwaAttention(32, 2, head_dim=32, value_dim=32, readout_dim=16,
                             vocab_size=16, window_size=window, layer_idx=0).double().eval()
    x = torch.randn(2, 11, 32, dtype=torch.float64)
    direction = torch.tensor([[True, False], [False, True]])
    hard = torch.rand(2, 2, 11) < .5
    expected = model(x, direction=direction, hard=hard)[0]
    parts, cache = [], DismCache()
    for start, end in ((0, 5), (5, 6), (6, 11)):
        parts.append(model(x[:, start:end], direction=direction, hard=hard[:, :, start:end],
                           past_key_values=cache, use_cache=True)[0])
        assert cache[0]['recurrent_state'][8].shape[1] == min(end, window-1)
    torch.testing.assert_close(torch.cat(parts, 1), expected, atol=1e-10, rtol=1e-10)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA')
def test_per_head_output_norm():
    model = DismAttention(64, 2, value_dim=32).cuda()
    values = torch.randn(1, 3, 2, 32, device='cuda')
    gate = torch.randn_like(values)
    values[:, :, 1] *= 100
    output = model.norm(values, gate)
    ref = F.rms_norm(values, (32,), model.norm.weight, model.norm.eps) * F.silu(gate)
    torch.testing.assert_close(output, ref, atol=2e-6, rtol=2e-5)
    changed = values.clone()
    changed[:, :, 1] *= 10
    torch.testing.assert_close(model.norm(changed, gate)[:, :, 0], output[:, :, 0], atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA')
def test_hybrid_varlen_matches_independent_documents():
    torch.manual_seed(13)
    model = DismSwaAttention(32, 2, head_dim=32, value_dim=32, readout_dim=16,
                             vocab_size=16, window_size=128).cuda()
    x = torch.randn(1, 768, 32, device='cuda', requires_grad=True)
    direction = torch.tensor([[False, True]], device='cuda')
    hard = torch.rand(1, 2, 768, device='cuda') < .5
    cu = torch.tensor([0, 256, 256, 768], dtype=torch.int32, device='cuda')
    with torch.autocast('cuda', dtype=torch.bfloat16):
        packed = model(x, cu_seqlens=cu, direction=direction, hard=hard)[0]
        separate = torch.cat([model(x[:, a:b], direction=direction, hard=hard[:, :, a:b])[0]
                              for a, b in ((0, 256), (256, 768))], dim=1)
    torch.testing.assert_close(packed, separate, atol=.005, rtol=.02)
    packed.float().square().mean().backward()
    grads = {name: p.grad.clone() for name, p in model.named_parameters()}
    dx = x.grad.clone()
    model.zero_grad(set_to_none=True)
    x.grad = None
    separate.float().square().mean().backward()
    torch.testing.assert_close(x.grad, dx, atol=3e-4, rtol=.03)
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter.grad, grads[name], atol=3e-4, rtol=.03)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA')
def test_hybrid_cuda_torch_prefill():
    torch.manual_seed(31)
    model = DismSwaAttention(32, 2, head_dim=32, value_dim=32, readout_dim=16,
                             vocab_size=16, layer_idx=0).cuda().eval()
    x = torch.randn(1, 256, 32, device='cuda')
    direction = torch.tensor([[False, True]], device='cuda')
    with torch.autocast('cuda', dtype=torch.bfloat16):
        actual = model(x, hard_prob=0., direction=direction)[0]
        reference = model(x, hard_prob=0., direction=direction,
                          past_key_values=DismCache(), use_cache=True)[0]
    assert F.cosine_similarity(actual.float().flatten(), reference.float().flatten(), dim=0) > .999
    assert (actual.float()-reference.float()).norm()/reference.float().norm() < .03
