import pytest
import torch
from torch.nn import functional as F

from dism_v2.softcap_cross_entropy import softcap_cross_entropy, FusedSoftcapCrossEntropyLoss

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('dtype', (torch.float32, torch.bfloat16, torch.float16))
@pytest.mark.parametrize('reduction', ('none', 'sum', 'mean'))
@pytest.mark.parametrize('rows,vocab', ((1, 1), (7, 33), (5, 4096), (5, 4097), (5, 50257), (2, 65537), (1031, 65)))
def test_torch_reference(dtype, reduction, rows, vocab):
    torch.manual_seed(41)
    # Nontrivial row stride: no implicit full-logit contiguous copy is allowed.
    backing = torch.randn(rows, vocab + 16, device='cuda', dtype=dtype) * 4
    x = backing[:, :vocab].detach().requires_grad_()
    original = x.detach().clone()
    labels = torch.randint(vocab, (rows * 2,), device='cuda')
    y = labels[::2]
    if rows > 1:
        y[0] = -100
    ref = x.detach().double().requires_grad_()
    expected = F.cross_entropy(10 * torch.tanh(ref / 10), y, reduction=reduction)
    actual = softcap_cross_entropy(x, y, 10., reduction=reduction)
    torch.testing.assert_close(actual.double(), expected, atol=3e-5, rtol=3e-6)
    upstream = torch.randn(rows * 2, device='cuda')[::2] if reduction == 'none' else torch.tensor(.37, device='cuda')
    actual.backward(upstream)
    expected.backward(upstream.double())
    tolerance = {torch.float32: (2e-6, 3e-5), torch.float16: (3e-5, .003), torch.bfloat16: (2e-4, .015)}[dtype]
    torch.testing.assert_close(x.grad.float(), ref.grad.float(), atol=tolerance[0], rtol=tolerance[1])
    torch.testing.assert_close(x.detach(), original, atol=0, rtol=0)


@pytest.mark.parametrize('cap', (.25, 1., 30.))
def test_saturation_and_nonunit_upstream(cap):
    x = torch.tensor([[-10000., -40., -3., 0., 3., 40., 10000.]], device='cuda', requires_grad=True)
    y = torch.tensor([2], device='cuda', dtype=torch.int32)
    ref = x.detach().double().requires_grad_()
    actual = FusedSoftcapCrossEntropyLoss(cap)(x, y)
    expected = F.cross_entropy(cap * torch.tanh(ref / cap), y.long())
    (actual * -3.25).backward(); (expected * -3.25).backward()
    torch.testing.assert_close(actual.double(), expected, atol=1e-5, rtol=2e-6)
    torch.testing.assert_close(x.grad.double(), ref.grad, atol=2e-6, rtol=3e-5)
    assert x.grad[0, 0] == 0 and x.grad[0, -1] == 0


@pytest.mark.parametrize('reduction', ('none', 'sum', 'mean'))
@pytest.mark.parametrize('rows', (0, 3))
def test_empty_and_all_ignored(reduction, rows):
    x = torch.randn(rows, 19, device='cuda', requires_grad=True)
    y = torch.full((rows,), 3, device='cuda')
    out = softcap_cross_entropy(x, y, 30, ignore_index=3, reduction=reduction)
    if reduction == 'mean':
        assert torch.isnan(out)
    else:
        assert torch.count_nonzero(out) == 0
    out.backward(torch.ones_like(out))
    assert torch.count_nonzero(x.grad) == 0


def test_invalid_labels_are_safe_and_visible():
    x = torch.randn(2, 33, device='cuda', requires_grad=True)
    out = softcap_cross_entropy(x, torch.tensor([-2, 33], device='cuda'), 10., reduction='none')
    assert torch.isnan(out).all()
    out.sum().backward()
    assert torch.isnan(x.grad).all()


def test_contract():
    x = torch.randn(3, 19, device='cuda')
    y = torch.zeros(3, device='cuda', dtype=torch.long)
    for cap in (0, -1, float('nan'), float('inf'), True, 1e-40, 1e40, torch.tensor(3.)):
        with pytest.raises(ValueError):
            softcap_cross_entropy(x, y, cap)
    with pytest.raises(ValueError):
        softcap_cross_entropy(x, y, 3., reduction='bad')
    with pytest.raises(ValueError):
        softcap_cross_entropy(x[:, ::2], y, 3.)
    with pytest.raises(TypeError):
        softcap_cross_entropy(x, y.float(), 3.)


@pytest.mark.parametrize('dtype', (torch.float32, torch.bfloat16, torch.float16))
@pytest.mark.parametrize('scale', (.25, 1., 4., 16., 64., 256.))
@pytest.mark.parametrize('reduction', ('none', 'mean'))
def test_gpt2_cap30(dtype, scale, reduction):
    torch.manual_seed(825)
    x = (torch.randn(17, 50257, device='cuda', dtype=dtype) * scale).requires_grad_()
    y = torch.randint(50257, (17,), device='cuda')
    y[3] = -100
    ref = x.detach().double().requires_grad_()
    expected = F.cross_entropy(30 * torch.tanh(ref / 30), y, reduction=reduction)
    actual = softcap_cross_entropy(x, y, 30., reduction=reduction)
    torch.testing.assert_close(actual.double(), expected, atol=3e-5, rtol=3e-6)
    grad = torch.randn_like(actual)
    actual.backward(grad); expected.backward(grad.double())
    atol, rtol = {torch.float32: (2e-6, 3e-5), torch.float16: (3e-5, .003), torch.bfloat16: (2e-4, .015)}[dtype]
    torch.testing.assert_close(x.grad.float(), ref.grad.float(), atol=atol, rtol=rtol)


def test_gpt2_codegen():
    import re
    from triton.tools.disasm import get_sass
    from dism_v2.softcap_cross_entropy import _row_forward, _merge_forward, _backward
    x = torch.randn(1, 50257, device='cuda', dtype=torch.bfloat16)
    y = torch.zeros(1, device='cuda', dtype=torch.long)
    lse = torch.empty(1, device='cuda'); loss = torch.empty_like(lse)
    partial = torch.empty(1, 13, device='cuda'); one = torch.ones(1, device='cuda')
    dx = torch.empty_like(x)
    compiled = [
        _row_forward[(1, 13)](x, y, lse, loss, partial, x.stride(0), 1, 50257, 30., -100, 13, 4096, num_warps=8),
        _merge_forward[(1,)](x, y, partial, lse, loss, x.stride(0), 1, 1, 50257, 30., -100, 13, 16, 32, num_warps=4),
        _backward[(1, 50)](x, y, lse, one, one, dx, x.stride(0), 1, 0, 50257, 30., -100, False, 1024, num_warps=4),
    ]
    for kernel in compiled:
        assert 'tanh.approx.f32' in kernel.asm['ptx']
        assert not re.search(r'\bCALL(?:\.|\s)', get_sass(kernel.asm['cubin']))
