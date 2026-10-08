import pytest
import torch
from nanochat.models import ModelSpec, build_model


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA')
def test_window_document_boundary_and_causality():
    torch.manual_seed(55)
    config = dict(sequence_len=512, vocab_size=128, n_layer=1, n_head=2,
                  n_embd=64, intermediate_size=128, head_dim=32, window_size=128)
    model = build_model(ModelSpec('swa_control', 1, config)).cuda().eval()
    model.init_weights()
    x = torch.randint(128, (1, 512), device='cuda')
    cu = torch.tensor([0, 256, 512], device='cuda', dtype=torch.int32)
    with torch.no_grad():
        original = model(x, cu_seqlens=cu)
        changed = x.clone()
        changed[:, :128] = (changed[:, :128]+1) % 128
        output = model(changed, cu_seqlens=cu)
        # At row255 the window starts at128. A single layer cannot see earlier tokens.
        torch.testing.assert_close(output[:, 255:], original[:, 255:], atol=0, rtol=0)
        assert not torch.equal(output[:, :128], original[:, :128])
        changed = x.clone()
        changed[:, 200:] = (changed[:, 200:]+1) % 128
        output = model(changed, cu_seqlens=cu)
        torch.testing.assert_close(output[:, :200], original[:, :200], atol=0, rtol=0)
        # Packed document uses fresh rotary positions and no cross-document keys.
        single = model(x[:, 256:], cu_seqlens=cu[:2])
        torch.testing.assert_close(single, original[:, 256:], atol=.02, rtol=.02)
    assert model.embedding.weight is not model.lm_head.weight
    assert not any('vocab' in name or 'conv' in name for name, _ in model.named_parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA')
def test_context_sized_window_equals_full_attention():
    from flash_attn import flash_attn_varlen_func
    from flash_dism.hybrid import _compiled_varlen_swa
    torch.manual_seed(59)
    inputs = [torch.randn(512, 2, 64, device='cuda', dtype=torch.bfloat16,
                          requires_grad=True) for _ in range(3)]
    cu = torch.tensor([0, 256, 512], device='cuda', dtype=torch.int32)
    actual = _compiled_varlen_swa(*inputs, cu, 2048, 2048)
    expected = flash_attn_varlen_func(*inputs, cu, cu, 2048, 2048,
                                     causal=True, window_size=(-1, -1))
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    grad = torch.randn_like(actual)
    ga = torch.autograd.grad(actual, inputs, grad, retain_graph=True)
    gb = torch.autograd.grad(expected, inputs, grad)
    for a, b in zip(ga, gb):
        torch.testing.assert_close(a, b, atol=.01, rtol=.01)
