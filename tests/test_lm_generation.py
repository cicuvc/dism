"""Inference-specific state and sampling checks; no checkpoint required."""
import pytest
import torch

from dism_v2.generate_lm import cap_logits, sample


def test_sampling():
    logits = torch.tensor([[1., 4., -2.]])
    rng = torch.Generator().manual_seed(7)
    assert sample(logits, 0., .9, rng).item() == 1
    assert sample(logits, 1., .01, rng).item() == 1
    torch.testing.assert_close(cap_logits(logits, 30.), 30 * (logits / 30).tanh())
    with pytest.raises(FloatingPointError):
        sample(logits * float('nan'), 0., 1., rng)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_shortconv_cached_matches_sequence():
    from fla.modules import ShortConvolution
    torch.manual_seed(123)
    conv = ShortConvolution(256, 4, activation='swish').cuda().eval()
    x = torch.randn(1, 129, 256, device='cuda', dtype=torch.bfloat16)
    with torch.inference_mode():
        expected = conv(x)[0]
        cache, parts = None, []
        for i in range(x.shape[1]):
            y, cache = conv(x[:, i:i+1], cache=cache, output_final_state=True)
            parts.append(y)
        torch.testing.assert_close(torch.cat(parts, 1), expected, atol=.016, rtol=.016)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_offset_rope_and_swa_window():
    from flash_attn import flash_attn_func
    from dism_v2.lm_model import LMConfig, SlidingAttention
    torch.manual_seed(321)
    s = SlidingAttention(LMConfig()).cuda().eval()
    q, k, v = [torch.randn(1, 137, 4, 64, device='cuda', dtype=torch.bfloat16) for _ in range(3)]
    with torch.inference_mode():
        qr, kr = s.rope(q), s.rope(k)
        expected = flash_attn_func(qr, kr, v, causal=True, window_size=(127, 0))
        for i in (0, 1, 126, 127, 128, 136):
            a, b = q[:, i:i+1, :, 0::2], q[:, i:i+1, :, 1::2]
            co, si = s.cos[i].to(q.dtype), s.sin[i].to(q.dtype)
            qi = torch.stack((a*co-b*si, a*si+b*co), -1).flatten(-2)
            torch.testing.assert_close(qi, qr[:, i:i+1], atol=0, rtol=0)
            start = max(0, i-127)
            got = flash_attn_func(qi, kr[:, start:i+1], v[:, start:i+1], causal=False)
            torch.testing.assert_close(got, expected[:, i:i+1], atol=.016, rtol=.016)
