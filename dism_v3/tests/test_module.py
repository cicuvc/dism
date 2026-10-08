import pytest
import torch
from flash_dism import DismAttention

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_vocab_initialization(monkeypatch):
    gaussian_samples = []
    original = torch.randn

    def record_sample(*args, **kwargs):
        result = original(*args, **kwargs)
        gaussian_samples.append(result.clone())
        return result

    monkeypatch.setattr(torch, "randn", record_sample)
    model = DismAttention(64, 2, head_dim=32, value_dim=32,
                          readout_dim=16, vocab_size=64)
    assert len(gaussian_samples) == 2
    for parameter, sample in zip((model.q_vocab, model.k_vocab), gaussian_samples):
        torch.testing.assert_close(parameter, torch.nn.functional.silu(sample), atol=0, rtol=0)
        assert parameter.is_leaf and parameter.dtype == torch.float32
        assert parameter._no_weight_decay


def test_vocab_has_no_runtime_activation(monkeypatch):
    import flash_dism
    model = DismAttention(64, 2, head_dim=32, value_dim=32,
                          readout_dim=16, vocab_size=64).cuda()
    original = flash_dism.voc_dism
    seen = []

    def check_codebooks(q, k, sq, sk, v, eq, ek, tau, **kwargs):
        # Passing the leaf itself also excludes a hidden activation Jacobian.
        assert eq is model.q_vocab and ek is model.k_vocab
        seen.append(True)
        return original(q, k, sq, sk, v, eq, ek, tau, **kwargs)

    monkeypatch.setattr(flash_dism, "voc_dism", check_codebooks)
    x = torch.randn(1, 256, 64, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output, _, _ = model(x, hard_prob=0.)
    output.float().square().mean().backward()
    assert seen == [True]
    for parameter in (model.q_vocab, model.k_vocab):
        assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0


@pytest.mark.parametrize("readout,dimension,value", [
    (r, d, v) for r in (16, 32) for d in (32, 64) for v in (32, 64)])
@pytest.mark.parametrize('readout_l2_norm', [False, True])
def test_module_gradients(readout, dimension, value, readout_l2_norm):
    torch.manual_seed(91)
    model = DismAttention(64, 2, head_dim=dimension, value_dim=value,
                          readout_dim=readout, vocab_size=64, readout_l2_norm=readout_l2_norm).cuda()
    x = torch.randn(1, 256, 64, device="cuda", requires_grad=True)
    direction = torch.tensor([[True, False]], device="cuda")
    hard = torch.rand(1, 2, 256, device="cuda") < .5
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output, attention, cache = model(x, direction=direction, hard=hard)
    assert attention is None and cache is None
    assert output.shape == x.shape and output.dtype == torch.bfloat16
    output.float().square().mean().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize("readout,dimension,value", [
    (r, d, v) for r in (16, 32) for d in (32, 64) for v in (32, 64)])
@pytest.mark.parametrize('readout_l2_norm', [False, True])
def test_module_packed_boundaries_and_gradients(readout, dimension, value, readout_l2_norm):
    torch.manual_seed(7)
    model = DismAttention(64, 2, head_dim=dimension, value_dim=value,
                          readout_dim=readout, vocab_size=64, readout_l2_norm=readout_l2_norm).cuda()
    x = torch.randn(1, 768, 64, device="cuda", requires_grad=True)
    direction = torch.tensor([[False, True]], device="cuda")
    hard = torch.rand(1, 2, 768, device="cuda") < .4
    boundaries = torch.tensor([0, 256, 256, 768], device="cuda", dtype=torch.int32)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        packed, _, _ = model(x, direction=direction, hard=hard, cu_seqlens=boundaries)
        separate = torch.cat([model(x[:, i:j], direction=direction,
                                    hard=hard[:, :, i:j])[0] for i, j in ((0, 256), (256, 768))], dim=1)
    torch.testing.assert_close(packed, separate, atol=.005, rtol=.02)
    packed.float().square().mean().backward()
    packed_dx = x.grad.clone()
    packed_grads = {name: p.grad.clone() for name, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    x.grad = None
    separate.float().square().mean().backward()
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter.grad, packed_grads[name], atol=3e-4, rtol=.03)
    torch.testing.assert_close(x.grad, packed_dx, atol=3e-4, rtol=.03)


def test_module_varlen_validation_and_isolation():
    model = DismAttention(64, 2, head_dim=32, value_dim=32,
                          readout_dim=16, vocab_size=64).cuda().eval()
    x = torch.randn(1, 512, 64, device="cuda", requires_grad=True)
    boundaries = torch.tensor([0, 256, 512], dtype=torch.int32)  # CPU boundaries accepted
    direction = torch.ones(1, 2, device="cuda", dtype=torch.bool)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output, _, _ = model(x, cu_seqlens=boundaries, max_seqlen=512, direction=direction)
        changed = x.detach().clone()
        changed[:, :256] *= 10
        other, _, _ = model(changed, cu_seqlens=boundaries, direction=direction)
    torch.testing.assert_close(output[:, 256:], other[:, 256:], atol=0, rtol=0)
    output[:, 256:].float().square().sum().backward()
    assert torch.count_nonzero(x.grad[:, :256]) == 0
    with pytest.raises(ValueError, match="longest document"):
        model(x, cu_seqlens=boundaries, max_seqlen=128)
    with pytest.raises(RuntimeError, match="256-token aligned"):
        model(x, cu_seqlens=torch.tensor([0, 255, 512], dtype=torch.int32))


def test_module_generator_replay():
    model = DismAttention(64, 2, head_dim=32, value_dim=32,
                          readout_dim=16, vocab_size=64).cuda()
    x = torch.randn(1, 256, 64, device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(21)
    state = generator.get_state()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        first = model(x, hard_prob=.5, generator=generator)[0]
        generator.set_state(state)
        second = model(x, hard_prob=.5, generator=generator)[0]
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    with pytest.raises(ValueError, match="256-aligned"):
        model(x[:, :255])


@pytest.mark.parametrize('packed', [False, True])
@pytest.mark.parametrize('tensor_seed', [False, True])
def test_explicit_hard_seed(packed, tensor_seed, monkeypatch):
    import flash_dism
    from flash_dism.kernels.random_bool import triton_rand_bool
    model = DismAttention(64, 2, head_dim=32, value_dim=32,
                          readout_dim=16, vocab_size=64).cuda()
    x = torch.randn(1, 512, 64, device='cuda')
    seed = torch.tensor(91, device='cuda', dtype=torch.int64) if tensor_seed else 91
    expected = triton_rand_bool((1, 2, 512), .4, device=x.device, seed=seed)
    original = flash_dism.voc_dism
    captured = []
    def capture(*args, **kwargs):
        captured.append(kwargs['hard'].clone())
        return original(*args, **kwargs)
    monkeypatch.setattr(flash_dism, 'voc_dism', capture)
    direction = torch.ones(1, 2, device='cuda', dtype=torch.bool)
    generator = torch.Generator(device='cuda').manual_seed(13)
    before = generator.get_state()
    kwargs = dict(direction=direction, hard_prob=.4, hard_seed=seed, generator=generator)
    if packed:
        kwargs['cu_seqlens'] = torch.tensor([0, 256, 512], device='cuda', dtype=torch.int32)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        a = model(x, **kwargs)[0]
        b = model(x, **kwargs)[0]
        model(x, **kwargs, hard=~expected)
    torch.testing.assert_close(captured[0], expected)
    torch.testing.assert_close(captured[1], expected)
    torch.testing.assert_close(captured[2], ~expected)
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert torch.equal(before, generator.get_state())


def test_layer_seed_salt(monkeypatch):
    import flash_dism
    from flash_dism.module import _salt_hard_seed
    for seed in (0, 123, -1, -(1 << 63), (1 << 63)-1):
        for layer in (0, 1, 7):
            value = _salt_hard_seed(seed, layer)
            assert value == _salt_hard_seed(torch.tensor(seed, device='cuda'), layer).item()
    model = DismAttention(64, 2, head_dim=32, value_dim=32,
                          readout_dim=16, vocab_size=64, layer_idx=0).cuda()
    masks = []
    original = flash_dism.voc_dism
    def capture(*args, **kwargs):
        masks.append(kwargs['hard'].clone())
        return original(*args, **kwargs)
    monkeypatch.setattr(flash_dism, 'voc_dism', capture)
    x = torch.randn(1, 256, 64, device='cuda')
    direction = torch.ones(1, 2, dtype=torch.bool, device='cuda')
    with torch.autocast('cuda', dtype=torch.bfloat16):
        for layer in (0, 1, 0):
            model.layer_idx = layer
            model(x, hard_prob=.5, hard_seed=123, direction=direction)
    assert torch.equal(masks[0], masks[2])
    assert not torch.equal(masks[0], masks[1])
