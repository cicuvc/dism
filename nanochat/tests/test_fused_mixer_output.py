"""Output-tail fused BF16 projection boundaries and full-gradient checks."""
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from fla.modules import FusedRMSNormGated
from nanochat.models.dism.output_gate import LayerNormRawGate, RMSNormRawGate, SigmoidInputLinear
from flash_dism.kernels.mixer_output import fused_mixer_output, supported

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def make_case(tokens=65, width=128, rank=16, heads=2, dim=32, bias=True):
    torch.manual_seed(20261027)
    def random(*shape, scale=1., dtype=torch.float32):
        return (torch.randn(*shape, device='cuda', dtype=dtype) * scale).requires_grad_()
    return [random(1, tokens, width, dtype=torch.bfloat16),
            random(1, tokens, heads, dim, dtype=torch.bfloat16),
            random(rank, width, scale=.02), random(rank, scale=.1) if bias else None,
            random(heads * dim, rank, scale=.02), random(heads * dim, scale=.1) if bias else None,
            random(dim, scale=.1).detach().add_(1).requires_grad_()]


def reference(values, center, precise=False):
    x, raw, wd, bd, wu, bu, nw = values
    if precise:
        # Independent FP64 contractions, with the same BF16 projection operand,
        # projection output and sigmoid output boundaries as BF16 autocast.
        cast = lambda t: None if t is None else t.bfloat16().double()
        first = F.linear(x.double(), cast(wd), cast(bd)).bfloat16()
        g = first.double().sigmoid().bfloat16()
        gate = F.linear(g.double(), cast(wu), cast(bu)).bfloat16()
        value, weight = raw.double(), nw.double()
    else:
        gate = F.linear(F.linear(x, wd, bd).sigmoid(), wu, bu)
        value, weight = raw.float(), nw.float()
    if center:
        # Preserve the production explicit path's two raw.float() graph edges.
        # They independently round their raw-input gradients back to BF16.
        value = value - (raw.double() if precise else raw.float()).mean(-1, keepdim=True)
    normalized = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-5)
    return (normalized * weight * (gate.unflatten(-1, raw.shape[-2:]).to(value.dtype) * .53)).to(raw.dtype)


def check_direction(actual, expected, name):
    a, b = actual.detach().float().flatten(), expected.detach().float().flatten()
    assert torch.isfinite(a).all(), name
    relative = (a - b).norm() / b.norm().clamp_min(1e-12)
    cosine = F.cosine_similarity(a, b, dim=0)
    signed_bias = (a - b).mean().abs() / b.square().mean().sqrt().clamp_min(1e-12)
    assert relative < .012, (name, 'relative', relative.item())
    assert cosine > .9999, (name, 'cosine', cosine.item())
    assert signed_bias < .004, (name, 'bias', signed_bias.item())


@pytest.mark.parametrize('center', [False, True])
@pytest.mark.parametrize('shape', [(17, 70, 7, 2, 17), (65, 768, 96, 12, 64), (129, 1024, 128, 3, 128)])
def test_forward_all_gradients(center, shape):
    values = make_case(*shape)
    explicit = [v.detach().clone().requires_grad_() for v in values]
    precise = [v.detach().clone().requires_grad_() for v in values]
    with torch.autocast('cuda', dtype=torch.bfloat16):
        actual = fused_mixer_output(*values, .53, 1e-5, center)
        expected = reference(explicit, center)
    with torch.autocast('cuda', enabled=False):
        oracle = reference(precise, center, precise=True)
    # Non-contiguous upstream gradients exercise the backward copy boundary.
    grad = torch.randn((*actual.shape[:-1], actual.shape[-1] * 2), device='cuda', dtype=actual.dtype)[..., ::2]
    for result in [actual, expected, oracle]:
        result.backward(grad)
    for label, refs, result in [('explicit', explicit, expected), ('fp64', precise, oracle)]:
        check_direction(actual, result, label + '/forward')
        for index, (value, ref) in enumerate(zip(values, refs)):
            check_direction(value.grad, ref.grad, label + '/gradient/' + str(index))


@pytest.mark.parametrize('center', [False, True])
def test_graph_replay_and_zero_gradient(center):
    values = make_case(tokens=65, width=128, rank=16, heads=2, dim=32, bias=False)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    dy = torch.randn_like(values[1])
    with torch.cuda.stream(stream):
        for _ in range(3):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                output = fused_mixer_output(*values, .53, 1e-5, center)
            output.backward(dy)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for value in values:
                if value is not None:
                    value.grad.zero_()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                output = fused_mixer_output(*values, .53, 1e-5, center)
            output.backward(dy)
    torch.cuda.current_stream().wait_stream(stream)
    for seed in [9, 10]:
        torch.manual_seed(seed)
        with torch.no_grad():
            values[0].normal_()
            values[1].normal_()
        graph.replay()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            expected = reference(values, center)
        check_direction(output, expected, 'graph output')
    dy.zero_()
    graph.replay()
    torch.cuda.synchronize()
    for value in values:
        if value is not None:
            assert torch.count_nonzero(value.grad).item() == 0


def test_runtime_switch_fallback_and_parameter_identity():
    down = nn.Linear(128, 16).cuda()
    up = SigmoidInputLinear(nn.Linear(16, 64).cuda())
    original = FusedRMSNormGated(32, eps=1e-5).cuda()
    norm = LayerNormRawGate(original, .53)
    assert norm.weight is original.weight
    state_keys = list(norm.state_dict())
    x = torch.randn(1, 17, 128, device='cuda', dtype=torch.bfloat16)
    raw = torch.randn(1, 17, 2, 32, device='cuda', dtype=torch.bfloat16)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        a = norm.fused_output(raw, x, down, up)
        norm.enable_fused_output = False
        b = norm.fused_output(raw, x, down, up)
        check_direction(a, b, 'runtime switch')
        assert list(norm.state_dict()) == state_keys
        # Non-contiguous input follows the original expression exactly.
        norm.enable_fused_output = True
        noncontiguous = torch.randn(1, 17, 256, device='cuda', dtype=torch.bfloat16)[..., ::2]
        assert not supported(noncontiguous, raw, down.weight, down.bias, up.weight, up.bias, norm.weight)
        actual = norm.fused_output(raw, noncontiguous, down, up)
        expected = norm(raw, up(down(noncontiguous)).unflatten(-1, (2, 32)))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not supported(x.float(), raw, down.weight, down.bias, up.weight, up.bias, norm.weight)
    assert not supported(x, raw, down.weight, down.bias, up.weight, up.bias, norm.weight)


def test_bf16_parameters_without_autocast_and_first_order_only():
    values = [None if value is None else value.detach().bfloat16().requires_grad_()
              for value in make_case(tokens=17)]
    refs = [None if value is None else value.detach().clone().requires_grad_() for value in values]
    with torch.autocast('cuda', enabled=False):
        assert supported(*values)
        output = fused_mixer_output(*values, .53, 1e-5, True)
        expected = reference(refs, True)
    grad = torch.randn_like(output)
    output.backward(grad)
    expected.backward(grad)
    check_direction(output, expected, 'bf16 parameters forward')
    for value, ref in zip(values, refs):
        check_direction(value.grad, ref.grad, 'bf16 parameters gradient')
    with torch.autocast('cuda', dtype=torch.float16):
        assert not supported(*values)
    output = fused_mixer_output(*values, .53, 1e-5, True)
    first = torch.autograd.grad(output, values[0], grad, create_graph=True)[0]
    assert not first.requires_grad


def test_empty_and_large_dimensions_use_fallback():
    values = make_case(tokens=17)
    empty = [values[0][:, :0], values[1][:, :0], *values[2:]]
    with torch.autocast('cuda', dtype=torch.bfloat16):
        assert not supported(*empty)
        assert not supported(*make_case(tokens=17, rank=129))
        assert not supported(*make_case(tokens=17, dim=129))


def test_autotune_cache_miss_during_capture_uses_safe_fixed_config(monkeypatch):
    from flash_dism.kernels import mixer_output as kernels
    values = make_case(tokens=23, width=128, rank=16, heads=2, dim=32)
    dy = torch.randn_like(values[1])
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                output = fused_mixer_output(*values, .53, 1e-5, True)
            output.backward(dy)
    torch.cuda.current_stream().wait_stream(stream)
    def forbidden_benchmark(*args, **kwargs):
        raise AssertionError('autotuning is forbidden during CUDA Graph capture')
    tuners = [kernels._down_sigmoid, kernels._up_norm_gate, kernels._norm_gate_backward]
    for tuner in tuners:
        assert tuner.cache
        monkeypatch.setattr(tuner, 'cache', {})
        monkeypatch.setattr(tuner, '_do_bench', forbidden_benchmark)
    with torch.cuda.stream(stream):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for value in values:
                value.grad.zero_()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                output = fused_mixer_output(*values, .53, 1e-5, True)
            output.backward(dy)
    torch.cuda.current_stream().wait_stream(stream)
    graph.replay()
    torch.cuda.synchronize()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        expected = reference(values, True)
    check_direction(output, expected, 'cold autotune cache graph')
    for tuner in tuners:
        # Fixed fallback must not masquerade as a measured tuning winner.
        assert not tuner.cache
    for value in values:
        assert torch.isfinite(value.grad).all()
