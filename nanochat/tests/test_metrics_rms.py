import pytest
import torch
from nanochat.metrics import snapshot_parameters, update_rms_max_per_group, _group_key

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16, torch.float16])
def test_update_rms(device, dtype):
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([torch.nn.Linear(17, 13), torch.nn.Linear(13, 7)])
    model.alias = model.layers[0].weight
    model.frozen = torch.nn.Parameter(torch.ones(2), requires_grad=False)
    model.to(device=device, dtype=dtype)
    state_keys = tuple(model.state_dict())
    rng = torch.get_rng_state()
    snapshot = snapshot_parameters(model)
    assert torch.equal(rng, torch.get_rng_state())
    assert 'frozen' not in snapshot
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.0123)
    expected = {}
    for name, p in model.named_parameters():
        if name in snapshot:
            rms = (p.detach() - snapshot[name]).float().square().mean().sqrt()
            expected.setdefault(_group_key(name), []).append(rms)
    expected = {k: torch.stack(v).max().item() for k, v in expected.items()}
    actual = update_rms_max_per_group(model, snapshot)
    assert actual.keys() == expected.keys()
    for k in actual:
        assert actual[k] == pytest.approx(expected[k], rel=1e-6, abs=1e-8)
    assert tuple(model.state_dict()) == state_keys
    assert all(p.grad is None for p in model.parameters())
    assert update_rms_max_per_group(model, {}) == {}


@pytest.mark.parametrize('shape', [(2, 37, 64), (1, 32768, 384), (3, 17, 13)])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_residual(shape, dtype):
    from nanochat.metrics_kernels import residual_rms_max
    x = torch.randn(shape, device='cuda', dtype=dtype)
    expected = x.float().square().mean(-1).sqrt().max()
    torch.testing.assert_close(residual_rms_max(x), expected, rtol=1e-6, atol=1e-6)
    x.flatten()[0] = float('nan')
    assert residual_rms_max(x).isnan()


def test_residual_compile():
    from nanochat.metrics_kernels import residual_rms_max
    fn = torch.compile(residual_rms_max, fullgraph=True)
    x = torch.randn(1, 100, 384, device='cuda')
    torch.testing.assert_close(fn(x), x.square().mean(-1).sqrt().max())


def test_update_noncontiguous_and_nonfinite():
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.randn(13, 17, device='cuda').T)
    snap = snapshot_parameters(model)
    with torch.no_grad():
        model.weight.add_(0.1)
    expected = (model.weight.detach() - snap['weight']).square().mean().sqrt().item()
    assert update_rms_max_per_group(model, snap)['weight'] == pytest.approx(expected, rel=1e-6)
    with torch.no_grad():
        model.weight[0, 0] = float('nan')
    import math
    assert math.isnan(update_rms_max_per_group(model, snap)['weight'])
    empty = torch.nn.Module()
    assert snapshot_parameters(empty) == {}
    empty.p = torch.nn.Parameter(torch.empty(0, device='cuda'))
    assert math.isnan(update_rms_max_per_group(empty, snapshot_parameters(empty))['p'])


def test_compiled_hooks_share_graph_and_keep_gradients():
    from nanochat.metrics import register_residual_rms_probe, remove_probe
    from torch._dynamo.testing import CompileCounter
    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp_norm = torch.nn.Identity()
            self.weight = torch.nn.Parameter(torch.ones((), device='cuda'))

        def forward(self, x):
            return self.mlp_norm(x * self.weight)

    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([Block() for _ in range(12)])
    handles, buffer = register_residual_rms_probe(model)
    counter = CompileCounter()
    try:
        for i, layer in enumerate(model.layers):
            x = torch.full((1, 37, 64), float(i + 1), device='cuda', requires_grad=True)
            out = torch.compile(layer, backend=counter, fullgraph=True)(x)
            out.sum().backward()
            torch.testing.assert_close(x.grad, torch.ones_like(x))
        torch.testing.assert_close(buffer, torch.arange(1, 13, device='cuda').float())
        assert counter.frame_count == 1
    finally:
        remove_probe(handles)
