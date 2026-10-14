import copy

import pytest
import torch
import torch.nn.functional as F

from nanochat.models.dism.conv import LinearShortConvSiLU, packed_conv_sequence_ids

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('bounds', [[0, 0, 7, 7, 19, 32, 32], [0, 32]])
def test_sequence_ids_empty_documents_and_inplace_boundary_update(bounds):
    cu = torch.tensor(bounds, dtype=torch.int32, device='cuda')
    def expected(values):
        return [i for i, (s, e) in enumerate(zip(values, values[1:])) for _ in range(s, e)]
    assert packed_conv_sequence_ids(cu, 32, cu.device).tolist() == [expected(bounds)]
    changed = [0] * (len(bounds)-1) + [32]
    cu.copy_(torch.tensor(changed, dtype=cu.dtype, device=cu.device))
    assert packed_conv_sequence_ids(cu, 32, cu.device).tolist() == [expected(changed)]


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('packed', [False, True])
def test_direct_cuda_matches_fla_forward_and_gradients(dtype, packed):
    from fla.modules import ShortConvolution
    torch.manual_seed(172)
    layer = LinearShortConvSiLU(32, 64, 4).cuda()
    with torch.no_grad():
        layer.weight.normal_(std=.1)
    old = ShortConvolution(64, 4, activation='silu', backend='cuda').cuda()
    old.load_state_dict(layer.conv.state_dict(), strict=True)
    x = torch.randn(1 if packed else 2, 256, 32, device='cuda', requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    wr = layer.weight.detach().clone().requires_grad_()
    cu = torch.tensor([0, 17, 17, 64, 256], dtype=torch.int32, device='cuda') if packed else None
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
        actual = layer(x, cu_seqlens=cu)
        expected = old(F.linear(xr, wr), cu_seqlens=cu)[0]
    dout = torch.randn_like(actual)
    ga = torch.autograd.grad(actual, (x, layer.weight, layer.conv.weight), dout)
    ge = torch.autograd.grad(expected, (xr, wr, old.weight), dout)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for a, e in zip(ga, ge):
        torch.testing.assert_close(a, e, rtol=1e-5, atol=2e-5)


def test_no_cross_document_output_or_gradient_leak():
    layer = LinearShortConvSiLU(32, 64, 4).cuda()
    with torch.no_grad():
        layer.weight.normal_(std=.1)
    x = torch.randn(1, 64, 32, device='cuda', requires_grad=True)
    cu = torch.tensor([0, 32, 32, 64], dtype=torch.int32, device='cuda')
    original = layer(x, cu_seqlens=cu)
    changed = x.detach().clone()
    changed[:, :32] += 100
    modified = layer(changed, cu_seqlens=cu)
    torch.testing.assert_close(original[:, 32:], modified[:, 32:], rtol=0, atol=0)
    grad, = torch.autograd.grad(original[:, 32:].sum(), (x,))
    assert torch.count_nonzero(grad[:, :32]) == 0


def test_fullgraph_dynamic_boundaries_and_no_host_scalar_reads():
    from torch.fx.experimental import _config as shape_config
    layer = LinearShortConvSiLU(32, 64, 4).cuda()
    with torch.no_grad():
        layer.weight.normal_(std=.1)
    def fn(x, cu):
        seq = packed_conv_sequence_ids(cu, x.shape[1], x.device)
        return layer(x, cu_seqlens=cu, seq_idx=seq)
    with shape_config.patch(use_duck_shape=False):
        compiled = torch.compile(fn, dynamic=True, fullgraph=True)
        for bounds in ([0, 64, 256], [0, 0, 17, 256], [0, 256]):
            x = torch.randn(1, 256, 32, device='cuda', requires_grad=True)
            cu = torch.tensor(bounds, dtype=torch.int32, device='cuda')
            with torch.autocast('cuda', dtype=torch.bfloat16):
                actual = compiled(x, cu)
                expected = fn(x, cu)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            params = (x, layer.weight, layer.conv.weight)
            ga = torch.autograd.grad(actual.float().square().mean(), params)
            ge = torch.autograd.grad(expected.float().square().mean(), params)
            for a, e in zip(ga, ge):
                torch.testing.assert_close(a, e, rtol=1e-5, atol=2e-5)
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out = compiled(x, cu)
            out.float().square().mean().backward()
        names = {e.key for e in prof.key_averages()}
        assert 'aten::_local_scalar_dense' not in names
        assert 'aten::item' not in names


def test_checkpoint_parameter_schema_and_initialization_unchanged():
    from fla.modules import ShortConvolution
    torch.manual_seed(88)
    old = ShortConvolution(64, 4, activation='silu', backend='cuda')
    state = torch.get_rng_state()
    torch.manual_seed(88)
    layer = LinearShortConvSiLU(32, 64, 4)
    assert torch.equal(torch.get_rng_state(), state)
    assert list(layer.state_dict()) == ['weight', 'conv.weight']
    assert torch.equal(layer.conv.weight, old.weight)
    layer.load_state_dict(copy.deepcopy(layer.state_dict()), strict=True)


def test_model_reuses_one_sequence_map_across_all_convolutions(monkeypatch):
    import nanochat.models.dism.model as adapter
    from nanochat.models import ModelSpec, build_model
    config = dict(sequence_len=512, vocab_size=128, n_layer=4, n_head=2,
                  n_embd=64, intermediate_size=128, head_dim=32, value_dim=32,
                  readout_dim=16, qk_vocab_size=32, rear_half_dism=True,
                  value_residual=True, conv_impl='linear_fla', conv_backend='cuda')
    model = build_model(ModelSpec('dism_gdn_hybrid', 1, config)).cuda().eval()
    model.init_weights()
    ids = torch.randint(128, (2, 512), device='cuda')
    cu = torch.tensor([0, 256, 512, 1024], dtype=torch.int32, device='cuda')
    maps, uses = [], []
    original = adapter.packed_conv_sequence_ids
    def build(*args):
        value = original(*args)
        maps.append(value)
        return value
    original_conv = LinearShortConvSiLU.depthwise_silu
    def call(self, hidden, cu_seqlens=None, seq_idx=None):
        assert seq_idx is not None
        uses.append(seq_idx)
        return original_conv(self, hidden, cu_seqlens, seq_idx)
    monkeypatch.setattr(adapter, 'packed_conv_sequence_ids', build)
    monkeypatch.setattr(LinearShortConvSiLU, 'depthwise_silu', call)
    with torch.no_grad():
        actual = model(ids, cu_seqlens=cu)
    assert len(maps) == 1
    assert len(uses) == 19  # 2 pure*3 + 2 hybrid*6 + first-layer value residual
    assert all(value is maps[0] for value in uses)
    # Compare complete value-residual model traversal to the old FLA wrapper.
    from fla.modules.conv.cuda.ops import causal_conv1d_cuda
    def legacy(self, hidden, cu_seqlens=None, seq_idx=None):
        return causal_conv1d_cuda(hidden, self.conv.weight.squeeze(1),
                                 activation='silu', cu_seqlens=cu_seqlens)[0]
    monkeypatch.setattr(LinearShortConvSiLU, 'depthwise_silu', legacy)
    with torch.no_grad():
        expected = model(ids, cu_seqlens=cu)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
