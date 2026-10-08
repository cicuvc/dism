"""End-to-end compiler contracts, including dynamic document count and backward."""
import pytest
import torch
from torch._dynamo.testing import CompileCounterWithBackend
from flash_dism import DismAttention, DismSwaAttention
from flash_dism.kernels.dynamo_utils import mark_cu_seqlens_dynamic

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('hybrid', [False, True])
@pytest.mark.parametrize('packed', [False, True])
def test_inductor_forward_backward_no_recompile(hybrid, packed):
    _check_compiled(hybrid, packed)


@pytest.mark.parametrize('hybrid', [False, True])
@pytest.mark.parametrize('duck_shape', [False, True])
@pytest.mark.parametrize('heads,first', [(2, [0, 512, 1024]), (2, [0, 1024]),
                                       (4, [0, 256, 512, 1024])])
def test_varlen_without_explicit_dynamic_mark(hybrid, duck_shape, heads, first):
    from torch.fx.experimental import _config as shape_config
    with shape_config.patch(use_duck_shape=duck_shape):
        if duck_shape and len(first) == heads:
            # Record the compiler limitation, rather than silently permitting
            # a warmup recompile or claiming dynamic=True always suffices.
            with pytest.raises(torch._dynamo.exc.RecompileError, match='duck sizing'):
                _check_compiled(hybrid, True, heads=heads, first=first, mark=False)
        else:
            _check_compiled(hybrid, True, heads=heads, first=first, mark=False)


def _check_compiled(hybrid, packed, *, heads=2, first=None, mark=True):
    torch._dynamo.reset()
    torch.manual_seed(17)
    cls = DismSwaAttention if hybrid else DismAttention
    model = cls(64, heads, head_dim=32, value_dim=32, readout_dim=16,
                vocab_size=64).cuda().train()
    x = torch.randn(1, 1024, 64, device='cuda', requires_grad=True)
    hard = torch.rand(1, heads, 1024, device='cuda') < .5
    direction = (torch.arange(heads, device='cuda') % 2 == 0)[None]
    packs = [[0, 512, 1024], [0, 256, 512, 768, 1024],
             [0, 256, 1024], [0, 0, 256, 256, 1024], [0, 1024]] if packed else [None, None]
    if first is not None:
        packs = [first, *packs]
    counter = CompileCounterWithBackend('inductor')
    with torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True), \
            torch._inductor.config.patch(compile_threads=1):
        compiled = torch.compile(model, backend=counter, fullgraph=True, dynamic=True)
        for index, boundaries in enumerate(packs):
            cu = None if boundaries is None else torch.tensor(boundaries, device='cuda', dtype=torch.int32)
            if cu is not None and mark:
                mark_cu_seqlens_dynamic(cu)
            kwargs = dict(hard=hard, direction=direction, use_cache=False)
            if packed:
                kwargs.update(cu_seqlens=cu, max_seqlen=1024)
            model.zero_grad(set_to_none=True)
            x.grad = None
            with torch.autocast('cuda', dtype=torch.bfloat16):
                expected = model(x, **kwargs)[0]
            expected.float().square().mean().backward()
            gradients = [p.grad.clone() for p in model.parameters()]
            dx = x.grad.clone()
            model.zero_grad(set_to_none=True)
            x.grad = None
            # Unlike frame_count alone, this also rejects Python-only retracing.
            with torch._dynamo.config.patch(error_on_recompile=index > 0):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    actual = compiled(x, **kwargs)[0]
                actual.float().square().mean().backward()
            torch.testing.assert_close(actual, expected, atol=.005, rtol=.02)
            torch.testing.assert_close(x.grad, dx, atol=.001, rtol=.03)
            for p, g in zip(model.parameters(), gradients):
                torch.testing.assert_close(p.grad, g, atol=.001, rtol=.03)
            if index == 0:
                initial_graphs = counter.frame_count
                assert initial_graphs > 0
            assert counter.frame_count == initial_graphs


def test_compiled_pack_rejects_bad_boundaries():
    from flash_dism.compiler import validate_pack
    counter = CompileCounterWithBackend('eager')
    compiled = torch.compile(validate_pack, backend=counter, fullgraph=True, dynamic=True)
    for values, maximum in [([0, 1, 1024], 1024), ([0, 1024], 256)]:
        cu = torch.tensor(values, device='cuda', dtype=torch.int32)
        with pytest.raises((RuntimeError, ValueError)):
            compiled(cu, 1024, maximum)


@pytest.mark.parametrize('packed', [False, True])
@pytest.mark.parametrize('r,d,dv', [(r, d, dv) for r in (16, 32) for d in (32, 64) for dv in (32, 64)])
@pytest.mark.parametrize('shared_vocab', [False, True])
def test_opaque_voc_gradient_matches_eager(packed, r, d, dv, shared_vocab):
    from flash_dism import voc_dism
    from flash_dism.compiler import voc_forward
    torch.manual_seed(23)
    vectors = [torch.randn(1, 256, 2, dim, device='cuda', dtype=torch.bfloat16).mul_(.2).requires_grad_()
               for dim in (d, d, r, r, dv)]
    tables = [torch.randn((64,d) if shared_vocab else (2,64,d), device='cuda', requires_grad=True)
              for _ in range(2)]
    tau = torch.ones(2, device='cuda', requires_grad=True)
    inputs = [*vectors, *tables, tau]
    direction = torch.tensor([[True, False]], device='cuda')
    hard = torch.rand(1, 2, 256, device='cuda') < .5
    cu = torch.tensor([0, 0, 256], device='cuda', dtype=torch.int32) if packed else None
    expected = voc_dism(*inputs, direction=direction, hard=hard, cu_seqlens=cu)
    actual = voc_forward(*inputs, direction, hard, cu)[0]
    dout = torch.randn_like(actual)
    expected_grad = torch.autograd.grad(expected, inputs, dout)
    actual_grad = torch.autograd.grad(actual, inputs, dout)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for a, e in zip(actual_grad, expected_grad):
        torch.testing.assert_close(a, e, atol=.001, rtol=.03)


def test_prepared_layout_owns_boundary_snapshot():
    from flash_dism.compiler import prepare_pack, voc_forward
    torch.manual_seed(31)
    vectors = [torch.randn(1, 512, 2, dim, device='cuda', dtype=torch.bfloat16)
               .mul_(.2).requires_grad_() for dim in (32, 32, 16, 16, 32)]
    tables = [torch.randn(2, 64, 32, device='cuda', requires_grad=True) for _ in range(2)]
    inputs = [*vectors, *tables, torch.ones(2, device='cuda', requires_grad=True)]
    direction = torch.tensor([[True, False]], device='cuda')
    hard = torch.rand(1, 2, 512, device='cuda') < .5
    # Reuse the same storage with distinct contents, retain both forwards, and
    # run backwards later. Neither pointer identity nor last-used-pack is safe.
    cu = torch.tensor([0, 256, 512], device='cuda', dtype=torch.int32)
    checked, layout = prepare_pack(cu, 512, 512)
    old = voc_forward(*inputs, direction, hard, checked, layout)[0]
    cu.copy_(torch.tensor([0, 512, 512], device='cuda', dtype=torch.int32))
    checked2, layout2 = prepare_pack(cu, 512, 512)
    new = voc_forward(*inputs, direction, hard, checked2, layout2)[0]
    expected_old = voc_forward(*inputs, direction, hard,
        torch.tensor([0, 256, 512], device='cuda', dtype=torch.int32))[0]
    # Compact and capacity-padded metadata must represent identical attention.
    expected_new = voc_forward(*inputs, direction, hard, cu[:2])[0]
    torch.testing.assert_close(old, expected_old, atol=0, rtol=0)
    torch.testing.assert_close(new, expected_new, atol=0, rtol=0)
    dout = torch.randn_like(old)
    actual = torch.autograd.grad(old + new, inputs, dout)
    expected = torch.autograd.grad(expected_old + expected_new, inputs, dout)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, atol=.001, rtol=.03)


@pytest.mark.parametrize('hybrid', [False, True])
def test_module_empty_tail_equivalence(hybrid):
    torch.manual_seed(51)
    cls = DismSwaAttention if hybrid else DismAttention
    model = cls(64, 2, head_dim=32, value_dim=32, readout_dim=16,
                vocab_size=64).cuda().train()
    x = torch.randn(1, 1024, 64, device='cuda', requires_grad=True)
    cu = torch.tensor([0, 256, 1024] + [1024]*12, dtype=torch.int32, device='cuda')
    hard = torch.rand(1, 2, 1024, device='cuda') < .5
    direction = torch.tensor([[True, False]], device='cuda')
    params = [x, *model.parameters()]
    outputs, grads = [], []
    for boundaries in (cu, cu[:3]):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = model(x, cu_seqlens=boundaries, max_seqlen=1024,
                        hard=hard, direction=direction, use_cache=False)[0]
        outputs.append(out)
        grads.append(torch.autograd.grad(out.float().square().mean(), params))
    torch.testing.assert_close(*outputs, atol=.005, rtol=.02)
    for a, e in zip(*grads):
        torch.testing.assert_close(a, e, atol=.001, rtol=.03)
