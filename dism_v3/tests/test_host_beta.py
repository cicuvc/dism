"""Host vocabulary temperature: no device scalar round-trip or stale backward."""
import pytest
import torch


@pytest.mark.parametrize('packed', [False, True])
def test_host_beta_compiled(packed):
    from flash_dism.compiler import voc_forward
    from flash_dism.varlen import VarlenLayout
    torch.manual_seed(932)
    values = [(torch.randn(1, 256, 2, d, device='cuda')*.2).bfloat16().requires_grad_()
              for d in (32, 32, 16, 16, 32)]
    values += [(torch.randn(2, 64, 32, device='cuda')*.2).requires_grad_() for _ in range(2)]
    values += [torch.ones(2, device='cuda', requires_grad=True)]
    hard = torch.zeros(1, 2, device='cuda', dtype=torch.bool)
    direction = torch.tensor([[True, False]], device='cuda')
    boundaries = torch.tensor([0, 0, 256], device='cuda', dtype=torch.int32) if packed else None
    layout = VarlenLayout.from_cu_seqlens(boundaries, 256).table.flatten() if packed else None
    def run(beta, *args):
        return voc_forward(*args, beta, direction, hard, boundaries, layout)[0]
    dout = torch.randn_like(values[4])
    with torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True), \
         torch._inductor.config.patch(compile_threads=1):
        compiled = torch.compile(run, fullgraph=True, dynamic=True)
        previous = None
        for value in (1., .5, 1.75):
            expected = voc_forward(*values, torch.tensor(value), direction, hard, boundaries, layout)[0]
            eg = torch.autograd.grad(expected, values, dout)
            actual = compiled(value, *values)
            ag = torch.autograd.grad(actual, values, dout)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            for a, e in zip(ag, eg):
                torch.testing.assert_close(a, e, atol=.001, rtol=.03)
            if previous is not None:
                assert not torch.equal(actual, previous)
            previous = actual.detach()
        with torch._dynamo.config.patch(error_on_recompile=True), \
             torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
            actual = compiled(1.75, *values)
            torch.autograd.grad(actual, values, dout)
        # CPU scalar plumbing by Dynamo is allowed; the custom ops must not
        # perform Tensor.item (in particular no device->host beta read).
        for event in prof.events():
            if event.name in ('aten::item', 'aten::_local_scalar_dense'):
                parent = event.cpu_parent
                while parent is not None:
                    assert not parent.name.startswith('flash_dism::voc_')
                    parent = parent.cpu_parent


def test_host_beta_backward_keeps_forward_value():
    from flash_dism.emb_kernel import interp_embeddings
    torch.manual_seed(934)
    q, k = [(torch.randn(1, 256, 2, 32, device='cuda')*.2).bfloat16().requires_grad_() for _ in range(2)]
    eq, ek = [(torch.randn(2, 64, 32, device='cuda')*.2).bfloat16().requires_grad_() for _ in range(2)]
    args = (q, k, eq, ek)
    first = interp_embeddings(*args, beta=.5)[0].float().sum()
    second = interp_embeddings(*args, beta=1.75)[0].float().sum()
    actual = torch.autograd.grad(first, args)
    expected = torch.autograd.grad(interp_embeddings(*args, beta=.5)[0].float().sum(), args)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, atol=.001, rtol=.03)
    assert torch.isfinite(second)
