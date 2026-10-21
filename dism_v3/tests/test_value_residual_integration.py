"""Common-helper dispatch, compiler contract and real-caller integration."""

import pytest
import torch
from torch._dynamo.testing import CompileCounter, CompileCounterWithBackend

import flash_dism.kernels.value_residual as value_residual
from flash_dism import DismAttention
from flash_dism.module import value_residual_mix, value_residual_mix_reference

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _make(batch, seq, heads, dim, dtype=torch.bfloat16, gate_dtype=None):
    gate_dtype = dtype if gate_dtype is None else gate_dtype
    v = torch.randn(batch, seq, heads, dim, device="cuda", dtype=dtype) * 3
    v_first = torch.randn(batch, seq, heads, dim, device="cuda", dtype=dtype) * 3
    gate = torch.randn(batch, seq, heads, device="cuda", dtype=gate_dtype) * 4
    dy = torch.randn(batch, seq, heads, dim, device="cuda", dtype=dtype)
    return v, v_first, gate, dy


def test_helper_dispatches_to_kernel_and_falls_back(monkeypatch):
    # Independent of any DISM_VALUE_RESIDUAL_KERNEL in the environment.
    monkeypatch.setattr(value_residual, "_KERNEL_MODE", "fused")
    torch.manual_seed(0)
    v, v_first, gate, _ = _make(2, 4, 2, 64)
    assert value_residual.value_residual_supported(v, v_first, gate)
    assert torch.equal(value_residual_mix(v, v_first, gate),
                       value_residual_mix_reference(v, v_first, gate))

    # FP64 is a maths reference only: it must not be silently narrowed.
    v64, vf64, g64, _ = _make(2, 4, 2, 64, dtype=torch.float64, gate_dtype=torch.float64)
    assert not value_residual.value_residual_supported(v64, vf64, g64)
    assert torch.equal(value_residual_mix(v64, vf64, g64),
                       value_residual_mix_reference(v64, vf64, g64))

    # CPU keeps the eager path (this also covers the decode/CPU users).
    cpu = torch.randn(1, 4, 2, 8)
    assert not value_residual.value_residual_supported(cpu, cpu, cpu[:, :, :, 0])
    assert torch.equal(value_residual_mix(cpu, cpu, cpu[:, :, :, 0]),
                       value_residual_mix_reference(cpu, cpu, cpu[:, :, :, 0]))

    # A broadcastable gate is not the [B,N,H] row layout the kernel indexes.
    assert not value_residual.value_residual_supported(v, v_first, gate[0])

    monkeypatch.setattr(value_residual, "_KERNEL_MODE", "reference")
    assert not value_residual.value_residual_supported(v, v_first, gate)
    assert torch.equal(value_residual_mix(v, v_first, gate),
                       value_residual_mix_reference(v, v_first, gate))


def test_compiled_helper_forward_backward_single_graph(monkeypatch):
    torch._dynamo.reset()
    monkeypatch.setattr(value_residual, "_KERNEL_MODE", "fused")
    torch.manual_seed(1)
    v, v_first, gate, dy = _make(2, 6, 2, 64)

    def loss(a, b, c, upstream):
        return (value_residual_mix(a, b, c).float() * upstream.float()).sum()

    counter = CompileCounter()
    compiled = torch.compile(loss, backend=counter, fullgraph=True, dynamic=True)
    for seed in (0, 1):
        torch.manual_seed(seed)
        a = v.detach().clone().requires_grad_(True)
        b = v_first.detach().clone().requires_grad_(True)
        c = gate.detach().clone().requires_grad_(True)
        compiled(a, b, c, dy).backward()
        torch.manual_seed(seed)
        ra = v.detach().clone().requires_grad_(True)
        rb = v_first.detach().clone().requires_grad_(True)
        rc = gate.detach().clone().requires_grad_(True)
        loss(ra, rb, rc, dy).backward()
        assert torch.equal(a.grad, ra.grad)
        # v_first is the one documented eager/kernel rounding divergence.
        assert (b.grad.float() - rb.grad.float()).abs().max() <= 2 ** -8 * dy.abs().max()
        assert (c.grad.float() - rc.grad.float()).abs().max() <= 1e-5
    assert counter.frame_count == 1
    assert counter.op_count > 0


def _dism_attention(**kwargs):
    return DismAttention(64, 2, head_dim=32, value_dim=32, readout_dim=16,
                         vocab_size=64, value_residual=True, **kwargs).cuda().train()


def _assert_caller_output_close(actual, expected):
    # Approximate tanh may cross BF16 rounding boundaries before attention.
    # Bound propagated error separately; parameter-gradient bounds below stay.
    a, e = actual.float(), expected.float()
    torch.testing.assert_close(a, e, rtol=1e-2, atol=1e-3*e.abs().max().item())
    assert (a-e).norm() <= 5e-4*e.norm()


def test_dism_caller_gradients_match_eager_reference(monkeypatch):
    monkeypatch.setattr(value_residual, "_KERNEL_MODE", "fused")
    torch.manual_seed(21)
    model = _dism_attention()
    x = torch.randn(1, 512, 64, device="cuda", requires_grad=True)
    v_first = torch.randn(1, 512, 2, 32, device="cuda", dtype=torch.bfloat16,
                          requires_grad=True)
    cu = torch.tensor([0, 256, 512], device="cuda", dtype=torch.int32)
    hard = torch.tensor([[True, False]], device="cuda")
    direction = torch.tensor([[True, False]], device="cuda")

    def run():
        model.zero_grad(set_to_none=True)
        v_first.grad = None
        if x.grad is not None:
            x.grad = None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x, cu_seqlens=cu, max_seqlen=512, hard=hard,
                        direction=direction, v_first=v_first)[0]
        out.float().square().mean().backward()
        return (out.detach().clone(),
                {name: p.grad.detach().clone() for name, p in model.named_parameters()},
                x.grad.detach().clone(), v_first.grad.detach().clone())

    out_fused, grads_fused, dx_fused, dvf_fused = run()
    monkeypatch.setattr(value_residual, "_KERNEL_MODE", "reference")
    out_eager, grads_eager, dx_eager, dvf_eager = run()

    _assert_caller_output_close(out_fused, out_eager)
    scale = v_first.abs().max()
    for name, grad in grads_fused.items():
        reference = grads_eager[name]
        assert grad.shape == reference.shape, name
        assert torch.isfinite(grad).all(), name
        relative = (grad.float() - reference.float()).norm() / (
            reference.float().norm() + 1e-12)
        assert relative < 5e-3, (name, relative.item())
    # The value-residual gate Linear is the only consumer of dgate: its weight
    # gradient includes both the DV reduction and sigmoid approximation error.
    gate_weight = grads_fused["v_residual_gate.weight"]
    # Autocast Linear rounds the accumulated weight gradient to BF16; tanh
    # can cross that boundary. Retain the relative 5e-3 check above and allow
    # one BF16 epsilon at the tensor scale here (formerly a fixed 1e-6).
    gate_reference = grads_eager["v_residual_gate.weight"].float()
    assert (gate_weight.float()
            - gate_reference).abs().max() <= torch.finfo(torch.bfloat16).eps*gate_reference.abs().max()
    assert (dvf_fused.float() - dvf_eager.float()).abs().max() <= 2 ** -8 * scale
    assert (dx_fused.float() - dx_eager.float()).abs().max() <= 1e-3


def test_dism_caller_compiles_without_extra_graphs(monkeypatch):
    torch._dynamo.reset()
    monkeypatch.setattr(value_residual, "_KERNEL_MODE", "fused")
    torch.manual_seed(22)
    model = _dism_attention()
    x = torch.randn(1, 512, 64, device="cuda", requires_grad=True)
    v_first = torch.randn(1, 512, 2, 32, device="cuda", dtype=torch.bfloat16,
                          requires_grad=True)
    hard = torch.tensor([[True, False]], device="cuda")
    direction = torch.tensor([[True, False]], device="cuda")
    counter = CompileCounter()
    compiled = torch.compile(model, backend=counter, fullgraph=True, dynamic=True)
    # 256-aligned packed boundaries, including a different document count.
    packs = ([0, 256, 512], [0, 256, 256, 512], [0, 512])
    with torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True):
        outputs = []
        for boundaries in packs:
            cu = torch.tensor(boundaries, device="cuda", dtype=torch.int32)
            torch._dynamo.mark_dynamic(cu, 0)
            model.zero_grad(set_to_none=True)
            if x.grad is not None:
                x.grad = None
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = compiled(x, cu_seqlens=cu, max_seqlen=512, hard=hard,
                               direction=direction, v_first=v_first)[0]
            out.float().square().mean().backward()
            outputs.append(out.detach().clone())
    assert counter.frame_count == 1, counter.frame_count
    assert all(torch.isfinite(o).all() for o in outputs)


def test_pure_gdn_caller_matches_eager_reference(monkeypatch):
    from flash_dism.pure_gdn import PureGdnAttention

    monkeypatch.setattr(value_residual, "_KERNEL_MODE", "fused")
    torch.manual_seed(23)
    model = PureGdnAttention(64, 2, value_residual=True).cuda().train()
    x = torch.randn(1, 512, 64, device="cuda", requires_grad=True)
    v_first = torch.randn(1, 512, 2, 64, device="cuda", dtype=torch.bfloat16,
                          requires_grad=True)
    cu = torch.tensor([0, 256, 512], device="cuda", dtype=torch.int32)

    def run():
        model.zero_grad(set_to_none=True)
        v_first.grad = None
        if x.grad is not None:
            x.grad = None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x, cu_seqlens=cu, max_seqlen=512, v_first=v_first)[0]
        out.float().square().mean().backward()
        return (out.detach().clone(),
                {name: p.grad.detach().clone() for name, p in model.named_parameters()},
                v_first.grad.detach().clone())

    out_fused, grads_fused, dvf_fused = run()
    monkeypatch.setattr(value_residual, "_KERNEL_MODE", "reference")
    out_eager, grads_eager, dvf_eager = run()
    _assert_caller_output_close(out_fused, out_eager)
    assert torch.isfinite(dvf_fused).all() and dvf_fused.abs().sum() > 0
    for name, grad in grads_fused.items():
        reference = grads_eager[name]
        assert torch.isfinite(grad).all(), name
        relative = (grad.float() - reference.float()).norm() / (
            reference.float().norm() + 1e-12)
        assert relative < 5e-3, (name, relative.item())
    assert (dvf_fused.float() - dvf_eager.float()).abs().max() <= (
        2 ** -8 * v_first.abs().max())
