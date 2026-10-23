"""Softcapped LSE, chain rule before quantization, and balanced chunked CE."""

import pytest
import torch
import torch.nn.functional as F

from flash_mxfp8.mxfp8_ce import CEPlan, merge_lse
from flash_mxfp8.core import quantize_weight
from flash_mxfp8.tk_mxfp8_lse import Library as LSELibrary, CONFIGS
from flash_mxfp8.tk_mxfp8_gemm import Library as GemmLibrary
from flash_mxfp8.tk_mxfp8_dgrad import Library as DgradLibrary
from flash_mxfp8.tk_mxfp8_wgrad import Library as WgradLibrary
from flash_mxfp8.tk_mxfp8_fused_ce import MXFP8FusedLinearCrossEntropy, fused_linear_cross_entropy_loss
from flash_mxfp8.tk_silu_bwd import SiluBwdQuantizeLibrary

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="SM120 required",
)


def inputs(m, h, v, seed=921):
    torch.manual_seed(seed)
    return (torch.randn(m, h, device="cuda", dtype=torch.bfloat16),
            torch.randn(v, h, device="cuda") * .05,
            torch.randint(v, (m,), device="cuda"))


def reference_weights(y, boundaries, ignore_index=-100):
    weights = torch.zeros_like(y, dtype=torch.float32)
    counts = []
    for begin, end in zip(boundaries[:-1], boundaries[1:]):
        valid = y[begin:end] != ignore_index
        count = valid.sum()
        counts.append(count)
        weights[begin:end] = valid.float() / count.clamp_min(1)
    return weights, torch.stack(counts)


@pytest.mark.parametrize("config,m,h,v", [(0, 1, 64, 64), (8, 137, 128, 320),
                                          (70, 257, 128, 32000), (71, 129, 128, 320)])
@pytest.mark.parametrize("softcap", [.3, 30.])
def test_softcap_partial_lse_and_rebind(config, m, h, v, softcap):
    x, w, y = inputs(m, h, v)
    xq = SiluBwdQuantizeLibrary().plan_direct(x); xq()
    wq, ws = quantize_weight(w)
    args = (xq.row_fp8, wq, xq.row_scale, ws)
    plain = GemmLibrary().plan(*args, config=config)
    plan = LSELibrary().plan(*args, config=config, softcap=softcap)
    for _ in range(2):
        z, partial = plan()
        # Raw logits are retained, even when tanh saturates.
        torch.testing.assert_close(z, plain(), rtol=0, atol=0)
        capped = softcap * torch.tanh(z.float() / softcap)
        block = CONFIGS[config][1]
        expected = torch.stack([t.logsumexp(-1) for t in capped.split(block, -1)], -1)
        torch.testing.assert_close(partial, expected, atol=2e-6, rtol=2e-6)
        lse, loss = merge_lse(z, partial, y, softcap=softcap)
        torch.testing.assert_close(lse, capped.logsumexp(-1), atol=2e-6, rtol=2e-6)
        # Hardware tanh has absolute approximation error, amplified by s.
        torch.testing.assert_close(loss, F.cross_entropy(capped, y, reduction="none"),
                                   atol=2e-6 + softcap * 5e-7, rtol=2e-6)
        x.mul_(2); xq()
        plan.rebind(*args, out=torch.empty_like(z), partial_lse=torch.empty_like(partial))


@pytest.mark.parametrize("softcap", [.1, 2., 30.])
@pytest.mark.parametrize("ignore_index", [-100, 0])
def test_softcap_ce_dual_quantization(softcap, ignore_index):
    m, v = 137, 320
    torch.manual_seed(119)
    z = (torch.randn(m, v, device="cuda") * 3).bfloat16()
    # Exactly zero and deeply saturated rows test both extremes of the derivative.
    z[0] = 0
    z[1] = 1000
    z[2] = -1000
    saved = z.clone()
    y = torch.arange(m, device="cuda") % v
    y[32:64] = ignore_index
    y[-1] = ignore_index
    row_weight = torch.linspace(.01, 1., m, device="cuda")
    raw = z.float().requires_grad_()
    capped = softcap * torch.tanh(raw / softcap)
    losses = F.cross_entropy(capped, y, reduction="none", ignore_index=ignore_index) * row_weight
    g, = torch.autograd.grad(losses.sum(), raw)
    reference = SiluBwdQuantizeLibrary().plan_direct(g.bfloat16(), transpose=True); reference()
    partial = torch.stack([t.logsumexp(-1) for t in capped.detach().split(128, -1)], -1)
    lse, loss = merge_lse(z, partial, y, softcap=softcap, row_weight=row_weight, ignore_index=ignore_index)
    torch.testing.assert_close(loss, losses, atol=2e-6, rtol=2e-6)
    plan = CEPlan(m, v, z.device)
    plan(z, lse, y, softcap=softcap, row_weight=row_weight, ignore_index=ignore_index)
    # Compare represented gradients, including scales. At tanh saturation the
    # approximation may round a tiny residual to zero, changing a whole group's
    # exponent even though its absolute gradient error is negligible.
    def dequant(q, scale):
        exponent = scale.view(torch.uint8).reshape(scale.shape[0], scale.shape[1], 4)
        exponent = exponent.permute(1, 0, 2).reshape(scale.shape[1], -1)
        exponent = exponent.repeat_interleave(32, 1)[:q.shape[0], :q.shape[1]].int() - 127
        return torch.ldexp(q.float(), exponent)
    for direction in ("row", "column"):
        actual = dequant(getattr(plan, direction + "_fp8"), getattr(plan, direction + "_scale"))
        expected = dequant(getattr(reference, direction + "_fp8"), getattr(reference, direction + "_scale"))
        assert ((actual - expected).norm() / expected.norm()).item() < .002
    for rows in (y == ignore_index, torch.arange(m, device="cuda") == 1,
                 torch.arange(m, device="cuda") == 2):
        assert torch.count_nonzero(plan.row_fp8.float()[rows]).item() == 0
        assert torch.count_nonzero(plan.column_fp8.float()[:, :m][:, rows]).item() == 0
    assert torch.count_nonzero(plan.column_fp8.float()[:, m:]).item() == 0
    torch.testing.assert_close(z, saved, rtol=0, atol=0)


def component_reference(x, w, y, softcap, weights, denominator):
    wq, ws = quantize_weight(w.float())
    dx, dw = torch.empty_like(x), torch.zeros_like(w, dtype=torch.float32)
    losses = []
    for start in range(0, len(x), 128):
        end = min(start + 128, len(x))
        xq = SiluBwdQuantizeLibrary().plan_direct(x[start:end]); xq()
        z = GemmLibrary().plan(xq.row_fp8, wq, xq.row_scale, ws, config=8)()
        raw = z.float().requires_grad_()
        capped = softcap * torch.tanh(raw / softcap)
        loss = (F.cross_entropy(capped, y[start:end], reduction="none") * weights[start:end]).sum()
        g, = torch.autograd.grad(loss, raw)
        losses.append(loss.detach())
        gq = SiluBwdQuantizeLibrary().plan_direct(g.bfloat16(), transpose=True); gq()
        dx[start:end] = DgradLibrary().plan(gq.row_fp8, wq, gq.row_scale, ws, config=12)()
        dw.add_(WgradLibrary().plan(gq.column_fp8, xq.column_fp8, gq.column_scale, xq.column_scale, config=12)())
    inv = denominator.clamp_min(1).float().reciprocal()
    return torch.stack(losses).sum() * inv, dx * inv, dw.to(w.dtype) * inv


@pytest.mark.parametrize("balance", [None, "token", "document"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_softcap_chunked_balanced_gradients(balance, dtype):
    x, w, y = inputs(385, 128, 256)
    w = w.to(dtype)
    y[::7] = -100
    y[1:31] = -100
    boundaries = [0, 0, 1, 31, 320, 384, 385, 385]
    cu = torch.tensor(boundaries, device="cuda", dtype=torch.int32)
    weights, counts = reference_weights(y, boundaries)
    if balance != "document":
        weights = torch.ones_like(weights)
    denominator = ((counts > 0).sum() if balance == "document" else
                   (y != -100).sum() if balance == "token" else y.new_tensor(1))
    reference = component_reference(x, w, y, .7, weights, denominator)
    x.requires_grad_(); w.requires_grad_()
    criterion = MXFP8FusedLinearCrossEntropy(chunk_size=128, balance=balance, softcap=.7)
    loss = criterion(x, y, w, cu_seqlens=cu)
    dx, dw = torch.autograd.grad(loss, (x, w), retain_graph=True)
    torch.testing.assert_close(loss, reference[0], atol=2e-4, rtol=2e-6)
    for actual, expected in zip((dx, dw), reference[1:]):
        assert ((actual.float() - expected.float()).norm() / expected.float().norm()).item() < .004
    # Changes to the criterion cannot alter gradients precomputed by this forward.
    criterion.softcap = 30.
    criterion(x, y, w, cu_seqlens=cu)
    scaled = torch.autograd.grad(loss * -.5, (x, w))
    for actual, expected in zip(scaled, (dx, dw)):
        torch.testing.assert_close(actual, expected * -.5, atol=0, rtol=0)
    functional = fused_linear_cross_entropy_loss(x, y, w, chunk_size=128, balance=balance, softcap=.7, cu_seqlens=cu)
    torch.testing.assert_close(functional, loss, atol=0, rtol=0)


@pytest.mark.parametrize("balance", [None, "token", "document"])
def test_softcap_graph_replay_and_all_ignored(balance):
    x, w, y = inputs(137, 128, 256)
    x.requires_grad_(); w.requires_grad_()
    wq, ws = quantize_weight(w.detach())
    cu = torch.tensor([0, 31, 137], device="cuda", dtype=torch.int32)
    criterion = MXFP8FusedLinearCrossEntropy(chunk_size=128, balance=balance, softcap=1.3)
    def step():
        loss = criterion(x, y, w, cu_seqlens=cu, quantized_weight=(wq, ws))
        return loss, *torch.autograd.grad(loss, (x, w))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = step()
    for ignored in (False, True, False):
        with torch.no_grad():
            x.mul_(1.5)
            y.copy_(torch.arange(len(y), device="cuda") % 256)
            y[::3] = -100
            if ignored:
                y.fill_(-100)
            cu[1] = 129
        expected = step()
        graph.replay()
        for actual, ref in zip(captured, expected):
            torch.testing.assert_close(actual, ref, atol=0, rtol=0)
            if ignored:
                assert torch.count_nonzero(actual).item() == 0


def test_softcap_validation():
    for invalid in (0, -1., True, float('nan'), float('inf'), 1e100, 1e-100, "30", torch.tensor(30.)):
        with pytest.raises(ValueError, match="softcap"):
            MXFP8FusedLinearCrossEntropy(softcap=invalid)
    assert MXFP8FusedLinearCrossEntropy(softcap=30).softcap == 30.


@pytest.mark.parametrize('softcap', [None, 30.])
@pytest.mark.parametrize('all_ignored', [False, True])
def test_document_sum_global_accumulation(softcap, all_ignored):
    from nanochat.document_loss import DocumentLossAccumulator

    x, w, y = inputs(512, 128, 256)
    y[:31] = -1
    y[::7] = -1
    if all_ignored:
        y.fill_(-1)
    x.requires_grad_(); w.requires_grad_()
    ref_x = x.detach().clone().requires_grad_()
    ref_w = w.detach().clone().requires_grad_()
    # Unequal numbers of valid documents across the accumulation boundary.
    cu = torch.tensor([0, 0, 31, 256, 300, 400, 512], device='cuda', dtype=torch.int32)
    mean = MXFP8FusedLinearCrossEntropy(chunk_size=128, balance='document',
                                       ignore_index=-1, softcap=softcap)
    expected = mean(ref_x, y, ref_w, cu_seqlens=cu)
    expected.backward()
    summed = MXFP8FusedLinearCrossEntropy(chunk_size=128, balance='document_sum',
                                         ignore_index=-1, softcap=softcap)
    accumulator = DocumentLossAccumulator(2, 'cuda')
    for start, end, boundaries in [(0, 256, [0, 0, 31, 256]), (256, 512, [0, 44, 144, 256])]:
        loss, count = summed(x[start:end], y[start:end], w,
                            cu_seqlens=torch.tensor(boundaries, device='cuda', dtype=torch.int32))
        assert not count.requires_grad and count.dtype == torch.int64
        accumulator.add(loss, count)
        (loss / 2).backward()
    actual = accumulator.finalize([x, w])
    assert accumulator.count.item() == (0 if all_ignored else 4)
    torch.testing.assert_close(actual.float(), expected, atol=2e-6, rtol=2e-6)
    # BF16 scaling/FP32 summation happen in a different order across partitions.
    for got, ref in [(x.grad, ref_x.grad), (w.grad, ref_w.grad)]:
        if all_ignored:
            assert torch.count_nonzero(got).item() == 0
        else:
            relative = (got.float()-ref.float()).norm()/ref.float().norm()
            assert relative < .006


def test_no_serialized_workspace_and_saved_gradient_reuse():
    x, w, y = inputs(257, 128, 256)
    x.requires_grad_(); w.requires_grad_()
    criterion = MXFP8FusedLinearCrossEntropy(chunk_size=128, balance='document_sum',
                                            softcap=30.)
    cu = torch.tensor([0, 32, 257], device='cuda', dtype=torch.int32)
    loss, count = criterion(x, y, w, cu_seqlens=cu)
    expected = torch.autograd.grad(loss, (x, w), retain_graph=True)
    criterion(x * 2, y, w, cu_seqlens=cu)
    got = torch.autograd.grad(loss * -.5, (x, w))
    assert count.item() == 2 and not criterion.state_dict()
    for a, b in zip(got, expected):
        torch.testing.assert_close(a, b * -.5, rtol=0, atol=0)


@pytest.mark.parametrize('h,v', [(128, 512), (768, 32000)])
def test_bf16_reference_accuracy(h, v):
    # The upstream MXFP8 approximation budget, also at the production H/V.
    x, w, y = inputs(257, h, v)
    x.requires_grad_(); w = w.bfloat16().requires_grad_()
    criterion = MXFP8FusedLinearCrossEntropy(chunk_size=128, balance='token', softcap=30.)
    loss = criterion(x, y, w)
    dx, dw = torch.autograd.grad(loss, (x, w))
    logits = F.linear(x, w).float()
    ref = F.cross_entropy(30 * torch.tanh(logits / 30), y)
    rx, rw = torch.autograd.grad(ref, (x, w))
    assert abs(loss.item() - ref.item()) < .01
    for a, b in ((dx, rx), (dw, rw)):
        assert ((a.float()-b.float()).norm()/b.float().norm()).item() < .08


def test_dynamo_boundary_backward():
    x, w, y = inputs(137, 128, 256)
    x.requires_grad_(); w.requires_grad_()
    criterion = MXFP8FusedLinearCrossEntropy(chunk_size=128, balance='document_sum', softcap=30.)
    cu = torch.tensor([0, 32, 137], device='cuda', dtype=torch.int32)
    def loss(a, b):
        numerator, count = criterion(a, y, b, cu_seqlens=cu)
        return numerator / count.clamp_min(1)
    expected = loss(x, w)
    expected_grads = torch.autograd.grad(expected, (x, w))
    compiled = torch.compile(loss, backend='eager')
    got = compiled(x, w)
    got_grads = torch.autograd.grad(got, (x, w))
    torch.testing.assert_close(got, expected, atol=0, rtol=0)
    for a, b in zip(got_grads, expected_grads):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
