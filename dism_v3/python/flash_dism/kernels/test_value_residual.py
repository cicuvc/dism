"""Correctness tests for the fused value-residual operator.

The primary reference is the eager autograd graph of the original helper
(``flash_dism.module.value_residual_mix``); an FP64 evaluation of the same
closed form separates input rounding from implementation error.  Every case
uses a random, non-uniform upstream gradient rather than a plain ``sum()``.
"""

import unittest

import torch

from value_residual import (
    torch_bwd_value_residual,
    torch_fwd_value_residual,
    value_residual_fused,
    value_residual_supported,
)


def reference_mix(v, v_first, gate):
    """Byte-for-byte the eager helper's forward expression."""
    g = torch.sigmoid(gate.float()).unsqueeze(-1)
    mixed = v_first.float() + (v.float() - v_first.float()) * g
    return mixed.to(v.dtype)


def fp64_reference(v, v_first, gate, dy):
    a, b = v.double(), v_first.double()
    s = torch.sigmoid(gate.double()).unsqueeze(-1)
    d = a - b
    out = b + d * s
    dv = dy.double() * s
    dvfirst = dy.double() - dv
    dgate = (dy.double() * d).sum(-1) * (1.0 - s.squeeze(-1)) * s.squeeze(-1)
    return out, dv, dvfirst, dgate


def metrics(actual, expected):
    a = actual.double().flatten()
    e = expected.double().flatten()
    diff = a - e
    denom = e.norm()
    cosine = torch.nn.functional.cosine_similarity(a, e, dim=0)
    return {
        "max_abs": diff.abs().max().item(),
        "rel_l2": (diff.norm() / denom).item() if denom > 0 else diff.norm().item(),
        "cosine": cosine.item(),
        "norm_ratio": (a.norm() / denom).item() if denom > 0 else float("nan"),
    }


def dtype_eps(dtype):
    return torch.finfo(dtype).eps


def max_ulp_difference(actual, expected, scale):
    """Max |a - e| expressed in units of the dtype ulp at ``scale``."""
    diff = (actual.float() - expected.float()).abs().max().item()
    if diff == 0.0:
        return 0.0
    info = torch.finfo(expected.dtype)
    step = info.eps * max(float(scale), float(info.tiny))
    return diff / step


def grads_of_forward(output, inputs, dy):
    return torch.autograd.grad(output, inputs, dy, retain_graph=True,
                               allow_unused=True)


def strided_leaf(data, strides):
    """A genuine non-contiguous leaf with a storage gap, like a real slice."""
    leaf = torch.empty_strided(data.shape, strides, dtype=data.dtype,
                               device=data.device)
    with torch.no_grad():
        leaf.copy_(data)
    return leaf.requires_grad_(True)


class ValueResidualCase(unittest.TestCase):
    def assert_sigmoid_close(self, actual, expected, fp32_l2=1e-5):
        # Explicit tanh contract (not a tolerance change to other kernels).
        # Approximation can cross a BF16/F16 rounding boundary. Permit one
        # output-dtype ulp plus FP32 cancellation error, and bound global L2.
        scale = expected.detach().float().abs().max().item()
        rtol = max(1e-5, torch.finfo(expected.dtype).eps)
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=1e-5*scale)
        error = (actual.double()-expected.double()).norm()
        bound = 3e-4 if expected.dtype == torch.bfloat16 else 1e-4 if expected.dtype == torch.float16 else fp32_l2
        self.assertLessEqual(error.item(), bound*expected.double().norm().item()+1e-12)

    def assert_dgate_close(self, actual, expected):
        if expected.dtype == torch.float32:
            self.assert_sigmoid_close(actual, expected, fp32_l2=2e-5)
        else:
            self.assertLessEqual(max_ulp_difference(actual, expected, expected.abs().max()), 4.)

    def setUp(self):
        torch.manual_seed(20261014)
        self.device = "cuda"

    def make(self, batch, seq, heads, dv, dtype, gate_dtype=None, first_dtype=None):
        gate_dtype = dtype if gate_dtype is None else gate_dtype
        first_dtype = dtype if first_dtype is None else first_dtype
        v = (torch.randn(batch, seq, heads, dv, device=self.device, dtype=dtype) * 3)
        v_first = (torch.randn(batch, seq, heads, dv, device=self.device,
                               dtype=first_dtype) * 3)
        gate = (torch.randn(batch, seq, heads, device=self.device,
                            dtype=gate_dtype) * 4)
        dy = torch.randn(batch, seq, heads, dv, device=self.device, dtype=dtype)
        return v, v_first, gate, dy

    def run_pair(self, v, v_first, gate, dy):
        """Return fused grads, eager grads and the fp64 reference for one case."""
        vf_ = v.detach().requires_grad_(True)
        vff_ = v_first.detach().requires_grad_(True)
        gf_ = gate.detach().requires_grad_(True)
        fused_out = value_residual_fused(vf_, vff_, gf_)
        fused = grads_of_forward(fused_out, (vf_, vff_, gf_), dy)

        vr_ = v.detach().requires_grad_(True)
        vfr_ = v_first.detach().requires_grad_(True)
        gr_ = gate.detach().requires_grad_(True)
        eager_out = reference_mix(vr_, vfr_, gr_)
        eager = grads_of_forward(eager_out, (vr_, vfr_, gr_), dy)
        exact = fp64_reference(v, v_first, gate, dy)
        return (fused_out, *fused), (eager_out, *eager), exact


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestValueResidualKernel(ValueResidualCase):
    def test_tanh_forward_and_gradients_match_reference(self):
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            for dv in (32, 64):
                with self.subTest(dtype=dtype, dv=dv):
                    v, vf, gate, dy = self.make(2, 5, 3, dv, dtype)
                    (fo, fdv, fdf, fdg), (eo, edv, edf, edg), exact = self.run_pair(
                        v, vf, gate, dy)
                    # Tanh approximation is allowed; mixture arithmetic remains
                    # FP32 and the gradient reduction stays inside the CTA.
                    self.assert_sigmoid_close(fo, eo)
                    self.assert_sigmoid_close(fdv, edv)
                    # Keep low-precision dgate bounds; FP32 has an explicit
                    # approximation bound for MUFU.TANH.
                    self.assert_dgate_close(fdg, edg)
                    # dvfirst is FP32 in the fused kernel by contract.
                    s = torch.sigmoid(gate.float()).unsqueeze(-1)
                    single = (dy.float() - dy.float() * s).to(dtype)
                    self.assert_sigmoid_close(fdf, single)
                    fused_error = max_ulp_difference(fdf, edf, dy.abs().max())
                    if dtype is torch.float32:
                        self.assert_sigmoid_close(fdf, edf)
                    else:
                        self.assertLessEqual(fused_error, 1.0)

    def test_eager_reference_rounds_v_first_twice(self):
        """Document *why* BF16/F16 dvfirst differs: two roundings, not a bug."""
        for dtype in (torch.bfloat16, torch.float16):
            v, vf, gate, dy = self.make(2, 6, 3, 64, dtype)
            _, (_, _, edf, _), _ = self.run_pair(v, vf, gate, dy)
            s = torch.sigmoid(gate.float()).unsqueeze(-1)
            twice = (dy.float().to(dtype).float()
                     + (-(dy.float() * s)).to(dtype).float()).to(dtype)
            self.assertTrue(torch.equal(edf, twice), dtype)
            # and the duplicated rounding is strictly less accurate than the
            # FP32 form used by the fused kernel
            exact = fp64_reference(v, vf, gate, dy)[2].to(dtype)
            single = (dy.float() - dy.float() * s).to(dtype)
            self.assertLessEqual(metrics(single, exact)["rel_l2"],
                                 metrics(twice, exact)["rel_l2"] + 1e-12)

    def test_tanh_sigmoid_dense_range(self):
        gate = torch.linspace(-100, 100, 65537, device='cuda').reshape(1, -1, 1)
        v = torch.ones(1, 65537, 1, 32, device='cuda')
        vf = torch.zeros_like(v)
        out = torch_fwd_value_residual(v, vf, gate)
        expected = gate.sigmoid().unsqueeze(-1).expand_as(v)
        torch.testing.assert_close(out, expected, rtol=0, atol=4e-6)
        self.assertTrue(torch.isfinite(out).all())
        dv, df, dg = torch_bwd_value_residual(v, vf, gate, torch.ones_like(v))
        self.assertTrue(torch.equal(out, dv))
        self.assertTrue(torch.equal(df, 1-out))
        # Check the gradient algebra independently of sigmoid approximation.
        actual_s = out[..., 0]
        self.assertTrue(torch.equal(dg, (32*(1-actual_s))*actual_s))
        s = gate.sigmoid()
        torch.testing.assert_close(dg, (32*(1-s))*s, rtol=0, atol=32*4e-6)

    def test_fp64_reference_bounds(self):
        v, vf, gate, dy = self.make(2, 8, 3, 64, torch.bfloat16)
        (fo, fdv, fdf, fdg), (eo, edv, edf, edg), exact = self.run_pair(
            v, vf, gate, dy)
        for name, fused, eager, ref in (
                ("out", fo, eo, exact[0]), ("dv", fdv, edv, exact[1]),
                ("dvfirst", fdf, edf, exact[2]), ("dgate", fdg, edg, exact[3])):
            with self.subTest(name=name):
                # The output is BF16, so one rounding of the FP64 result is the
                # irreducible floor; require the fused path to stay there.
                self.assertLessEqual(
                    max_ulp_difference(fused, ref.to(fused.dtype), ref.abs().max()),
                    1.0)
            fused_vs_exact = metrics(fused, ref)["rel_l2"]
            eager_vs_exact = metrics(eager, ref)["rel_l2"]
            self.assertLessEqual(fused_vs_exact, 3e-3, name)
            if name == "dvfirst":
                # The eager graph's duplicated ``v_first.float()`` costs it a
                # whole BF16 rounding on the dominant term.
                self.assertGreater(eager_vs_exact, fused_vs_exact)

    def test_production_shape_and_stride(self):
        shape = (1, 32768, 6, 64)
        v = torch.randn(*shape, device=self.device, dtype=torch.bfloat16)
        vf = torch.randn(*shape, device=self.device, dtype=torch.bfloat16)
        gate = torch.randn(1, 32768, 6, device=self.device, dtype=torch.bfloat16)
        dy = torch.randn(*shape, device=self.device, dtype=torch.bfloat16)
        self.assert_sigmoid_close(torch_fwd_value_residual(v, vf, gate), reference_mix(v, vf, gate))
        dv, df, dg = torch_bwd_value_residual(v, vf, gate, dy)
        s = torch.sigmoid(gate.float()).unsqueeze(-1)
        self.assert_sigmoid_close(dv, (dy.float() * s).to(torch.bfloat16))
        self.assert_sigmoid_close(df, (dy.float() - dy.float() * s).to(torch.bfloat16))
        self.assertEqual(dg.shape, gate.shape)
        self.assertEqual(dg.dtype, gate.dtype)
        v2, vf2, g2 = (t.clone().requires_grad_(True) for t in (v, vf, gate))
        edv, _, edg = torch.autograd.grad(
            reference_mix(v2, vf2, g2), (v2, vf2, g2), dy)
        self.assert_sigmoid_close(dv, edv)
        # DV=64 reduction-order slack, measured in ulps of the sum.
        self.assert_dgate_close(dg, edg)

    def test_rows_not_multiple_of_block(self):
        # 3*7*5 = 105 rows, not a multiple of the 8-row block.
        v, vf, gate, dy = self.make(3, 7, 5, 64, torch.float32)
        (fo, fdv, fdf, fdg), (eo, edv, edf, edg), _ = self.run_pair(v, vf, gate, dy)
        self.assert_sigmoid_close(fo, eo)
        self.assert_sigmoid_close(fdv, edv)
        self.assert_sigmoid_close(fdf, edf)
        self.assert_dgate_close(fdg, edg)
        # non power-of-two DV with masking
        v, vf, gate, dy = self.make(2, 5, 3, 40, torch.bfloat16)
        (fo, fdv, fdf, fdg), (eo, edv, edf, edg), _ = self.run_pair(v, vf, gate, dy)
        self.assert_sigmoid_close(fo, eo)
        self.assert_sigmoid_close(fdv, edv)
        self.assert_sigmoid_close(fdf, (dy.float() - dy.float() * torch.sigmoid(
            gate.float()).unsqueeze(-1)).to(torch.bfloat16))

    def test_last_axis_strides_and_expanded_upstream(self):
        for dtype in (torch.bfloat16, torch.float32):
            for layout in ('slice', 'transpose', 'expand'):
                with self.subTest(dtype=dtype, layout=layout):
                    def make():
                        if layout == 'slice':
                            return torch.randn(2, 7, 3, 128, device='cuda', dtype=dtype)[..., ::2]
                        if layout == 'transpose':
                            return torch.randn(2, 7, 64, 3, device='cuda', dtype=dtype).transpose(-1, -2)
                        return torch.randn(2, 7, 3, 1, device='cuda', dtype=dtype).expand(2, 7, 3, 64)
                    v, vf, dy = make(), make(), make()
                    gate = torch.randn(2, 7, 3, device='cuda', dtype=dtype)
                    (fo, dv, df, dg), (eo, edv, _, edg), _ = self.run_pair(v, vf, gate, dy)
                    self.assertTrue(value_residual_supported(v, vf, gate))
                    self.assert_sigmoid_close(fo, eo)
                    self.assert_sigmoid_close(dv, edv)
                    s = gate.float().sigmoid().unsqueeze(-1)
                    self.assert_sigmoid_close(df, (dy.float() - dy.float()*s).to(dtype))
                    self.assert_dgate_close(dg, edg)

    def test_sum_backward_zero_stride_scalar_upstream(self):
        for dtype in (torch.bfloat16, torch.float32):
            v, vf, gate, _ = self.make(2, 7, 3, 64, dtype)
            v, vf, gate = [t.requires_grad_() for t in (v, vf, gate)]
            value_residual_fused(v, vf, gate).sum().backward()
            s = gate.float().sigmoid().unsqueeze(-1)
            self.assert_sigmoid_close(v.grad, s.expand_as(v).to(dtype))
            self.assert_sigmoid_close(vf.grad, (1-s).expand_as(vf).to(dtype))
            expected = (((v.float()-vf.float()).sum(-1) * (1-s.squeeze(-1))) * s.squeeze(-1)).to(dtype)
            self.assert_dgate_close(gate.grad, expected)

    def test_raw_backward_rejects_cpu_upstream(self):
        v, vf, gate, dy = self.make(1, 2, 1, 32, torch.bfloat16)
        with self.assertRaisesRegex(ValueError, 'CUDA device'):
            torch_bwd_value_residual(v, vf, gate, dy.cpu())

    def test_saturating_and_degenerate_gates(self):
        batch, seq, heads, dim = 2, 7, 3, 64
        v = torch.randn(batch, seq, heads, dim, device=self.device,
                        dtype=torch.bfloat16)
        vf = torch.randn(batch, seq, heads, dim, device=self.device,
                         dtype=torch.bfloat16)
        gate = torch.zeros(batch, seq, heads, device=self.device,
                           dtype=torch.bfloat16)
        gate[0] = 60.0     # saturated high
        gate[1] = -60.0    # saturated low
        gate[0, 0, 0] = 0.0
        dy = torch.randn_like(v)
        out = torch_fwd_value_residual(v, vf, gate)
        self.assert_sigmoid_close(out, reference_mix(v, vf, gate))
        dv, df, dg = torch_bwd_value_residual(v, vf, gate, dy)
        self.assertTrue(torch.isfinite(dv).all() and torch.isfinite(df).all())
        self.assertTrue(torch.isfinite(dg).all())
        # v == v_first kills both the forward delta and dgate.
        same = torch.randn(batch, seq, heads, dim, device=self.device,
                           dtype=torch.bfloat16) * 100.0
        self.assertTrue(torch.equal(torch_fwd_value_residual(same, same, gate),
                                    reference_mix(same, same, gate)))
        _, df_same, dg_same = torch_bwd_value_residual(same, same, gate, dy)
        self.assertTrue(torch.equal(dg_same, torch.zeros_like(gate)))
        self.assertTrue(torch.equal(df_same, (dy.float() - dy.float() * torch.sigmoid(
            gate.float()).unsqueeze(-1)).to(torch.bfloat16)))

    def test_mixed_input_dtypes_return_their_own_dtype(self):
        v, vf, gate, dy = self.make(2, 4, 2, 64, torch.bfloat16,
                                    gate_dtype=torch.float32,
                                    first_dtype=torch.float32)
        (fo, fdv, fdf, fdg), (eo, edv, edf, edg), _ = self.run_pair(v, vf, gate, dy)
        self.assertEqual(fo.dtype, v.dtype)
        self.assertEqual(fdv.dtype, v.dtype)
        self.assertEqual(fdf.dtype, vf.dtype)
        self.assertEqual(fdg.dtype, gate.dtype)
        self.assert_sigmoid_close(fo, eo)
        self.assert_sigmoid_close(fdv, edv)
        self.assert_sigmoid_close(fdf, edf)

    def test_non_contiguous_inputs_and_upstream(self):
        batch, seq, heads, dv = 2, 8, 3, 64
        dtype = torch.bfloat16
        base = torch.randn(batch, seq, heads, dv, device=self.device, dtype=dtype)
        # storage gaps between heads and rows, exactly what a slice looks like
        v = strided_leaf(base, (seq * heads * dv * 2, heads * dv * 2, dv * 2, 1))
        vf = strided_leaf(base * 2, (seq * heads * dv * 2, heads * dv * 2,
                                     dv * 2, 1))
        gate = torch.randn(batch, seq, heads, device=self.device, dtype=dtype)
        dy = strided_leaf(base * 3, (seq * heads * dv, heads * dv, dv * 3, 1)).detach()
        self.assertFalse(v.is_contiguous())
        self.assertFalse(dy.is_contiguous())
        v2 = v.detach().requires_grad_(True)
        vf2 = vf.detach().requires_grad_(True)
        g2 = gate.clone().requires_grad_(True)
        out = value_residual_fused(v2, vf2, g2)
        self.assert_sigmoid_close(out, reference_mix(v, vf, gate))
        fdv, fdf, fdg = torch.autograd.grad(out, (v2, vf2, g2), dy)
        v3 = v.clone().requires_grad_(True)
        vf3 = vf.clone().requires_grad_(True)
        g3 = gate.clone().requires_grad_(True)
        eout = reference_mix(v3, vf3, g3)
        edv, edf, edg = torch.autograd.grad(eout, (v3, vf3, g3), dy)
        self.assert_sigmoid_close(fdv, edv)
        # dy is a strided slice, so the kernel must consume dout's strides.
        self.assert_sigmoid_close(fdf, (dy.float() - dy.float() * torch.sigmoid(
            gate.float()).unsqueeze(-1)).to(torch.bfloat16))
        self.assert_dgate_close(fdg, edg)

    def test_partial_gradients_and_frozen_inputs(self):
        v, vf, gate, dy = self.make(2, 5, 2, 64, torch.bfloat16)
        for frozen in ("v", "v_first", "gate"):
            with self.subTest(frozen=frozen):
                vv = v.detach().clone().requires_grad_(frozen != "v")
                ff = vf.detach().clone().requires_grad_(frozen != "v_first")
                gg = gate.detach().clone().requires_grad_(frozen != "gate")
                out = value_residual_fused(vv, ff, gg)
                wanted = [t for t in (vv, ff, gg) if t.requires_grad]
                grads = torch.autograd.grad(out, wanted, dy, allow_unused=True)
                self.assertTrue(all(g is not None for g in grads))
                self.assertEqual(len(wanted), 2)
                self.assertTrue(all(torch.isfinite(g).all() for g in grads))
                frozen_tensor = {"v": vv, "v_first": ff, "gate": gg}[frozen]
                self.assertFalse(frozen_tensor.requires_grad)
        # only the gate needs grad: still differentiable, no extra backward work
        vv = v.detach()
        ff = vf.detach()
        gg = gate.detach().clone().requires_grad_(True)
        out = value_residual_fused(vv, ff, gg)
        (ggrad,) = torch.autograd.grad(out, gg, dy)
        self.assertEqual(ggrad.dtype, gate.dtype)

    def test_shared_v_first_accumulates_across_calls(self):
        dv, dtype = 64, torch.bfloat16
        v1 = torch.randn(2, 5, 3, dv, device=self.device, dtype=dtype)
        v2 = torch.randn(2, 5, 3, dv, device=self.device, dtype=dtype)
        vf = torch.randn(2, 5, 3, dv, device=self.device, dtype=dtype)
        gate = torch.randn(2, 5, 3, device=self.device, dtype=dtype)
        dy1 = torch.randn_like(v1)
        dy2 = torch.randn_like(v2)
        a = v1.detach().clone().requires_grad_(True)
        b = v2.detach().clone().requires_grad_(True)
        f = vf.detach().clone().requires_grad_(True)
        g = gate.detach().clone().requires_grad_(True)
        out = value_residual_fused(a, f, g) + value_residual_fused(b, f, g)
        ga, gb, gf, gg = torch.autograd.grad(out, (a, b, f, g), dy1 + dy2)
        va = v1.detach().clone().requires_grad_(True)
        vb = v2.detach().clone().requires_grad_(True)
        vf_ = vf.detach().clone().requires_grad_(True)
        gt = gate.detach().clone().requires_grad_(True)
        ref = reference_mix(va, vf_, gt) + reference_mix(vb, vf_, gt)
        ra, rb, rf, rg = torch.autograd.grad(ref, (va, vb, vf_, gt), dy1 + dy2)
        self.assertTrue(torch.equal(ga, ra))
        self.assertTrue(torch.equal(gb, rb))
        # v_first accumulates both branches in one leaf; only the documented
        # BF16 double-rounding of the eager graph separates them.
        self.assertLessEqual(
            max_ulp_difference(gf, rf, dy1.abs().max() + dy2.abs().max()), 1.0)
        self.assert_dgate_close(gg, rg)

    def test_v_aliases_v_first(self):
        v, vf, gate, dy = self.make(2, 5, 3, 64, torch.bfloat16)
        alias = v.detach().clone().requires_grad_(True)
        g = gate.detach().clone().requires_grad_(True)
        out = value_residual_fused(alias, alias, g)
        (d_alias, d_gate) = torch.autograd.grad(out, (alias, g), dy)
        ref = v.detach().clone().requires_grad_(True)
        gr = gate.detach().clone().requires_grad_(True)
        eout = reference_mix(ref, ref, gr)
        ed_alias, ed_gate = torch.autograd.grad(eout, (ref, gr), dy)
        # dv + dvfirst = dy only up to the two input-dtype roundings.
        self.assertLessEqual(max_ulp_difference(d_alias, ed_alias, dy.abs().max()), 2.0)
        self.assert_dgate_close(d_gate, ed_gate)

    def test_retained_graph_two_different_backwards(self):
        v, vf, gate, _ = self.make(2, 5, 3, 64, torch.bfloat16)
        vv, ff, gg = (t.detach().clone().requires_grad_(True) for t in (v, vf, gate))
        out = value_residual_fused(vv, ff, gg)
        first = torch.randn_like(v)
        second = torch.randn_like(v)
        got1 = torch.autograd.grad(out, (vv, ff, gg), first, retain_graph=True)
        got2 = torch.autograd.grad(out, (vv, ff, gg), second)
        for upstream, got in ((first, got1), (second, got2)):
            rv, rf, rg = (t.detach().clone().requires_grad_(True) for t in (v, vf, gate))
            eout = reference_mix(rv, rf, rg)
            want = torch.autograd.grad(eout, (rv, rf, rg), upstream)
            self.assert_sigmoid_close(got[0], want[0])
            self.assert_sigmoid_close(got[1], (upstream.float() - upstream.float()
                            * torch.sigmoid(gate.float()).unsqueeze(-1)).to(torch.bfloat16))
        self.assertFalse(torch.equal(got1[1], got2[1]))

    def test_unknown_dtype_and_layout_are_rejected(self):
        v = torch.randn(1, 4, 2, 64, device=self.device, dtype=torch.float64)
        vf = torch.randn(1, 4, 2, 64, device=self.device, dtype=torch.float64)
        gate = torch.randn(1, 4, 2, device=self.device, dtype=torch.float64)
        self.assertFalse(value_residual_supported(v, vf, gate))
        with self.assertRaises(ValueError):
            torch_fwd_value_residual(v, vf, gate)
        # [B,N,H,DV] is required; a broadcast gate must fall back, not silently
        # use the wrong row mapping.
        bf = v.to(torch.bfloat16)
        self.assertFalse(value_residual_supported(bf, bf, gate.to(torch.bfloat16)[0]))
        cpu = torch.randn(1, 4, 2, 64)
        self.assertFalse(value_residual_supported(cpu, cpu, cpu[:, :, :, 0]))


if __name__ == "__main__":
    unittest.main()
