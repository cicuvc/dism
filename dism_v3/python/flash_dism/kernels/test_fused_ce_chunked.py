import unittest

import torch

from fused_ce_chunked import (
    chunked_ce_backward,
    chunked_ce_forward,
    fused_ce_chunked,
    fused_ce_chunked_stats,
)
from fused_cross_entropy import (
    reference_fused_cross_entropy,
    reference_varlen_fused_cross_entropy,
)


def diagnostics(actual, expected):
    """Max-abs, relative L2, cosine and norm ratio against a reference."""
    a = actual.detach().float().flatten()
    b = expected.detach().float().flatten()
    delta = a - b
    return {
        "max_abs": float(delta.abs().max()),
        "relative_l2": float(delta.norm() / b.norm().clamp_min(1e-30)),
        "cosine": float(torch.nn.functional.cosine_similarity(a, b, dim=0)),
        "norm_ratio": float(a.norm() / b.norm().clamp_min(1e-30)),
    }


# FP32 chunk buffers reproduce the explicit reference up to GEMM ordering.
_FP32_GRAD = {"max_abs": 2e-4, "relative_l2": 1e-4, "cosine": 0.99999}
# BF16 chunk buffers round the logits, the softmax gradient and both GEMM
# outputs once each; the measured deviation on these shapes is <=0.6% relative
# L2 (see docs/TRAINING_BENCHMARK.md), so these bounds are ~4x the measurement
# and are never used to accept a wrong formula.
_BF16_GRAD = {"max_abs": 5e-2, "relative_l2": 3e-2, "cosine": 0.999}


class ChunkedCETestCase(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261014)
        self.device = torch.device("cuda")

    def make_inputs(self, batch, sequence, channels, vocab, dtype=torch.float32):
        hidden = torch.randn(
            batch, sequence, channels, device=self.device, dtype=dtype)
        weight = torch.randn(vocab, channels, device=self.device, dtype=dtype)
        labels = torch.randint(
            0, vocab, (batch, sequence), device=self.device, dtype=torch.int64)
        return hidden, weight, labels

    def check_gradients(self, actual, expected, bounds, label):
        for name, got, want in zip(("dhidden", "dweight"), actual, expected):
            stats = diagnostics(got, want)
            for key, bound in bounds.items():
                if key == "cosine":
                    self.assertGreater(
                        stats[key], bound,
                        f"{label}/{name} cosine={stats[key]:.7f} below {bound}")
                else:
                    self.assertLessEqual(
                        stats[key], bound,
                        f"{label}/{name} {key}={stats[key]:.3e} exceeds {bound:.3e}")
        return actual

    def public_gradients(self, hidden, weight, labels, **kwargs):
        dloss = kwargs.pop("upstream", torch.ones((), device=self.device))
        x = hidden.detach().clone().requires_grad_()
        w = weight.detach().clone().requires_grad_()
        loss = fused_ce_chunked(x, w, labels, **kwargs)
        grads = torch.autograd.grad(loss, (x, w), grad_outputs=dloss)
        return loss.detach(), grads

    def reference(self, hidden, weight, labels, cu_seqlens=None, **kwargs):
        x = hidden.detach().float().requires_grad_()
        w = weight.detach().float().requires_grad_()
        if cu_seqlens is None:
            loss = reference_fused_cross_entropy(x, w, labels, **kwargs)
        else:
            loss = reference_varlen_fused_cross_entropy(
                x, w, labels, cu_seqlens, **kwargs)
        return loss, (x, w)


class TestChunkedCrossEntropy(ChunkedCETestCase):
    def test_token_mean_matches_fp32_reference(self):
        for chunk_rows in (3, 7, 64):
            for upstream in (1.0, 0.37):
                with self.subTest(chunk_rows=chunk_rows, upstream=upstream):
                    hidden, weight, labels = self.make_inputs(2, 9, 16, 41)
                    labels.flatten()[::4] = -1
                    loss, grads = self.public_gradients(
                        hidden, weight, labels, chunk_rows=chunk_rows,
                        upstream=torch.tensor(upstream, device=self.device))
                    expected, (x, w) = self.reference(
                        hidden, weight, labels, ignore_index=-1)
                    expected.backward(torch.tensor(upstream, device=self.device))
                    torch.testing.assert_close(
                        loss, expected.detach(), rtol=1e-5, atol=1e-5)
                    self.check_gradients(
                        grads, (x.grad, w.grad), _FP32_GRAD, "token-fp32")
                    self.assertEqual(grads[0].dtype, torch.bfloat16
                                     if hidden.dtype == torch.bfloat16
                                     else torch.float32)

    def test_zero_length_and_single_token_documents(self):
        cu_seqlens = torch.tensor(
            [0, 0, 1, 1, 4], device=self.device, dtype=torch.int32)
        hidden, weight, labels = self.make_inputs(1, 4, 8, 13)
        loss, grads = self.public_gradients(
            hidden, weight, labels, cu_seqlens=cu_seqlens, chunk_rows=2,
            ignore_index=-1)
        expected, (x, w) = self.reference(
            hidden, weight, labels, cu_seqlens=cu_seqlens, ignore_index=-1)
        expected.backward()
        torch.testing.assert_close(loss, expected.detach(), rtol=1e-5, atol=1e-5)
        self.check_gradients(
            grads, (x.grad, w.grad), _FP32_GRAD, "empty-documents")

    def test_softcap_matches_fp32_reference(self):
        for softcap in (30.0, 1.0, 0.25):
            with self.subTest(softcap=softcap):
                hidden, weight, labels = self.make_inputs(1, 12, 24, 57)
                labels.flatten()[::5] = -1
                loss, grads = self.public_gradients(
                    hidden, weight, labels, softcap=softcap, chunk_rows=5)
                expected, (x, w) = self.reference(
                    hidden, weight, labels, ignore_index=-1, softcap=softcap)
                expected.backward()
                torch.testing.assert_close(
                    loss, expected.detach(), rtol=1e-5, atol=1e-5)
                self.check_gradients(
                    grads, (x.grad, w.grad), _FP32_GRAD, "softcap")

    def test_saturated_softcap_gradient(self):
        # Every logit is 960 / cap, so tanh saturates exactly, 1 - tanh^2 is
        # exactly zero and the whole softmax gradient must collapse with it.
        channels, vocab = 32, 29
        hidden = torch.full(
            (1, 6, channels), 1.0, device=self.device)
        weight = torch.full((vocab, channels), 30.0, device=self.device)
        labels = torch.randint(
            0, vocab, (1, 6), device=self.device, dtype=torch.int64)
        loss, grads = self.public_gradients(
            hidden, weight, labels, softcap=1.0, chunk_rows=4)
        expected, (x, w) = self.reference(
            hidden, weight, labels, ignore_index=-1, softcap=1.0)
        expected.backward()
        torch.testing.assert_close(loss, expected.detach(), rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(
            loss, torch.log(torch.tensor(float(vocab), device=self.device)),
            rtol=1e-5, atol=1e-5)
        self.assertEqual(int(torch.count_nonzero(grads[0])), 0)
        self.assertEqual(int(torch.count_nonzero(grads[1])), 0)

    def test_mixed_saturation_matches_reference(self):
        # Partly saturated logits (|z| / cap up to ~1700) must follow the same
        # capped softmax in the LSE, the target logit and the gradient.
        hidden, weight, labels = self.make_inputs(1, 8, 32, 29)
        hidden = hidden * 30.0
        loss, grads = self.public_gradients(
            hidden, weight, labels, softcap=0.25, chunk_rows=4)
        expected, (x, w) = self.reference(
            hidden, weight, labels, ignore_index=-1, softcap=0.25)
        expected.backward()
        torch.testing.assert_close(loss, expected.detach(), rtol=1e-5, atol=1e-5)
        self.check_gradients(
            grads, (x.grad, w.grad), _FP32_GRAD, "mixed-saturation")

    def test_document_balanced_matches_fp32_reference(self):
        lengths = [0, 5, 3, 17, 9]
        boundaries = [0]
        for length in lengths:
            boundaries.append(boundaries[-1] + length)
        cu_seqlens = torch.tensor(
            boundaries, device=self.device, dtype=torch.int32)
        for chunk_rows in (4, 7, 64):
            with self.subTest(chunk_rows=chunk_rows):
                hidden, weight, labels = self.make_inputs(
                    1, boundaries[-1], 24, 61)
                # The length-3 document is fully ignored; other documents carry
                # different numbers of ignored tokens, and boundaries are never
                # aligned with the chunk size.
                labels[0, boundaries[2]:boundaries[3]] = -1
                labels[0, boundaries[3] + 1] = -1
                labels[0, boundaries[4] + 2] = -1
                labels[0, 0] = -1
                loss, grads = self.public_gradients(
                    hidden, weight, labels, cu_seqlens=cu_seqlens,
                    chunk_rows=chunk_rows, softcap=2.0)
                expected, (x, w) = self.reference(
                    hidden, weight, labels, cu_seqlens=cu_seqlens,
                    ignore_index=-1, softcap=2.0)
                expected.backward()
                torch.testing.assert_close(
                    loss, expected.detach(), rtol=1e-5, atol=1e-5)
                self.check_gradients(
                    grads, (x.grad, w.grad), _FP32_GRAD, "document-balanced")

    def test_document_normalizers_are_reported(self):
        cu_seqlens = torch.tensor(
            [0, 2, 5, 6], device=self.device, dtype=torch.int64)
        hidden, weight, labels = self.make_inputs(1, 6, 16, 23)
        labels[0, 1] = -1
        labels[0, 5] = -1  # whole final document ignored
        state = chunked_ce_forward(
            hidden, weight, labels, ignore_index=-1, cu_seqlens=cu_seqlens)
        self.assertEqual(state.reduction, "document_mean")
        self.assertEqual(int(state.count.item()), 2)
        expected = reference_varlen_fused_cross_entropy(
            hidden, weight, labels, cu_seqlens, ignore_index=-1)
        torch.testing.assert_close(
            state.loss, expected, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(
            state.mean_sum, expected * 2, rtol=1e-5, atol=1e-5)
        self.assertEqual(state.document_loss.shape, (3,))
        self.assertEqual(float(state.document_loss[2]), 0.0)
        valid = state.document_loss[[0, 1]]
        torch.testing.assert_close(
            valid.mean(), expected, rtol=1e-5, atol=1e-5)

        token_state = chunked_ce_forward(hidden, weight, labels, ignore_index=-1)
        self.assertEqual(token_state.reduction, "token_mean")
        self.assertEqual(int(token_state.count.item()), 4)
        token_expected = reference_fused_cross_entropy(
            hidden, weight, labels, ignore_index=-1)
        torch.testing.assert_close(
            token_state.loss, token_expected, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(
            token_state.mean_sum, token_expected * 4, rtol=1e-5, atol=1e-5)

    def test_microbatch_split_with_external_normalization(self):
        boundaries = torch.tensor(
            [0, 6, 9, 20, 21, 30], device=self.device, dtype=torch.int32)
        hidden, weight, labels = self.make_inputs(1, 30, 16, 37)
        labels[0, 2] = -1
        labels[0, 9:15] = -1
        labels[0, 21:27] = -1
        # Split at a document boundary into unequal valid-document counts.
        split = int(boundaries[3].item())
        whole_loss, whole_grads = self.public_gradients(
            hidden, weight, labels, cu_seqlens=boundaries, chunk_rows=4)

        parts = []
        for lo, hi in ((0, split), (split, labels.shape[1])):
            cu_part = torch.tensor(
                boundaries[(boundaries >= lo) & (boundaries <= hi)].tolist(),
                device=self.device, dtype=torch.int32) - lo
            state = chunked_ce_forward(
                hidden[:, lo:hi].contiguous(), weight,
                labels[:, lo:hi].contiguous(), ignore_index=-1,
                cu_seqlens=cu_part, chunk_rows=4, normalize=False)
            parts.append(state)
        total = (sum(float(p.mean_sum) for p in parts)
                 * torch.ones((), device=self.device))
        documents = sum(int(p.count) for p in parts)
        combined = total / documents
        torch.testing.assert_close(combined, whole_loss, rtol=1e-5, atol=1e-5)
        whole_loss_again, whole_sum, whole_documents = fused_ce_chunked_stats(
            hidden, weight, labels, ignore_index=-1, cu_seqlens=boundaries,
            chunk_rows=4)
        torch.testing.assert_close(
            whole_loss_again, whole_loss, rtol=1e-6, atol=0.0)
        self.assertEqual(int(whole_documents), documents)
        torch.testing.assert_close(
            whole_sum, whole_loss * documents, rtol=1e-5, atol=0.0)

        combined_dhidden = torch.zeros_like(whole_grads[0])
        combined_dweight = torch.zeros_like(whole_grads[1])
        for part, (lo, hi) in zip(parts, ((0, split), (split, labels.shape[1]))):
            dx, dw = chunked_ce_backward(
                part, torch.ones((), device=self.device), torch.float32,
                torch.float32)
            combined_dhidden[0, lo:hi] = dx
            combined_dweight += dw
        combined_dhidden /= documents
        combined_dweight /= documents
        for got, want, name in ((combined_dhidden, whole_grads[0], "dhidden"),
                                (combined_dweight, whole_grads[1], "dweight")):
            stats = diagnostics(got, want)
            self.assertLess(stats["relative_l2"], 1e-5, f"{name} {stats}")

        # The naive average of microbatch document means is *not* the global
        # document mean when the two microbatches carry different document
        # counts; this is exactly what the reported (sum, count) pair fixes.
        naive = sum(float(p.loss) for p in parts) / len(parts)
        self.assertGreater(abs(naive - float(whole_loss)), 1e-3)

    def test_upstream_gradient_scales_linearly(self):
        hidden, weight, labels = self.make_inputs(1, 14, 16, 31)
        cu_seqlens = torch.tensor(
            [0, 3, 8, 14], device=self.device, dtype=torch.int32)
        loss, unit = self.public_gradients(
            hidden, weight, labels, cu_seqlens=cu_seqlens, softcap=30.0,
            chunk_rows=4)
        _, quarter = self.public_gradients(
            hidden, weight, labels, cu_seqlens=cu_seqlens, softcap=30.0,
            chunk_rows=4, upstream=torch.tensor(0.25, device=self.device))
        for got, want, name in zip(quarter, unit, ("dhidden", "dweight")):
            torch.testing.assert_close(
                got, (want.float() * 0.25).to(got.dtype), rtol=0, atol=0)
        self.assertAlmostEqual(float(loss), float(loss))

    def test_repeated_backward_keeps_saved_gradients(self):
        hidden, weight, labels = self.make_inputs(1, 11, 16, 29)
        x = hidden.clone().requires_grad_()
        w = weight.clone().requires_grad_()
        loss = fused_ce_chunked(x, w, labels, softcap=3.0, chunk_rows=3)
        first = torch.autograd.grad(loss, (x, w), retain_graph=True)
        second = torch.autograd.grad(
            loss, (x, w), retain_graph=True,
            grad_outputs=torch.tensor(2.0, device=self.device))
        third = torch.autograd.grad(loss, (x, w))
        for got, want in zip(second, first):
            torch.testing.assert_close(got, want.float() * 2, rtol=0, atol=0)
        for got, want in zip(third, first):
            torch.testing.assert_close(got, want, rtol=0, atol=0)

    def test_all_ignored_is_zero_without_nan(self):
        for softcap in (None, 30.0):
            with self.subTest(softcap=softcap):
                cu_seqlens = torch.tensor(
                    [0, 3, 7], device=self.device, dtype=torch.int32)
                hidden, weight, labels = self.make_inputs(1, 7, 16, 19)
                labels.fill_(-1)
                state = chunked_ce_forward(
                    hidden, weight, labels, ignore_index=-1, softcap=softcap,
                    cu_seqlens=cu_seqlens, chunk_rows=2)
                self.assertTrue(torch.isfinite(state.loss))
                self.assertEqual(float(state.loss), 0.0)
                self.assertEqual(int(state.count), 0)
                self.assertEqual(float(state.mean_sum), 0.0)
                dx, dw = chunked_ce_backward(
                    state, torch.ones((), device=self.device), torch.float32,
                    torch.float32)
                self.assertTrue(bool(torch.isfinite(dx).all()))
                self.assertTrue(bool(torch.isfinite(dw).all()))
                self.assertEqual(int(torch.count_nonzero(dx)), 0)
                self.assertEqual(int(torch.count_nonzero(dw)), 0)
                token_state = chunked_ce_forward(
                    hidden, weight, labels, ignore_index=-1, softcap=softcap)
                self.assertEqual(float(token_state.loss), 0.0)
                self.assertEqual(int(token_state.count), 0)

    def test_bf16_paths_and_gradient_dtypes(self):
        hidden, weight, labels = self.make_inputs(
            1, 21, 32, 73, dtype=torch.bfloat16)
        labels.flatten()[::6] = -1
        loss, grads = self.public_gradients(
            hidden, weight, labels, chunk_rows=5, softcap=30.0)
        expected, (x, w) = self.reference(
            hidden, weight, labels, ignore_index=-1, softcap=30.0)
        expected.backward()
        self.assertLess(
            abs(float(loss) - float(expected)) / float(expected.abs()), 1e-2)
        self.check_gradients(grads, (x.grad, w.grad), _BF16_GRAD, "bf16")
        self.assertEqual(grads[0].dtype, torch.bfloat16)
        self.assertEqual(grads[1].dtype, torch.bfloat16)

        state = chunked_ce_forward(
            hidden, weight, labels, ignore_index=-1, chunk_rows=5,
            softcap=30.0)
        self.assertEqual(state.dhidden.dtype, torch.bfloat16)
        self.assertEqual(state.dweight.dtype, torch.float32)
        self.assertEqual(state.count.dtype, torch.int32)
        self.assertEqual(state.mean_sum.dtype, torch.float32)
        # FP32 chunk buffers from BF16 inputs keep only the input rounding.
        precise = chunked_ce_forward(
            hidden, weight, labels, ignore_index=-1, chunk_rows=5,
            softcap=30.0, logits_dtype=torch.float32)
        precise_loss = float(precise.loss)
        torch.testing.assert_close(
            torch.tensor(precise_loss, device=self.device), expected.detach(),
            rtol=1e-5, atol=1e-5)
        dx, dw = chunked_ce_backward(
            precise, torch.ones((), device=self.device), torch.float32,
            torch.float32)
        self.check_gradients(
            (dx, dw), (x.grad, w.grad), {"cosine": 0.9999}, "bf16/fp32-buffer")

    def test_compiled_matches_eager(self):
        hidden, weight, labels = self.make_inputs(
            1, 24, 16, 43, dtype=torch.bfloat16)
        labels[0, ::7] = -1
        cu_seqlens = torch.tensor(
            [0, 5, 11, 24], device=self.device, dtype=torch.int32)

        def entry(x, w):
            return fused_ce_chunked(
                x, w, labels, ignore_index=-1, softcap=30.0,
                cu_seqlens=cu_seqlens, max_seqlen=13, chunk_rows=5)

        compiled = torch.compile(entry, fullgraph=True)
        x = hidden.clone().requires_grad_()
        w = weight.clone().requires_grad_()
        eager_loss = entry(x, w)
        eager = torch.autograd.grad(eager_loss, (x, w), retain_graph=True)
        compiled_loss = compiled(x, w)
        got = torch.autograd.grad(compiled_loss, (x, w))
        torch.testing.assert_close(compiled_loss, eager_loss, rtol=0, atol=0)
        for actual, expected in zip(got, eager):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_int32_labels_and_default_chunking(self):
        hidden, weight, labels = self.make_inputs(1, 9, 16, 1000)
        labels = labels.to(torch.int32)
        labels.flatten()[3] = -1
        loss, _ = self.public_gradients(hidden, weight, labels)
        expected = reference_fused_cross_entropy(
            hidden, weight, labels, ignore_index=-1)
        torch.testing.assert_close(loss, expected, rtol=1e-5, atol=1e-5)
        self.assertEqual(
            chunked_ce_forward(hidden, weight, labels).dhidden.dtype,
            torch.float32)

    def test_validation(self):
        hidden, weight, labels = self.make_inputs(1, 5, 16, 17)
        with self.assertRaisesRegex(TypeError, "share one dtype"):
            fused_ce_chunked(
                hidden, weight.bfloat16(), labels)
        with self.assertRaisesRegex(TypeError, "dtype"):
            fused_ce_chunked(
                hidden.bfloat16(), weight.bfloat16(), labels,
                logits_dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, "round FP32 inputs"):
            fused_ce_chunked(
                hidden, weight, labels, logits_dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "every label"):
            bad = labels.clone()
            bad[0, 0] = 17
            fused_ce_chunked(hidden, weight, bad)
        with self.assertRaisesRegex(ValueError, "chunk_rows"):
            fused_ce_chunked(hidden, weight, labels, chunk_rows=0)
        with self.assertRaisesRegex(TypeError, "chunk_rows"):
            fused_ce_chunked(hidden, weight, labels, chunk_rows=2.5)
        for softcap in (0.0, -1.0, float("inf"), float("nan")):
            with self.subTest(softcap=softcap):
                with self.assertRaisesRegex(ValueError, "softcap"):
                    fused_ce_chunked(hidden, weight, labels, softcap=softcap)
        with self.assertRaisesRegex(TypeError, "softcap"):
            fused_ce_chunked(hidden, weight, labels, softcap="30")
        with self.assertRaisesRegex(ValueError, "requires cu_seqlens"):
            fused_ce_chunked(hidden, weight, labels, max_seqlen=4)
        cu_seqlens = torch.tensor(
            [0, 2, 5], device=self.device, dtype=torch.int32)
        with self.assertRaisesRegex(ValueError, "batch size 1"):
            fused_ce_chunked(
                hidden.expand(2, -1, -1).contiguous(),
                weight, labels.expand(2, -1).contiguous(),
                cu_seqlens=cu_seqlens)
        with self.assertRaisesRegex(ValueError, "nondecreasing"):
            fused_ce_chunked(
                hidden, weight, labels,
                cu_seqlens=torch.tensor(
                    [0, 5, 2], device=self.device, dtype=torch.int32))
        with self.assertRaisesRegex(ValueError, "max_seqlen"):
            fused_ce_chunked(
                hidden, weight, labels, cu_seqlens=cu_seqlens, max_seqlen=1)


if __name__ == "__main__":
    unittest.main()
