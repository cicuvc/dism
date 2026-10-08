import unittest

import torch

from fused_cross_entropy import (
    fused_cross_entropy,
    fused_cross_entropy_per_position,
    reference_fused_cross_entropy,
    reference_fused_cross_entropy_per_position,
    reference_varlen_fused_cross_entropy,
    reference_varlen_fused_cross_entropy_per_position,
    torch_bwd_fused_cross_entropy,
    torch_fwd_fused_cross_entropy,
    torch_fwd_varlen_fused_cross_entropy,
)


def relative_mean_error(actual, expected):
    error = (actual.float() - expected.float()).abs().mean()
    return (error / expected.float().abs().mean().clamp_min(1e-12)).item()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestFusedCrossEntropy(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2026)
        self.device = torch.device("cuda")

    def make_inputs(self, batch, sequence, channels, vocab):
        hidden = torch.randn(
            batch, sequence, channels, device=self.device,
            dtype=torch.bfloat16)
        weight = torch.randn(
            vocab, channels, device=self.device, dtype=torch.bfloat16)
        labels = torch.randint(
            vocab, (batch, sequence), device=self.device,
            dtype=torch.int64)
        return hidden, weight, labels

    def test_mean_forward_and_backward_with_ignored_labels(self):
        for shape in ((1, 2, 16, 17), (2, 7, 33, 67), (2, 17, 64, 257)):
            with self.subTest(shape=shape):
                hidden, weight, labels = self.make_inputs(*shape)
                labels.flatten()[::5] = -100
                hidden.requires_grad_()
                weight.requires_grad_()
                hidden_ref = hidden.detach().float().requires_grad_()
                weight_ref = weight.detach().float().requires_grad_()

                loss = fused_cross_entropy(
                    hidden, weight, labels, chunk_size=3)
                expected = reference_fused_cross_entropy(
                    hidden_ref, weight_ref, labels)
                scale = torch.tensor(1.7, device=self.device)
                loss.backward(scale)
                expected.backward(scale)

                torch.testing.assert_close(
                    loss, expected, rtol=2e-5, atol=2e-5)
                self.assertLess(
                    relative_mean_error(hidden.grad, hidden_ref.grad), 0.004)
                self.assertLess(
                    relative_mean_error(weight.grad, weight_ref.grad), 0.004)
                self.assertEqual(hidden.grad.dtype, torch.bfloat16)
                self.assertEqual(weight.grad.dtype, torch.bfloat16)

    def test_manual_backward_returns_fp32_weight_gradient(self):
        hidden, weight, labels = self.make_inputs(2, 9, 33, 129)
        labels[0, 1] = -7
        labels[1, 4] = -7
        loss, lse, count = torch_fwd_fused_cross_entropy(
            hidden, weight, labels, ignore_index=-7)
        dloss = torch.tensor(0.3, device=self.device)
        dhidden, dweight = torch_bwd_fused_cross_entropy(
            hidden, weight, labels, lse, count, dloss,
            ignore_index=-7, chunk_size=4)

        hidden_ref = hidden.float().requires_grad_()
        weight_ref = weight.float().requires_grad_()
        expected = reference_fused_cross_entropy(
            hidden_ref, weight_ref, labels, ignore_index=-7)
        expected.backward(dloss)

        self.assertEqual(dhidden.dtype, torch.bfloat16)
        self.assertEqual(dweight.dtype, torch.float32)
        self.assertLess(relative_mean_error(dhidden, hidden_ref.grad), 0.004)
        self.assertLess(relative_mean_error(dweight, weight_ref.grad), 0.004)
        torch.testing.assert_close(loss, expected, rtol=2e-5, atol=2e-5)

    def test_per_position_uses_valid_batch_count(self):
        hidden, weight, labels = self.make_inputs(3, 5, 31, 71)
        labels[0, 0] = -100
        labels[:2, 2] = -100
        labels[:, 4] = -100
        output = fused_cross_entropy_per_position(hidden, weight, labels)
        expected = reference_fused_cross_entropy_per_position(
            hidden, weight, labels)
        torch.testing.assert_close(output, expected, rtol=2e-5, atol=2e-5)
        self.assertEqual(output.dtype, torch.float32)
        self.assertEqual(output[4].item(), 0.0)
        self.assertFalse(output.requires_grad)

    def test_softcap_forward_backward_and_per_position(self):
        for softcap in (30.0, 1.0, 0.25):
            with self.subTest(softcap=softcap):
                hidden, weight, labels = self.make_inputs(2, 17, 64, 257)
                labels.flatten()[::6] = -100
                hidden.requires_grad_()
                weight.requires_grad_()
                hidden_ref = hidden.detach().float().requires_grad_()
                weight_ref = weight.detach().float().requires_grad_()

                loss = fused_cross_entropy(
                    hidden, weight, labels, chunk_size=5,
                    softcap=softcap)
                expected = reference_fused_cross_entropy(
                    hidden_ref, weight_ref, labels, softcap=softcap)
                loss.backward()
                expected.backward()

                torch.testing.assert_close(
                    loss, expected, rtol=2e-5, atol=2e-5)
                self.assertLess(
                    relative_mean_error(hidden.grad, hidden_ref.grad), 0.004)
                self.assertLess(
                    relative_mean_error(weight.grad, weight_ref.grad), 0.004)

                position_loss = fused_cross_entropy_per_position(
                    hidden.detach(), weight.detach(), labels,
                    softcap=softcap)
                position_expected = (
                    reference_fused_cross_entropy_per_position(
                        hidden.detach(), weight.detach(), labels,
                        softcap=softcap))
                torch.testing.assert_close(
                    position_loss, position_expected,
                    rtol=2e-5, atol=2e-5)

    def test_varlen_sequence_balanced_forward_and_backward(self):
        lengths = [0, 1, 3, 7, 4]
        boundaries = [0]
        for length in lengths:
            boundaries.append(boundaries[-1] + length)
        cu_seqlens = torch.tensor(
            boundaries, device=self.device, dtype=torch.int32)

        for softcap in (None, 2.0):
            with self.subTest(softcap=softcap):
                hidden, weight, labels = self.make_inputs(
                    1, boundaries[-1], 33, 67)
                # The length-3 document is fully ignored. Other documents
                # contain different numbers of ignored and valid tokens.
                labels[0, boundaries[2]:boundaries[3]] = -100
                labels[0, boundaries[3] + 1] = -100
                labels[0, boundaries[4] + 2] = -100
                hidden.requires_grad_()
                weight.requires_grad_()
                hidden_ref = hidden.detach().float().requires_grad_()
                weight_ref = weight.detach().float().requires_grad_()

                loss = fused_cross_entropy(
                    hidden, weight, labels, chunk_size=3,
                    softcap=softcap, cu_seqlens=cu_seqlens)
                expected = reference_varlen_fused_cross_entropy(
                    hidden_ref, weight_ref, labels, cu_seqlens,
                    softcap=softcap)
                scale = torch.tensor(0.7, device=self.device)
                loss.backward(scale)
                expected.backward(scale)

                torch.testing.assert_close(
                    loss, expected, rtol=2e-5, atol=2e-5)
                self.assertLess(
                    relative_mean_error(hidden.grad, hidden_ref.grad), 0.004)
                self.assertLess(
                    relative_mean_error(weight.grad, weight_ref.grad), 0.004)

    def test_varlen_normalization_weights_and_per_position(self):
        lengths = [2, 3, 1]
        cu_seqlens = torch.tensor(
            [0, 2, 5, 6], device=self.device, dtype=torch.int64)
        hidden, weight, labels = self.make_inputs(1, 6, 31, 71)
        labels[0, 1] = -100
        labels[0, 3] = -100
        labels[0, 5] = -100  # Entire final document is ignored.

        loss, _, valid_documents, token_inv_count = (
            torch_fwd_varlen_fused_cross_entropy(
                hidden, weight, labels, cu_seqlens))
        torch.testing.assert_close(
            token_inv_count,
            torch.tensor(
                [1.0, 0.0, 0.5, 0.0, 0.5, 0.0],
                device=self.device))
        self.assertEqual(valid_documents.item(), 2)
        expected = reference_varlen_fused_cross_entropy(
            hidden, weight, labels, cu_seqlens)
        torch.testing.assert_close(loss, expected, rtol=2e-5, atol=2e-5)

        output = fused_cross_entropy_per_position(
            hidden, weight, labels, cu_seqlens=cu_seqlens,
            max_seqlen=5)
        position_expected = reference_varlen_fused_cross_entropy_per_position(
            hidden, weight, labels, cu_seqlens, max_seqlen=5)
        torch.testing.assert_close(
            output, position_expected, rtol=2e-5, atol=2e-5)
        self.assertEqual(output.shape, (5,))
        self.assertEqual(output[-1].item(), 0.0)

    def test_varlen_all_documents_ignored(self):
        cu_seqlens = torch.tensor(
            [0, 0, 2, 5], device=self.device, dtype=torch.int32)
        hidden, weight, labels = self.make_inputs(1, 5, 16, 33)
        labels.fill_(-100)
        hidden.requires_grad_()
        weight.requires_grad_()
        loss = fused_cross_entropy(
            hidden, weight, labels, softcap=1.0,
            cu_seqlens=cu_seqlens)
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        self.assertEqual(torch.count_nonzero(hidden.grad).item(), 0)
        self.assertEqual(torch.count_nonzero(weight.grad).item(), 0)

    def test_all_ignored_is_zero_with_zero_gradients(self):
        hidden, weight, labels = self.make_inputs(2, 3, 16, 33)
        labels.fill_(-100)
        hidden.requires_grad_()
        weight.requires_grad_()
        loss = fused_cross_entropy(
            hidden, weight, labels, chunk_size=2, softcap=1.0)
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        self.assertEqual(torch.count_nonzero(hidden.grad).item(), 0)
        self.assertEqual(torch.count_nonzero(weight.grad).item(), 0)

    def test_int32_labels_and_default_chunking(self):
        hidden, weight, labels = self.make_inputs(1, 11, 32, 1000)
        labels = labels.to(torch.int32)
        labels.flatten()[3] = -100
        output = fused_cross_entropy(hidden, weight, labels)
        expected = reference_fused_cross_entropy(hidden, weight, labels)
        torch.testing.assert_close(output, expected, rtol=2e-5, atol=2e-5)

    def test_validation(self):
        hidden, weight, labels = self.make_inputs(1, 3, 16, 17)
        with self.assertRaisesRegex(TypeError, "bfloat16"):
            fused_cross_entropy(hidden.float(), weight, labels)
        with self.assertRaisesRegex(ValueError, "every label"):
            bad_labels = labels.clone()
            bad_labels[0, 0] = 17
            fused_cross_entropy(hidden, weight, bad_labels)
        with self.assertRaisesRegex(ValueError, "chunk_size"):
            fused_cross_entropy(hidden, weight, labels, chunk_size=0)
        for softcap in (0.0, -1.0, float("inf"), float("nan")):
            with self.subTest(softcap=softcap):
                with self.assertRaisesRegex(ValueError, "softcap"):
                    fused_cross_entropy(
                        hidden, weight, labels, softcap=softcap)
        with self.assertRaisesRegex(TypeError, "softcap"):
            fused_cross_entropy(hidden, weight, labels, softcap="30")

        cu_seqlens = torch.tensor(
            [0, 1, 3], device=self.device, dtype=torch.int32)
        with self.assertRaisesRegex(ValueError, "batch size 1"):
            fused_cross_entropy(
                hidden.expand(2, -1, -1).contiguous(), weight,
                labels.expand(2, -1).contiguous(),
                cu_seqlens=cu_seqlens)
        with self.assertRaisesRegex(ValueError, "nondecreasing"):
            fused_cross_entropy(
                hidden, weight, labels,
                cu_seqlens=torch.tensor(
                    [0, 3, 2], device=self.device, dtype=torch.int32))
        with self.assertRaisesRegex(ValueError, "max_seqlen"):
            fused_cross_entropy(
                hidden, weight, labels, cu_seqlens=cu_seqlens,
                max_seqlen=1)


if __name__ == "__main__":
    unittest.main()
