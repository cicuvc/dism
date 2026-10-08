import unittest

import torch

from linear_rmsnorm_rope import (
    FusedLinearRMSNormRoPE,
    linear_rmsnorm_rope,
    reference_linear_rmsnorm_rope,
    reference_varlen_linear_rmsnorm_rope,
    torch_bwd_linear_rmsnorm_rope,
    torch_fwd_linear_rmsnorm_rope,
)


def make_rope(sequence, head_dim, device, dtype):
    position = torch.arange(sequence, device=device, dtype=torch.float32)
    frequency = torch.arange(
        head_dim // 2, device=device, dtype=torch.float32)
    frequency = 1.0 / (10000 ** (2 * frequency / head_dim))
    angle = position[:, None] * frequency[None, :]
    return angle.cos().to(dtype), angle.sin().to(dtype)


def relative_mean_error(actual, expected):
    error = (actual.float() - expected.float()).abs().mean()
    return (error / expected.float().abs().mean().clamp_min(1e-12)).item()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestLinearRMSNormRoPE(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2026)
        self.device = torch.device("cuda")

    def make_inputs(self, batch, sequence, in_channels, heads, head_dim, dtype):
        x = torch.randn(
            batch, sequence, in_channels, device=self.device,
            dtype=dtype, requires_grad=True)
        weight = torch.randn(
            heads * head_dim, in_channels, device=self.device,
            dtype=dtype, requires_grad=True)
        rms_weight = torch.randn(
            heads, head_dim, device=self.device,
            dtype=dtype, requires_grad=True)
        cos, sin = make_rope(sequence, head_dim, self.device, dtype)
        return x, weight, rms_weight, cos, sin

    def assert_float_close(self, actual, expected):
        torch.testing.assert_close(actual, expected, rtol=3e-4, atol=4e-4)

    def test_float32_forward_and_backward_shapes(self):
        shapes = (
            (1, 1, 33, 1, 32),
            (2, 17, 33, 3, 64),
            (1, 65, 129, 5, 128),
        )
        for shape in shapes:
            with self.subTest(shape=shape):
                x, weight, rms_weight, cos, sin = self.make_inputs(
                    *shape, torch.float32)
                x_ref = x.detach().clone().requires_grad_()
                weight_ref = weight.detach().clone().requires_grad_()
                rms_ref = rms_weight.detach().clone().requires_grad_()

                output = linear_rmsnorm_rope(
                    x, weight, rms_weight, cos, sin)
                expected = reference_linear_rmsnorm_rope(
                    x_ref, weight_ref, rms_ref, cos, sin)
                doutput = torch.randn_like(output)
                output.backward(doutput)
                expected.backward(doutput)

                self.assertEqual(output.shape, (*shape[:2], shape[3], shape[4]))
                self.assert_float_close(output, expected)
                self.assert_float_close(x.grad, x_ref.grad)
                self.assert_float_close(weight.grad, weight_ref.grad)
                self.assert_float_close(rms_weight.grad, rms_ref.grad)

    def test_split_half_rope_and_identity(self):
        batch, sequence, in_channels, heads, head_dim = 1, 7, 16, 2, 32
        x, weight, rms_weight, _, _ = self.make_inputs(
            batch, sequence, in_channels, heads, head_dim, torch.float32)
        cos = torch.ones(
            sequence, head_dim // 2, device=self.device)
        sin = torch.zeros_like(cos)
        output = torch_fwd_linear_rmsnorm_rope(
            x.detach(), weight.detach(), rms_weight.detach(), cos, sin)

        projected = torch.nn.functional.linear(x, weight).unflatten(
            -1, (heads, head_dim))
        expected = projected * torch.rsqrt(
            projected.square().mean(-1, keepdim=True) + 1e-6)
        expected = expected * rms_weight
        self.assert_float_close(output, expected)

        # A quarter turn makes the split-half convention explicit: [a,b] -> [-b,a].
        zero = torch.zeros_like(cos)
        one = torch.ones_like(cos)
        rotated = torch_fwd_linear_rmsnorm_rope(
            x.detach(), weight.detach(), rms_weight.detach(), zero, one)
        first, second = expected.chunk(2, dim=-1)
        self.assert_float_close(rotated, torch.cat((-second, first), dim=-1))

    def test_module_and_parameter_gradients(self):
        module = FusedLinearRMSNormRoPE(
            33, 3, 64, device=self.device, dtype=torch.float32)
        x = torch.randn(
            2, 17, 33, device=self.device, requires_grad=True)
        cos, sin = make_rope(17, 64, self.device, torch.float32)
        output = module(x, cos, sin)
        output.square().mean().backward()
        self.assertEqual(output.shape, (2, 17, 3, 64))
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(module.weight.grad)
        self.assertIsNotNone(module.rms_weight.grad)
        self.assertEqual(
            repr(module),
            "FusedLinearRMSNormRoPE(in_channels=33, num_heads=3, "
            "head_dim=64, eps=1e-06)")

    def test_bfloat16_autocast(self):
        module = FusedLinearRMSNormRoPE(
            64, 4, 64, device=self.device, dtype=torch.float32)
        x = torch.randn(
            2, 65, 64, device=self.device, dtype=torch.float32,
            requires_grad=True)
        cos, sin = make_rope(65, 64, self.device, torch.float32)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = module(x, cos, sin)
            loss = output.square().mean()
        self.assertEqual(output.dtype, torch.bfloat16)
        loss.backward()
        self.assertEqual(x.grad.dtype, torch.float32)
        self.assertEqual(module.weight.grad.dtype, torch.float32)
        self.assertEqual(module.rms_weight.grad.dtype, torch.float32)

    def test_varlen_forward_backward_and_empty_group(self):
        lengths = [0, 1, 17, 31, 33]
        boundaries = [0]
        for length in lengths:
            boundaries.append(boundaries[-1] + length)
        cu_seqlens = torch.tensor(
            boundaries, device=self.device, dtype=torch.int32)
        shape = (1, boundaries[-1], 33, 3, 64)
        x, weight, rms_weight, cos, sin = self.make_inputs(
            *shape, torch.float32)
        # Varlen cos/sin cover positions within a group, not total tokens.
        cos, sin = make_rope(
            max(lengths), shape[-1], self.device, torch.float32)
        x_ref = x.detach().clone().requires_grad_()
        weight_ref = weight.detach().clone().requires_grad_()
        rms_ref = rms_weight.detach().clone().requires_grad_()

        output = linear_rmsnorm_rope(
            x, weight, rms_weight, cos, sin,
            cu_seqlens=cu_seqlens, max_seqlen=max(lengths))
        expected = reference_varlen_linear_rmsnorm_rope(
            x_ref, weight_ref, rms_ref, cos, sin, cu_seqlens)
        doutput = torch.randn_like(output)
        output.backward(doutput)
        expected.backward(doutput)

        self.assert_float_close(output, expected)
        self.assert_float_close(x.grad, x_ref.grad)
        self.assert_float_close(weight.grad, weight_ref.grad)
        self.assert_float_close(rms_weight.grad, rms_ref.grad)

    def test_varlen_autocast_and_inferred_max_seqlen(self):
        lengths = [3, 16, 5]
        cu_seqlens = torch.tensor(
            [0, 3, 19, 24], device=self.device, dtype=torch.int32)
        module = FusedLinearRMSNormRoPE(
            32, 2, 32, device=self.device, dtype=torch.float32)
        x = torch.randn(
            1, sum(lengths), 32, device=self.device,
            dtype=torch.float32, requires_grad=True)
        cos, sin = make_rope(
            max(lengths), 32, self.device, torch.float32)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = module(x, cos, sin, cu_seqlens=cu_seqlens)
        self.assertEqual(output.dtype, torch.bfloat16)
        output.float().square().mean().backward()
        self.assertEqual(x.grad.dtype, torch.float32)
        self.assertEqual(module.weight.grad.dtype, torch.float32)

    def test_bfloat16_dlinear_error(self):
        # This exercises a large reduction after the BF16 dlinear temporary.
        shape = (2, 129, 512, 8, 128)
        x, weight, rms_weight, cos, sin = self.make_inputs(
            *shape, torch.bfloat16)
        doutput = torch.randn(
            *shape[:2], shape[3], shape[4],
            device=self.device, dtype=torch.bfloat16)
        got = torch_bwd_linear_rmsnorm_rope(
            x.detach(), weight.detach(), rms_weight.detach(),
            cos, sin, doutput)

        x_ref = x.detach().float().requires_grad_()
        weight_ref = weight.detach().float().requires_grad_()
        rms_ref = rms_weight.detach().float().requires_grad_()
        expected = reference_linear_rmsnorm_rope(
            x_ref, weight_ref, rms_ref, cos.float(), sin.float())
        expected.backward(doutput.float())

        self.assertLess(relative_mean_error(got[0], x_ref.grad), 0.004)
        self.assertLess(relative_mean_error(got[1], weight_ref.grad), 0.004)
        self.assertLess(relative_mean_error(got[2], rms_ref.grad), 1e-5)

    def test_validation(self):
        x = torch.randn(1, 4, 16, device=self.device)
        weight = torch.randn(48, 16, device=self.device)
        rms_weight = torch.randn(3, 16, device=self.device)
        cos, sin = make_rope(4, 16, self.device, torch.float32)
        with self.assertRaisesRegex(ValueError, "power of two"):
            torch_fwd_linear_rmsnorm_rope(
                x, torch.randn(72, 16, device=self.device),
                torch.randn(3, 24, device=self.device),
                torch.randn(4, 12, device=self.device),
                torch.randn(4, 12, device=self.device))
        with self.assertRaisesRegex(ValueError, "weight must have shape"):
            torch_fwd_linear_rmsnorm_rope(
                x, weight[:-1], rms_weight, cos, sin)


if __name__ == "__main__":
    unittest.main()
