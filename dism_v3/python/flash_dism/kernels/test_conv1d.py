import unittest

import torch

from conv1d import CausalShortConv1d, short_conv_silu


def reference_short_conv(x, weight, conv_weight, input_state=None):
    batch, n_seq, _ = x.shape
    out_channels = weight.shape[0]
    state_size = conv_weight.shape[0] - 1
    if input_state is None:
        input_state = torch.zeros(
            batch, state_size, out_channels, dtype=x.dtype, device=x.device)
    projected = torch.nn.functional.linear(x, weight)
    history = torch.cat((input_state, projected), dim=1)
    conv = sum(
        history[:, state_size - i:state_size - i + n_seq] * conv_weight[i]
        for i in range(conv_weight.shape[0])
    )
    output_state = history[:, -state_size:] if state_size else history[:, :0]
    return torch.nn.functional.silu(conv), output_state


def reference_varlen_short_conv(
    x, weight, conv_weight, cu_seqlens, input_state=None,
):
    outputs = []
    output_states = []
    boundaries = cu_seqlens.cpu().tolist()
    for group, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        state = None if input_state is None else input_state[group:group + 1]
        output, output_state = reference_short_conv(
            x[:, start:end], weight, conv_weight, state)
        outputs.append(output)
        output_states.append(output_state)
    return torch.cat(outputs, dim=1), torch.cat(output_states, dim=0)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestCausalShortConv1d(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.backends.cuda.matmul.allow_tf32 = False

    def setUp(self):
        torch.manual_seed(1234)
        self.device = torch.device("cuda")
        self.in_channels = 33
        self.out_channels = 47
        self.kernel_size = 4

    def make_inputs(self, batch, n_seq, dtype=torch.float32, state=False):
        x = torch.randn(
            batch, n_seq, self.in_channels, device=self.device,
            dtype=dtype, requires_grad=True)
        weight = torch.randn(
            self.out_channels, self.in_channels, device=self.device,
            dtype=dtype, requires_grad=True)
        conv_weight = torch.randn(
            self.kernel_size, self.out_channels, device=self.device,
            dtype=dtype, requires_grad=True)
        input_state = None
        if state:
            input_state = torch.randn(
                batch, self.kernel_size - 1, self.out_channels,
                device=self.device, dtype=dtype, requires_grad=True)
        return x, weight, conv_weight, input_state

    def assert_float_close(self, actual, expected):
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=3e-4)

    def test_fixed_forward_and_backward_without_state(self):
        x, weight, conv_weight, _ = self.make_inputs(2, 65)
        x_ref = x.detach().clone().requires_grad_()
        weight_ref = weight.detach().clone().requires_grad_()
        conv_weight_ref = conv_weight.detach().clone().requires_grad_()

        output = short_conv_silu(x, weight, conv_weight)
        output_ref, _ = reference_short_conv(
            x_ref, weight_ref, conv_weight_ref)
        dout = torch.randn_like(output)
        output.backward(dout)
        output_ref.backward(dout)

        self.assert_float_close(output, output_ref)
        self.assert_float_close(x.grad, x_ref.grad)
        self.assert_float_close(weight.grad, weight_ref.grad)
        self.assert_float_close(conv_weight.grad, conv_weight_ref.grad)

    def test_state_forward_and_full_backward(self):
        # 65 exercises a state spanning the current block and its halo.
        x, weight, conv_weight, input_state = self.make_inputs(2, 65, state=True)
        x_ref = x.detach().clone().requires_grad_()
        weight_ref = weight.detach().clone().requires_grad_()
        conv_weight_ref = conv_weight.detach().clone().requires_grad_()
        state_ref = input_state.detach().clone().requires_grad_()

        output, output_state = short_conv_silu(
            x, weight, conv_weight, input_state=input_state)
        output_ref, output_state_ref = reference_short_conv(
            x_ref, weight_ref, conv_weight_ref, state_ref)
        dout = torch.randn_like(output)
        doutput_state = torch.randn_like(output_state)
        torch.autograd.backward((output, output_state), (dout, doutput_state))
        torch.autograd.backward(
            (output_ref, output_state_ref), (dout, doutput_state))

        self.assert_float_close(output, output_ref)
        self.assert_float_close(output_state, output_state_ref)
        for actual, expected in (
            (x.grad, x_ref.grad),
            (weight.grad, weight_ref.grad),
            (conv_weight.grad, conv_weight_ref.grad),
            (input_state.grad, state_ref.grad),
        ):
            self.assert_float_close(actual, expected)

    def test_varlen_state_and_empty_group(self):
        lengths = [0, 2, 17, 64, 65]
        boundaries = [0]
        for length in lengths:
            boundaries.append(boundaries[-1] + length)
        cu_seqlens = torch.tensor(
            boundaries, dtype=torch.int32, device=self.device)
        x, weight, conv_weight, _ = self.make_inputs(1, boundaries[-1])
        input_state = torch.randn(
            len(lengths), self.kernel_size - 1, self.out_channels,
            device=self.device, requires_grad=True)

        x_ref = x.detach().clone().requires_grad_()
        weight_ref = weight.detach().clone().requires_grad_()
        conv_weight_ref = conv_weight.detach().clone().requires_grad_()
        state_ref = input_state.detach().clone().requires_grad_()
        output, output_state = short_conv_silu(
            x, weight, conv_weight, cu_seqlens, max(lengths), input_state)
        output_ref, output_state_ref = reference_varlen_short_conv(
            x_ref, weight_ref, conv_weight_ref, cu_seqlens, state_ref)
        dout = torch.randn_like(output)
        doutput_state = torch.randn_like(output_state)
        torch.autograd.backward((output, output_state), (dout, doutput_state))
        torch.autograd.backward(
            (output_ref, output_state_ref), (dout, doutput_state))

        self.assert_float_close(output, output_ref)
        self.assert_float_close(output_state, output_state_ref)
        for actual, expected in (
            (x.grad, x_ref.grad),
            (weight.grad, weight_ref.grad),
            (conv_weight.grad, conv_weight_ref.grad),
            (input_state.grad, state_ref.grad),
        ):
            self.assert_float_close(actual, expected)

    def test_module_chunked_decoding_and_parameter_grads(self):
        module = CausalShortConv1d(
            self.in_channels, self.out_channels, self.kernel_size,
            device=self.device)
        x = torch.randn(
            2, 130, self.in_channels, device=self.device,
            requires_grad=True)
        x_ref = x.detach().clone().requires_grad_()
        weight_ref = module.weight.detach().clone().requires_grad_()
        conv_weight_ref = module.conv_weight.detach().clone().requires_grad_()
        state = torch.zeros(
            2, self.kernel_size - 1, self.out_channels,
            device=self.device)
        chunks = []
        offset = 0
        for size in (1, 2, 63, 64):
            output, state = module(
                x[:, offset:offset + size], input_state=state)
            chunks.append(output)
            offset += size
        decoded = torch.cat(chunks, dim=1)
        expected, expected_state = reference_short_conv(
            x_ref, weight_ref, conv_weight_ref)

        self.assert_float_close(decoded, expected)
        self.assert_float_close(state, expected_state)
        dout = torch.randn_like(decoded)
        dstate = torch.randn_like(state)
        torch.autograd.backward((decoded, state), (dout, dstate))
        torch.autograd.backward((expected, expected_state), (dout, dstate))
        self.assert_float_close(x.grad, x_ref.grad)
        self.assert_float_close(module.weight.grad, weight_ref.grad)
        self.assert_float_close(module.conv_weight.grad, conv_weight_ref.grad)
        self.assertEqual(
            repr(module),
            "CausalShortConv1d(in_channels=33, out_channels=47, kernel_size=4)")

    def test_bfloat16_state(self):
        x, weight, conv_weight, input_state = self.make_inputs(
            1, 65, dtype=torch.bfloat16, state=True)
        x_ref = x.detach().clone().requires_grad_()
        weight_ref = weight.detach().clone().requires_grad_()
        conv_weight_ref = conv_weight.detach().clone().requires_grad_()
        state_ref = input_state.detach().clone().requires_grad_()
        output, output_state = short_conv_silu(
            x, weight, conv_weight, input_state=input_state)
        output_ref, output_state_ref = reference_short_conv(
            x_ref, weight_ref, conv_weight_ref, state_ref)
        dout = torch.randn_like(output)
        doutput_state = torch.randn_like(output_state)
        torch.autograd.backward((output, output_state), (dout, doutput_state))
        torch.autograd.backward(
            (output_ref, output_state_ref), (dout, doutput_state))

        for actual, expected in (
            (output, output_ref),
            (output_state, output_state_ref),
            (x.grad, x_ref.grad),
            (weight.grad, weight_ref.grad),
            (conv_weight.grad, conv_weight_ref.grad),
            (input_state.grad, state_ref.grad),
        ):
            torch.testing.assert_close(actual, expected, rtol=0.04, atol=4.0)

    def test_bfloat16_autocast(self):
        module = CausalShortConv1d(
            self.in_channels, self.out_channels, self.kernel_size,
            device=self.device, dtype=torch.float32)
        x = torch.randn(
            2, 65, self.in_channels, device=self.device,
            dtype=torch.float32, requires_grad=True)
        input_state = torch.randn(
            2, self.kernel_size - 1, self.out_channels,
            device=self.device, dtype=torch.float32, requires_grad=True)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            output, output_state = module(x, input_state=input_state)
            loss = output.square().mean() + output_state.square().mean()
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertEqual(output_state.dtype, torch.bfloat16)
        loss.backward()

        self.assertEqual(x.grad.dtype, torch.float32)
        self.assertEqual(input_state.grad.dtype, torch.float32)
        self.assertEqual(module.weight.grad.dtype, torch.float32)
        self.assertEqual(module.conv_weight.grad.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
