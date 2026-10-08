import unittest

import torch
from torch._dynamo.testing import CompileCounter

from conv1d import short_conv_silu
from dynamo_utils import mark_cu_seqlens_dynamic
from fused_cross_entropy import fused_cross_entropy
from linear_rmsnorm_rope import linear_rmsnorm_rope


def make_rope(sequence, head_dim, device):
    position = torch.arange(sequence, device=device, dtype=torch.float32)
    frequency = torch.arange(
        head_dim // 2, device=device, dtype=torch.float32)
    angle = position[:, None] * frequency[None, :] * 0.01
    return angle.cos().bfloat16(), angle.sin().bfloat16()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestDynamoCompatibility(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2030)
        self.device = torch.device("cuda")

    def compile_and_check(self, function, calls):
        counter = CompileCounter()
        compiled = torch.compile(
            function, backend=counter, fullgraph=True, dynamic=True)
        for args in calls:
            compiled(*args)
        self.assertEqual(counter.frame_count, 1)

    def test_fixed_fullgraph_capture(self):
        x = torch.randn(
            2, 17, 33, device=self.device, dtype=torch.bfloat16)
        conv_weight = torch.randn(
            47, 33, device=self.device, dtype=torch.bfloat16)
        kernel = torch.randn(
            4, 47, device=self.device, dtype=torch.bfloat16)
        self.compile_and_check(
            lambda a, b, c: short_conv_silu(a, b, c),
            [(x, conv_weight, kernel)])

        rope_weight = torch.randn(
            192, 33, device=self.device, dtype=torch.bfloat16)
        rms_weight = torch.randn(
            3, 64, device=self.device, dtype=torch.bfloat16)
        cos, sin = make_rope(17, 64, self.device)
        self.compile_and_check(
            lambda a, b, c, d, e: linear_rmsnorm_rope(a, b, c, d, e),
            [(x, rope_weight, rms_weight, cos, sin)])

        labels = torch.randint(67, (2, 17), device=self.device)
        labels[:, ::5] = -100
        ce_weight = torch.randn(
            67, 33, device=self.device, dtype=torch.bfloat16)
        self.compile_and_check(
            lambda a, b, c: fused_cross_entropy(
                a, b, c, softcap=2.0, chunk_size=5),
            [(x, ce_weight, labels)])

    def test_varlen_values_and_shapes_do_not_recompile(self):
        total_tokens = 17
        max_seqlen = 10
        boundaries = (
            [0, 2, 7, 17],
            [0, 1, 9, 17],
            [0, 1, 4, 8, 17],
            [0, 1, 3, 6, 10, 17],
        )
        cu_seqlens = [torch.tensor(
            values, device=self.device, dtype=torch.int32)
            for values in boundaries]
        # The first length deliberately equals the conv kernel size. This
        # catches Dynamo duck-shape specialization unless explicitly marked.
        mark_cu_seqlens_dynamic(cu_seqlens[0])

        x = torch.randn(
            1, total_tokens, 33,
            device=self.device, dtype=torch.bfloat16)
        conv_weight = torch.randn(
            47, 33, device=self.device, dtype=torch.bfloat16)
        kernel = torch.randn(
            4, 47, device=self.device, dtype=torch.bfloat16)
        self.compile_and_check(
            lambda a, b, c, q: short_conv_silu(
                a, b, c, q, max_seqlen),
            [(x, conv_weight, kernel, q) for q in cu_seqlens])

        rope_weight = torch.randn(
            192, 33, device=self.device, dtype=torch.bfloat16)
        rms_weight = torch.randn(
            3, 64, device=self.device, dtype=torch.bfloat16)
        cos, sin = make_rope(max_seqlen, 64, self.device)
        self.compile_and_check(
            lambda a, b, c, d, e, q: linear_rmsnorm_rope(
                a, b, c, d, e,
                cu_seqlens=q, max_seqlen=max_seqlen),
            [(x, rope_weight, rms_weight, cos, sin, q)
             for q in cu_seqlens])

        labels = torch.randint(
            67, (1, total_tokens), device=self.device)
        labels[:, ::5] = -100
        ce_weight = torch.randn(
            67, 33, device=self.device, dtype=torch.bfloat16)
        self.compile_and_check(
            lambda a, b, c, q: fused_cross_entropy(
                a, b, c, softcap=2.0, chunk_size=5,
                cu_seqlens=q, max_seqlen=max_seqlen),
            [(x, ce_weight, labels, q) for q in cu_seqlens])


if __name__ == "__main__":
    unittest.main()
