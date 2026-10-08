import unittest

import torch
from torch._dynamo.testing import CompileCounter

from random_bool import triton_rand_bool


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestRandomBool(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda")

    def test_arbitrary_shapes_and_dtypes(self):
        for shape in ((17,), (2, 3, 5), torch.Size([3, 7, 11])):
            for dtype in (torch.bool, torch.uint8):
                with self.subTest(shape=shape, dtype=dtype):
                    output = triton_rand_bool(
                        shape, 0.4, device=self.device,
                        seed=1234, dtype=dtype)
                    self.assertEqual(output.shape, torch.Size(shape))
                    self.assertEqual(output.dtype, dtype)
                    self.assertTrue(bool(((output == 0) | (output == 1)).all()))
                    self.assertEqual(output.element_size(), 1)

    def test_seed_is_reproducible(self):
        first = triton_rand_bool(
            (1025,), 0.37, device=self.device, seed=2026)
        second = triton_rand_bool(
            (1025,), 0.37, device=self.device, seed=2026)
        different = triton_rand_bool(
            (1025,), 0.37, device=self.device, seed=2027)
        torch.testing.assert_close(first, second)
        self.assertFalse(torch.equal(first, different))

        tensor_seed = torch.tensor(
            2026, device=self.device, dtype=torch.int64)
        torch.testing.assert_close(
            first,
            triton_rand_bool(
                (1025,), 0.37, device=self.device, seed=tensor_seed))

        torch.manual_seed(31415)
        from_global_rng = triton_rand_bool(
            (1025,), 0.37, device=self.device)
        torch.manual_seed(31415)
        torch.testing.assert_close(
            from_global_rng,
            triton_rand_bool((1025,), 0.37, device=self.device))

    def test_probability_and_empty_boundaries(self):
        self.assertEqual(
            torch.count_nonzero(triton_rand_bool(
                (257,), 0.0, device=self.device, seed=1)).item(), 0)
        self.assertEqual(
            torch.count_nonzero(triton_rand_bool(
                (257,), 1.0, device=self.device, seed=1)).item(), 257)
        empty = triton_rand_bool(
            (2, 0, 3), 0.5, device=self.device, seed=1)
        self.assertEqual(empty.shape, (2, 0, 3))
        self.assertEqual(empty.numel(), 0)

    def test_empirical_probability(self):
        output = triton_rand_bool(
            (1_000_000,), 0.3, device=self.device, seed=99)
        self.assertLess(abs(output.float().mean().item() - 0.3), 0.003)

    def test_fullgraph_capture_with_tensor_seed(self):
        seed = torch.tensor(7, device=self.device, dtype=torch.int64)
        counter = CompileCounter()
        compiled = torch.compile(
            lambda s: triton_rand_bool(
                (33, 65), 0.2, device=self.device, seed=s),
            backend=counter, fullgraph=True)
        first = compiled(seed)
        second = compiled(seed + 1)
        self.assertEqual(counter.frame_count, 1)
        self.assertFalse(torch.equal(first, second))

    def test_validation(self):
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            triton_rand_bool((4,), 1.1, device=self.device)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            triton_rand_bool((2, -1), 0.5, device=self.device)
        with self.assertRaisesRegex(TypeError, "dtype"):
            triton_rand_bool(
                (4,), 0.5, device=self.device, dtype=torch.float32)


if __name__ == "__main__":
    unittest.main()
