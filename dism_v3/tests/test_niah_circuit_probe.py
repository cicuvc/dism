"""CPU checks for opt-in circuit diagnostics, independent of CUDA execution."""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from niah_circuit_probe import ablate
from niah_head_ablation import zero_slot
from niah_read_rows import hard_row


class CircuitProbeTests(unittest.TestCase):
    def test_head_ablation_masks_only_selected_premix_slot_and_layer(self):
        compiler = ModuleType("flash_dism.compiler")
        package = ModuleType("flash_dism")
        package.compiler = compiler
        values = torch.ones(1, 2, 12, 3)
        original = lambda: [values, "saved-state"]
        compiler.voc_forward = original
        model = SimpleNamespace(layers=[SimpleNamespace(attn=torch.nn.Identity()) for _ in range(12)])
        with patch.dict(sys.modules, {"flash_dism": package, "flash_dism.compiler": compiler}):
            with zero_slot(model, 8, 3) as calls:
                model.layers[7].attn(torch.tensor(0))
                self.assertIs(compiler.voc_forward()[0], values)
                model.layers[8].attn(torch.tensor(0))
                output = compiler.voc_forward()
                self.assertEqual(output[0][:, :, 3].sum().item(), 0)
                self.assertEqual(output[0].sum().item(), 66)
                self.assertEqual(output[1], "saved-state")
                self.assertEqual(values.sum().item(), 72)
                self.assertEqual(calls, [8])
            self.assertIs(compiler.voc_forward, original)
            self.assertTrue(all(not layer.attn._forward_pre_hooks for layer in model.layers))

    def test_ablation_preserves_gdn_kwargs_and_restores_after_exception(self):
        class Hybrid:
            gdn_raw = True

            def _combine_cuda(self, output, gdn, *, conv_seq_idx):
                self.last_seq_idx = conv_seq_idx
                return output + gdn

        attn = Hybrid()
        model = SimpleNamespace(layers=[SimpleNamespace(attn=attn)])
        with self.assertRaisesRegex(RuntimeError, "injected"):
            with ablate(model, 0) as count:
                result = attn._combine_cuda(torch.tensor([9.]), torch.tensor([2.]), conv_seq_idx=17)
                self.assertEqual(result.item(), 2.)
                self.assertEqual(count["calls"], 1)
                self.assertEqual(attn.last_seq_idx, 17)
                raise RuntimeError("injected")
        self.assertNotIn("_combine_cuda", attn.__dict__)
        self.assertEqual(attn._combine_cuda(torch.tensor([9.]), torch.tensor([2.]), conv_seq_idx=17).item(), 11.)

    def test_rows_against_full_fp64_oracle(self):
        # Import the authoritative reference without importing CUDA package init.
        spec = importlib.util.spec_from_file_location("oracle", ROOT / "python/flash_dism/reference/dism_v3_ref.py")
        oracle = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(oracle)
        gen = torch.Generator().manual_seed(742)
        n, h = 32, 3
        iq = torch.randint(0, 4, (1, h, n), generator=gen)
        ik = torch.randint(0, 4, (1, h, n), generator=gen)
        iq[:, 0] = 0
        ik[:, 0] = 0  # long unbroken suffix
        iq[:, 1] = 5  # no matches
        sq, sk, v = [torch.randn(1, n, h, 4, generator=gen, dtype=torch.float64) for _ in range(3)]
        q = torch.zeros_like(sq)
        lse = torch.zeros(1, n, h, dtype=torch.float64)
        flags = torch.ones(1, h, dtype=torch.bool)
        for tau_value in (0., 1e-9, 0.3, 30., -0.2):
            tau = torch.full((h,), tau_value, dtype=torch.float64)
            expected = oracle.dism_ref(q, q, sq, sk, lse, lse, iq, ik, flags,
                                      flags[..., None].expand(1, h, n), v, tau)[0]
            for head in range(h):
                for i in (0, 7, n-1):
                    lengths, mass, fallback = hard_row(iq[0, head].numpy(), ik[0, head].numpy(), i, tau_value)
                    signed = sk[0, :i+1, head].numpy() @ sq[0, i, head].numpy()
                    actual = ((mass*signed)[:, None] * v[0, :i+1, head].numpy()).sum(0)
                    np.testing.assert_allclose(actual, expected[i, head].numpy(), rtol=1e-12, atol=1e-12)
                    self.assertAlmostEqual(float(mass.sum()+fallback), 1.)
                    if head == 0:
                        np.testing.assert_array_equal(lengths, np.arange(1, i+2))
                    if head == 1:
                        self.assertEqual(fallback, 1.)


if __name__ == "__main__":
    unittest.main()
