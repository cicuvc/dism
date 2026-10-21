import importlib

import pytest
import torch

from nanochat.models import ModelSpec, build_model


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_compiled_gdn_metadata_shared_per_forward_not_across_mutation(monkeypatch):
    from torch.fx.experimental import _config as shape_config
    chunk = importlib.import_module('fla.ops.gated_delta_rule.chunk')
    original = chunk.prepare_chunk_indices
    calls = []
    fresh = False

    def observed(cu, *args, **kwargs):
        indices = original(cu.clone() if fresh else cu, *args, **kwargs)
        calls.append((cu, indices))
        return indices

    monkeypatch.setattr(chunk, 'prepare_chunk_indices', observed)
    torch.manual_seed(173)
    config = dict(sequence_len=512, vocab_size=128, n_layer=4, n_head=2,
                  n_embd=64, intermediate_size=128, head_dim=32, value_dim=32,
                  readout_dim=16, qk_vocab_size=32,
                  value_residual=True)
    model = build_model(ModelSpec('dism_gdn_hybrid', 1, config)).cuda().train()
    model.init_weights()
    ids = torch.randint(128, (2, 512), device='cuda')
    cu = torch.tensor([0, 256, 512, 1024], dtype=torch.int32, device='cuda')
    previous = None
    with shape_config.patch(use_duck_shape=False), \
            torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True), \
            torch._inductor.config.patch(compile_threads=1):
        compiled = torch.compile(model, dynamic=True)
        for bounds in ([0, 256, 512, 1024], [0, 512, 768, 1024]):
            cu.copy_(torch.tensor(bounds, dtype=cu.dtype, device=cu.device))
            calls.clear()
            fresh = False
            counter = model.rng_counter.clone()
            model.hard_probability.zero_()
            actual = compiled(ids, cu_seqlens=cu)
            assert len(calls) == 4
            shared = calls[0][0]
            assert shared is not cu and shared is not previous
            assert all(boundary is shared for boundary, _ in calls)
            assert all(index is calls[0][1] for _, index in calls)
            torch.testing.assert_close(shared, cu, rtol=0, atol=0)
            previous = shared
            actual.float().square().mean().backward()
            grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
            model.zero_grad(set_to_none=True)
            # Force the old cache-miss behavior without changing kernel math.
            fresh = True
            model.rng_counter.copy_(counter)
            model.hard_probability.zero_()
            expected = compiled(ids, cu_seqlens=cu)
            expected.float().square().mean().backward()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for n, p in model.named_parameters():
                if n in grads:
                    torch.testing.assert_close(p.grad, grads[n], rtol=.03, atol=.001)
            model.zero_grad(set_to_none=True)
