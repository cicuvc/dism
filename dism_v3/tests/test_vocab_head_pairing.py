"""Compact head-paired Q/K codebooks with role swapping in the second half."""

import pytest
import torch

from nanochat.models.dism import DismConfig
from nanochat.models.dism.model import DismLM


def attention():
    from flash_dism.hybrid_gdn import DismGdnAttention
    return DismGdnAttention(
        64, 4, head_dim=32, value_dim=32, readout_dim=16, vocab_size=32,
        vocab_head_pairing='swap_halves')


def test_swap_halves_shapes_and_equivalent_expanded_roles():
    from flash_dism.hybrid_gdn import DismGdnAttention
    from flash_dism.vocab_roles import expand_qk_vocab

    paired = attention()
    assert tuple(paired.q_vocab.shape) == (2, 32, 32)
    assert tuple(paired.k_vocab.shape) == (2, 32, 32)

    full = DismGdnAttention(
        64, 4, head_dim=32, value_dim=32, readout_dim=16, vocab_size=32)
    state = paired.state_dict()
    state['q_vocab'], state['k_vocab'] = expand_qk_vocab(
        paired.q_vocab, paired.k_vocab, 'swap_halves', heads=4, kv_heads=4)
    state['q_vocab'] = state['q_vocab'].detach().clone()
    state['k_vocab'] = state['k_vocab'].detach().clone()
    full.load_state_dict(state, strict=True)

    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    paired, full = paired.cuda().eval(), full.cuda().eval()
    x = torch.randn(1, 256, 64, device='cuda')
    cu = torch.tensor([0, 256], dtype=torch.int32, device='cuda')
    hard = torch.ones(1, 4, dtype=torch.bool, device='cuda')
    direction = torch.ones(1, 4, dtype=torch.bool, device='cuda')
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        a, _, _ = paired(
            x, cu_seqlens=cu, max_seqlen=256, hard=hard,
            direction=direction, direction_true=True)
        b, _, _ = full(
            x, cu_seqlens=cu, max_seqlen=256, hard=hard,
            direction=direction, direction_true=True)
    torch.testing.assert_close(a, b, atol=0, rtol=0)


def build_config():
    return DismConfig(
        sequence_len=512,
        vocab_size=128,
        n_layer=2,
        n_head=2,
        n_embd=64,
        intermediate_size=128,
        head_dim=64,
        value_dim=64,
        readout_dim=16,
        qk_vocab_size=512,
        window_size=128,
        anneal_steps=10,
        post_norm=True,
        rear_half_dism=True,
        direction_policy='true',
        value_residual=True,
        vocab_head_pairing='swap_halves',
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_swap_halves_compiled_model_forward_backward_and_metrics():
    torch.manual_seed(41)
    model = DismLM(build_config(), attention_type='hybrid_gdn').cuda().train()
    model.init_weights()
    attn = model.layers[1].attn
    assert tuple(attn.q_vocab.shape) == (1, 512, 64)
    assert tuple(attn.k_vocab.shape) == (1, 512, 64)
    compiled = torch.compile(model, dynamic=True)
    tokens = torch.randint(0, 128, (1, 512), device='cuda')
    cu = torch.tensor([0, 256, 512], dtype=torch.int32, device='cuda')
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss = compiled(tokens, targets=tokens, cu_seqlens=cu)
    loss.backward()
    assert attn.q_vocab.grad is not None and torch.isfinite(attn.q_vocab.grad).all()
    assert attn.k_vocab.grad is not None and torch.isfinite(attn.k_vocab.grad).all()

    from nanochat.metrics import probe_qk_vocab_usage
    metrics = probe_qk_vocab_usage(model, tokens, cu, 512, chunk_size=256)
    assert metrics[1]['q']['effective_symbols'] >= 0
