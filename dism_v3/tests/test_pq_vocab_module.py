"""Training integration for 512 = 16 x 32 product-quantised codebooks."""

import pytest
import torch

from nanochat.models.dism import DismConfig
from nanochat.models.dism.model import DismLM


def build_config():
    return DismConfig(
        sequence_len=512,
        vocab_size=128,
        n_layer=2,
        n_head=1,
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
        pq_vocab_sizes=(16, 32),
    )


def test_pq_configuration_and_codebook_shape():
    from flash_dism.pq_vocab import codebook_shape, validate_sizes

    assert validate_sizes((16, 32), vocab_size=512, head_dim=64) == (16, 32)
    assert codebook_shape(6, (16, 32), 64) == (6, 2, 32, 32)
    with pytest.raises(ValueError, match='product'):
        validate_sizes((16, 16), vocab_size=512, head_dim=64)

    model = DismLM(build_config(), attention_type='hybrid_gdn')
    attn = model.layers[1].attn
    assert attn.pq_vocab_sizes == (16, 32)
    assert tuple(attn.q_vocab.shape) == (1, 2, 32, 32)
    assert tuple(attn.k_vocab.shape) == (1, 2, 32, 32)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_pq_compiled_forward_backward_and_metrics():
    torch.manual_seed(37)
    model = DismLM(build_config(), attention_type='hybrid_gdn').cuda().train()
    model.init_weights()
    compiled = torch.compile(model, dynamic=True)
    tokens = torch.randint(0, 128, (1, 512), device='cuda')
    cu_seqlens = torch.tensor([0, 256, 512], dtype=torch.int32, device='cuda')
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss = compiled(tokens, targets=tokens, cu_seqlens=cu_seqlens)
    assert torch.isfinite(loss)
    loss.backward()
    attn = model.layers[1].attn
    assert attn.q_vocab.grad is not None and torch.isfinite(attn.q_vocab.grad).all()
    assert attn.k_vocab.grad is not None and torch.isfinite(attn.k_vocab.grad).all()

    from nanochat.metrics import probe_qk_vocab_usage
    metrics = probe_qk_vocab_usage(model, tokens, cu_seqlens, 512, chunk_size=256)
    assert metrics[1]['q']['effective_symbols'] >= 0
    assert metrics[1]['k']['effective_symbols'] >= 0
