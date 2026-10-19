"""Every N DISM layers share same-head Q/K codebooks."""

import copy

import pytest
import torch

from nanochat.models.dism import DismConfig
from nanochat.models.dism.model import DismLM


def build_config(group_size=2):
    return DismConfig(
        sequence_len=512,
        vocab_size=128,
        n_layer=4,
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
        vocab_layer_group_size=group_size,
    )


def test_layer_pairs_alias_q_and_k_tables_independently():
    first = DismLM(build_config(), attention_type='hybrid_gdn')
    a, b = first.layers[2].attn, first.layers[3].attn
    assert a.q_vocab is b.q_vocab
    assert a.k_vocab is b.k_vocab
    assert a.q_vocab is not a.k_vocab

    restored = DismLM(build_config(), attention_type='hybrid_gdn')
    restored.load_state_dict(copy.deepcopy(first.state_dict()), strict=True)
    a, b = restored.layers[2].attn, restored.layers[3].attn
    assert a.q_vocab is b.q_vocab
    assert a.k_vocab is b.k_vocab


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_layer_pair_compiled_forward_backward_and_metrics():
    torch.manual_seed(43)
    model = DismLM(build_config(), attention_type='hybrid_gdn').cuda().train()
    model.init_weights()
    a, b = model.layers[2].attn, model.layers[3].attn
    compiled = torch.compile(model, dynamic=True)
    tokens = torch.randint(0, 128, (1, 512), device='cuda')
    cu = torch.tensor([0, 256, 512], dtype=torch.int32, device='cuda')
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss = compiled(tokens, targets=tokens, cu_seqlens=cu)
    loss.backward()
    assert a.q_vocab.grad is not None and torch.isfinite(a.q_vocab.grad).all()
    assert b.q_vocab is a.q_vocab
    assert a.k_vocab.grad is not None and torch.isfinite(a.k_vocab.grad).all()

    from nanochat.metrics import probe_qk_vocab_usage
    metrics = probe_qk_vocab_usage(model, tokens, cu, 512, chunk_size=256)
    assert metrics[2]['q']['effective_symbols'] >= 0
