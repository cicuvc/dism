"""Sliding-window PaTH mixers share the SDN feature/output interfaces."""

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
        qk_vocab_size=32,
        window_size=128,
        anneal_steps=10,
        post_norm=True,
        rear_half_dism=True,
        direction_policy='true',
        value_residual=True,
        recurrent_mixer='sliding_path',
        sliding_delta_chunk_size=32,
    )


def test_layout_and_parameter_compatibility_with_sdn():
    from flash_dism.sliding_delta import (
        DismSlidingDeltaAttention,
        PureSlidingDeltaAttention,
    )
    from flash_dism.sliding_path import (
        DismSlidingPathAttention,
        PureSlidingPathAttention,
    )

    model = DismLM(build_config(), attention_type='hybrid_gdn')
    pure, hybrid = model.layers[0].attn, model.layers[1].attn
    assert isinstance(pure, PureSlidingPathAttention)
    assert isinstance(hybrid, DismSlidingPathAttention)
    assert pure.window_size == hybrid.window_size == 128
    assert pure.head_dim == pure.head_v_dim == 64
    assert hybrid.head_dim == hybrid.value_dim == 64
    assert hybrid.q_conv.weight is hybrid.path_q_conv.weight
    assert hybrid.k_conv.weight is hybrid.path_k_conv.weight

    sdn = PureSlidingDeltaAttention(
        64, 1, head_dim=64, value_dim=64, window_size=128,
        sliding_delta_chunk_size=32)
    path = PureSlidingPathAttention(
        64, 1, head_dim=64, value_dim=64, window_size=128,
        sliding_path_chunk_size=32)
    assert sdn.state_dict().keys() == path.state_dict().keys()
    path.load_state_dict(sdn.state_dict(), strict=True)

    sdn_hybrid = DismSlidingDeltaAttention(
        64, 1, head_dim=64, value_dim=64, readout_dim=16, vocab_size=32,
        window_size=128, sliding_delta_chunk_size=32)
    path_hybrid = DismSlidingPathAttention(
        64, 1, head_dim=64, value_dim=64, readout_dim=16, vocab_size=32,
        window_size=128, sliding_path_chunk_size=32)
    assert sum(p.numel() for p in sdn_hybrid.parameters()) == sum(
        p.numel() for p in path_hybrid.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_path_full_model_forward_backward_with_value_residual():
    torch.manual_seed(29)
    model = DismLM(build_config(), attention_type='hybrid_gdn').cuda().train()
    model.init_weights()
    tokens = torch.randint(0, 128, (1, 512), device='cuda')
    cu_seqlens = torch.tensor([0, 256, 512], dtype=torch.int32, device='cuda')
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss = model(tokens, targets=tokens, cu_seqlens=cu_seqlens)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), name
    assert model.layers[1].attn.path_v_residual_gate.weight.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_path_hybrid_packed_documents_do_not_leak():
    from flash_dism.sliding_path import DismSlidingPathAttention

    torch.manual_seed(31)
    module = DismSlidingPathAttention(
        64, 1, head_dim=64, value_dim=64, readout_dim=16, vocab_size=32,
        window_size=128, sliding_path_chunk_size=32).cuda().eval()
    cu_seqlens = torch.tensor([0, 256, 512], dtype=torch.int32, device='cuda')
    hard = torch.ones(1, 1, dtype=torch.bool, device='cuda')
    direction = torch.ones(1, 1, dtype=torch.bool, device='cuda')
    x = torch.randn(1, 512, 64, device='cuda')
    changed = x.clone()
    changed[:, :256] *= 100
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        original, _, _ = module(
            x, cu_seqlens=cu_seqlens, max_seqlen=256, hard=hard,
            direction=direction, direction_true=True)
        other, _, _ = module(
            changed, cu_seqlens=cu_seqlens, max_seqlen=256, hard=hard,
            direction=direction, direction_true=True)
    torch.testing.assert_close(original[:, 256:], other[:, 256:], atol=0, rtol=0)
