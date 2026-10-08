import torch
import pytest

import nanochat.flash_attention as fa_module
import nanochat.models.llama2.model as llama_module
from nanochat.dataloader import build_varlen_metadata
from nanochat.models import ModelSpec, PretrainOptimizerConfig, build_model, optimizer_schema


def llama_spec(**overrides):
    config = {
        "sequence_len": 32,
        "vocab_size": 64,
        "n_layer": 2,
        "n_head": 4,
        "n_kv_head": 2,
        "n_embd": 32,
        "intermediate_size": 64,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "initializer_range": 0.02,
    }
    config.update(overrides)
    return ModelSpec("llama2", 1, config)


def test_llama2_registry_and_optimizer_cover_all_parameters():
    model = build_model(llama_spec())
    model.init_weights()
    optimizer = model.setup_pretraining_optimizer(PretrainOptimizerConfig(
        unembedding_lr=1e-3,
        embedding_lr=1e-3,
        matrix_lr=1e-3,
        scalar_lr=1e-3,
        weight_decay=0.1,
    ))

    assert model.capabilities.varlen
    assert not model.capabilities.generation
    schema = optimizer_schema(model, optimizer)
    assert sum(len(group["parameters"]) for group in schema["groups"]) == len(list(model.parameters()))


def test_llama2_matches_transformers_forward():
    from transformers import LlamaConfig, LlamaForCausalLM

    spec = llama_spec()
    hf_config = LlamaConfig(
        vocab_size=spec.config["vocab_size"],
        hidden_size=spec.config["n_embd"],
        intermediate_size=spec.config["intermediate_size"],
        num_hidden_layers=spec.config["n_layer"],
        num_attention_heads=spec.config["n_head"],
        num_key_value_heads=spec.config["n_kv_head"],
        max_position_embeddings=spec.config["sequence_len"],
        rms_norm_eps=spec.config["rms_norm_eps"],
        rope_theta=spec.config["rope_theta"],
        attention_bias=False,
        mlp_bias=False,
        attention_dropout=0.0,
        tie_word_embeddings=False,
    )
    torch.manual_seed(7)
    reference = LlamaForCausalLM(hf_config).eval().float()
    model = build_model(spec).eval().float()
    model.load_state_dict(reference.state_dict(), strict=True)
    input_ids = torch.randint(0, spec.config["vocab_size"], (2, 17))

    old_override = fa_module._override_impl
    old_dtype = llama_module.COMPUTE_DTYPE
    try:
        fa_module._override_impl = "sdpa"
        fa_module._refresh_impl_flags()
        llama_module.COMPUTE_DTYPE = torch.float32
        with torch.no_grad():
            expected = reference(input_ids).logits
            actual = model(input_ids)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    finally:
        llama_module.COMPUTE_DTYPE = old_dtype
        fa_module._override_impl = old_override
        fa_module._refresh_impl_flags()


def test_llama2_varlen_blocks_cross_document_attention():
    model = build_model(llama_spec(sequence_len=16, n_layer=1)).eval()
    model.init_weights()
    input_ids = torch.randint(0, 64, (1, 16))
    changed = input_ids.clone()
    changed[:, 8:] = torch.randint(0, 64, (1, 8))
    segment_ids = torch.tensor([[0] * 8 + [1] * 8], dtype=torch.int32)
    cu_seqlens, _ = build_varlen_metadata(segment_ids, max_segments=4)

    old_override = fa_module._override_impl
    try:
        fa_module._override_impl = "sdpa"
        fa_module._refresh_impl_flags()
        with torch.no_grad():
            original_logits = model(input_ids, cu_seqlens=cu_seqlens, segment_ids=segment_ids)
            changed_logits = model(changed, cu_seqlens=cu_seqlens, segment_ids=segment_ids)
        torch.testing.assert_close(original_logits[:, :8], changed_logits[:, :8], rtol=0, atol=0)
    finally:
        fa_module._override_impl = old_override
        fa_module._refresh_impl_flags()


@pytest.mark.skipif(not fa_module.HAS_FA2 or not torch.cuda.is_available(), reason="CUDA FA2 required")
def test_compiled_llama2_fa2_does_not_recompile_for_document_count():
    model = build_model(llama_spec(sequence_len=32, n_layer=1)).cuda()
    model.init_weights()
    compile_count = 0

    def counting_backend(graph_module, example_inputs):
        nonlocal compile_count
        compile_count += 1
        return graph_module.forward

    old_override = fa_module._override_impl
    try:
        fa_module._override_impl = "fa2"
        fa_module._refresh_impl_flags()
        torch._dynamo.reset()
        compiled = torch.compile(model, backend=counting_backend, dynamic=False, fullgraph=True)
        inputs = torch.randint(0, 64, (2, 32), device="cuda")
        targets = torch.randint(0, 64, (2, 32), device="cuda")
        for segment_len in (16, 8):
            segment_ids = (
                torch.arange(64, device="cuda") // segment_len
            ).view(2, 32).to(torch.int32)
            cu_seqlens, _ = build_varlen_metadata(segment_ids, max_segments=16)
            loss = compiled(
                inputs, targets,
                cu_seqlens=cu_seqlens, segment_ids=segment_ids,
            )
            loss.backward()
            model.zero_grad(set_to_none=True)
        assert compile_count == 1
    finally:
        torch._dynamo.reset()
        fa_module._override_impl = old_override
        fa_module._refresh_impl_flags()
