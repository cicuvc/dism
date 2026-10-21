"""MXFP8 hybrid training integration on SM120."""
import json
import copy
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from nanochat.models import ModelSpec, PretrainOptimizerConfig, build_model
from flash_mxfp8 import FusedAdamWMXFP8
from flash_mxfp8.build import build_all
from flash_mxfp8.tk_mxfp8_down import MXFP8DownProjection
from flash_mxfp8.tk_mxfp8_linear import MXFP8Linear
from flash_mxfp8.tk_mxfp8_swiglu_layer import MXFP8SwiGLU
from nanochat.models.dism.conv import LinearShortConvSiLU, MXFP8LinearShortConvSiLU


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="SM120 required",
)


@pytest.fixture(scope="module", autouse=True)
def compiled_mxfp8():
    build_all()


def tiny_spec():
    return ModelSpec(
        "dism_gdn_hybrid",
        1,
        dict(
            sequence_len=256,
            vocab_size=128,
            n_layer=2,
            n_head=4,
            n_embd=128,
            intermediate_size=256,
            head_dim=32,
            value_dim=32,
            readout_dim=16,
            qk_vocab_size=32,
            anneal_steps=100,
            hard_prob_max=.5,
            mxfp8=True,
            mxfp8_projections=True,
            mxfp8_optimizer=True,
        ),
    )


def test_mxfp8_layer_scope():
    torch.manual_seed(7)
    model = build_model(tiny_spec()).cuda().train()
    model.init_weights()
    assert model.embedding.weight.dtype == torch.bfloat16
    assert model.lm_head.weight.dtype == torch.bfloat16
    assert isinstance(model.layers[0].attn.gdn_q_conv, MXFP8LinearShortConvSiLU)
    assert isinstance(model.layers[1].attn.q_conv, MXFP8LinearShortConvSiLU)
    assert isinstance(model.layers[1].attn.sq_proj, MXFP8Linear)
    assert isinstance(model.layers[1].attn.sk_proj, MXFP8Linear)
    assert isinstance(model.layers[1].mlp.mxfp8_mlp, MXFP8SwiGLU)
    assert isinstance(model.layers[1].attn.o_proj, nn.Linear)
    assert model.layers[1].attn.o_proj.weight.dtype == torch.bfloat16
    assert not any("_weight_fp8" in name or "_weight_scale" in name
                   for name in model.state_dict())


def test_mxfp8_forward_backward_and_fused_optimizer():
    torch.manual_seed(8)
    model = build_model(tiny_spec()).cuda().train()
    model.init_weights()
    model.set_training_step(50)
    ids = torch.randint(0, 128, (1, 256), device="cuda")
    targets = torch.randint(0, 128, (1, 256), device="cuda")
    cu = torch.tensor([0, 256], device="cuda", dtype=torch.int32)
    loss = model(ids, targets, cu_seqlens=cu)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    optimizer = model.setup_pretraining_optimizer(
        PretrainOptimizerConfig(.001, .001, .001, .001, .01)
    )
    assert isinstance(optimizer, FusedAdamWMXFP8)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        logits = model(ids, cu_seqlens=cu)
    assert torch.isfinite(logits).all()
    assert any(
        state["exp_avg"].dtype == torch.bfloat16
        for state in optimizer.state.values()
        if state
    )


def test_standard_adamw_refreshes_fp8_snapshots():
    from flash_mxfp8.core import quantize_weight

    torch.manual_seed(10)
    model = build_model(tiny_spec()).cuda().train()
    model.init_weights()
    model.config.mxfp8_optimizer = False
    optimizer = model.setup_pretraining_optimizer(
        PretrainOptimizerConfig(.008, .008, .008, .008, .01)
    )
    assert isinstance(optimizer, torch.optim.AdamW)
    assert not isinstance(optimizer, FusedAdamWMXFP8)

    for step in range(2):
        torch.manual_seed(100 + step)
        for parameter in model.parameters():
            parameter.grad = torch.randn_like(parameter) * 1e-3
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        for name, module in model.named_modules():
            bindings = getattr(module, "fp8_parameter_bindings", None)
            if not callable(bindings):
                continue
            for parameter, weight_fp8, weight_scale in bindings():
                fresh_fp8, fresh_scale = quantize_weight(parameter.detach())
                assert torch.equal(weight_fp8, fresh_fp8), name
                assert torch.equal(weight_scale, fresh_scale), name


def test_mxfp8_quantizer_large_m_matches_bf16():
    def cosine(actual, expected):
        actual = actual.detach().float().flatten()
        expected = expected.detach().float().flatten()
        return float(
            torch.dot(actual, expected)
            / (actual.norm() * expected.norm()).clamp_min(1e-30)
        )

    # The direct quantizer feeds both the down projection and weight-gradient
    # input. This M requires more than one persistent tile pass per worker.
    torch.manual_seed(31)
    down = MXFP8DownProjection(
        3968, 512, device="cuda", raster_group=13
    )
    activation = torch.randn(
        3072, 3968, device="cuda", dtype=torch.bfloat16
    )
    actual = down(activation)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        expected = F.linear(activation, down.weight)
    assert cosine(actual, expected) > 0.999

    # The same double-buffered quantizer emits dG/dU for the fused SwiGLU
    # backward. M=1024 already crosses into repeated worker tile passes.
    torch.manual_seed(32)
    swiglu = MXFP8SwiGLU(
        512,
        3968,
        512,
        device="cuda",
        forward_config=8,
        dx_config=12,
        wgrad_config=14,
        down_forward_config=67,
        down_dx_config=17,
        down_dw_config=12,
        raster_group=13,
    )
    reference = nn.Module()
    reference.gate_weight = nn.Parameter(swiglu.gate_weight.detach().clone())
    reference.up_weight = nn.Parameter(swiglu.up_weight.detach().clone())
    reference.down_weight = nn.Parameter(
        swiglu.down_proj.weight.detach().clone()
    )
    inputs = torch.randn(
        1024, 512, device="cuda", dtype=torch.bfloat16
    )
    grad_output = torch.randn(
        1024, 512, device="cuda", dtype=torch.bfloat16
    )
    mxfp8_input = inputs.clone().requires_grad_()
    reference_input = inputs.clone().requires_grad_()
    swiglu(mxfp8_input).backward(grad_output)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = F.silu(F.linear(reference_input, reference.gate_weight))
        hidden = hidden * F.linear(reference_input, reference.up_weight)
        reference_output = F.linear(hidden, reference.down_weight)
    reference_output.backward(grad_output)
    assert cosine(mxfp8_input.grad, reference_input.grad) > 0.99
    assert cosine(swiglu.gate_weight.grad, reference.gate_weight.grad) > 0.99


def test_selected_mxfp8_config_shape():
    path = Path(__file__).parents[1] / "configs/sequence_true_72m_valres_mxfp8.json"
    spec = ModelSpec.from_dict(json.loads(path.read_text()))
    with torch.device("meta"):
        model = build_model(spec)
    assert model.config.intermediate_size == 2560
    assert sum(p.numel() for p in model.parameters()) == 71_814_624
    assert model.embedding.weight.dtype == torch.bfloat16
    assert model.lm_head.weight.dtype == torch.bfloat16

    path = Path(__file__).parents[1] / "configs/sequence_true_125m_valres_mxfp8.json"
    spec = ModelSpec.from_dict(json.loads(path.read_text()))
    with torch.device("meta"):
        model = build_model(spec)
    assert model.config.n_embd == 512
    assert model.config.n_head == 8
    assert model.config.intermediate_size == 3968
    assert sum(p.numel() for p in model.parameters()) == 125_876_608


def test_mxfp8_checkpoint_roundtrip_rebuilds_caches():
    torch.manual_seed(9)
    spec = tiny_spec()
    model = build_model(spec).cuda().train()
    model.init_weights()
    optimizer = model.setup_pretraining_optimizer(
        PretrainOptimizerConfig(.001, .001, .001, .001, .01)
    )
    model_state = copy.deepcopy(model.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict())

    restored = build_model(spec).cuda().train()
    restored.init_weights()
    restored.load_state_dict(model_state, strict=True)
    restored_optimizer = restored.setup_pretraining_optimizer(
        PretrainOptimizerConfig(.001, .001, .001, .001, .01)
    )
    restored_optimizer.load_state_dict(optimizer_state)
    assert isinstance(restored_optimizer, FusedAdamWMXFP8)
    assert restored.layers[1].mlp.mxfp8_mlp._gate_weight_fp8.numel() > 0


def test_ffn_only_mxfp8_keeps_attention_projections_bf16():
    path = Path(__file__).parents[1] / (
        "configs/sequence_true_125m_valres_mxfp8_ffnonly.json"
    )
    spec = ModelSpec.from_dict(json.loads(path.read_text()))
    with torch.device("meta"):
        model = build_model(spec)
    assert type(model.layers[0].attn.gdn_q_conv) is LinearShortConvSiLU
    assert type(model.layers[6].attn.q_conv) is LinearShortConvSiLU
    assert type(model.layers[6].attn.sq_proj) is nn.Linear
    assert isinstance(model.layers[6].mlp.mxfp8_mlp, MXFP8SwiGLU)
    assert model.layers[6].attn.o_proj.weight.dtype == torch.bfloat16
