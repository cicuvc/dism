"""Experimental incremental hard (SAM) inference for the GDN-hybrid DISM LM.

Prefill runs parallel CPU SAM planning plus Triton execution and primes the
native hard cache; decoding then advances the same cache one token at a time
while a FLA GDN recurrent state carries the gated-delta-rule half. Projections,
short convolutions, gate/RMSNorm, MLP and the post-norm DismBlock structure are
the model's own math -- only the DISM core and the GDN recurrence are driven
incrementally here.

This is a diagnostic harness, NOT the supported model path: nanochat marks the
GDN hybrid generation=False and upstream does not implement cached hybrid
decoding. It is not wired into model dispatch or `DismAttention.forward`.

Requires `flash_dism` and `fla` importable (repo convention:
``PYTHONPATH=dism_v3/python:nanochat``). Paths are never hardcoded: the caller
passes a checkpoint directory and, optionally, a tokenizer directory.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.nn import functional as F

from flash_dism.decoding import _linear
from flash_dism.inference import HardDismDecoder
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule

from .models import build_model, ModelSpec


def find_checkpoint_step(checkpoint, step=None):
    """Highest ``meta_*.json`` step in a checkpoint directory, or the given one."""
    checkpoint = Path(checkpoint)
    steps = [int(p.stem.split("_")[1]) for p in sorted(checkpoint.glob("meta_*.json"))]
    if not steps:
        raise FileNotFoundError(f"no meta_*.json under {checkpoint}")
    step = max(steps) if step is None else int(step)
    if step not in steps:
        raise FileNotFoundError(f"step {step} has no meta_*.json under {checkpoint}")
    if not (checkpoint / f"model_{step:06d}.pt").is_file():
        raise FileNotFoundError(f"missing model_{step:06d}.pt under {checkpoint}")
    return step


def load_hard_model(checkpoint, *, step=None, device="cuda"):
    """Build a DISM model from ``checkpoint`` and load its weights.

    Returns ``(model, meta, step)``. The model is put on ``device`` in eval mode.
    """
    checkpoint = Path(checkpoint)
    step = find_checkpoint_step(checkpoint, step)
    meta = json.loads((checkpoint / f"meta_{step:06d}.json").read_text())
    with torch.device("meta"):
        model = build_model(ModelSpec.from_dict(meta["model_spec"]))
    model.to_empty(device="cpu")
    state = torch.load(checkpoint / f"model_{step:06d}.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), meta, step


def default_tokenizer_dir(checkpoint):
    """Nearest sibling ``tokenizer/`` of the run that owns ``checkpoint``."""
    checkpoint = Path(checkpoint)
    for candidate in (checkpoint / "tokenizer", checkpoint.parent / "tokenizer",
                      checkpoint.parent.parent / "tokenizer"):
        if candidate.is_dir():
            return candidate
    return checkpoint.parent / "tokenizer"


def _conv(x, layer, state):
    """Cached causal short conv in BF16, matching the production kernel dtype."""
    dtype = torch.bfloat16
    xb = x.to(dtype)
    projected = F.linear(xb, layer.weight.to(dtype))
    history_size = layer.kernel_size - 1
    if state is None:
        state = projected.new_zeros(xb.shape[0], history_size, layer.out_channels)
    if (state.shape != (xb.shape[0], history_size, layer.out_channels)
            or state.device != xb.device or state.dtype != dtype):
        raise ValueError("convolution cache shape/device/dtype changed")
    history = torch.cat((state, projected), dim=1)
    weight = layer.conv_weight.to(dtype)
    values = sum(history[:, history_size - lag:history_size - lag + xb.shape[1]] * weight[lag]
                 for lag in range(layer.kernel_size))
    updated = history[:, -history_size:].clone() if history_size else history[:, :0].clone()
    return F.silu(values), updated


class HardInferenceEngine:
    """Post-norm DismBlock with a SAM DISM cache and a FLA GDN recurrent state.

    Single sequence, single device. Call once with the full prompt to prefill,
    then once per new token. ``reset()`` clears all per-layer state so the
    engine can be reused for another sequence.
    """

    def __init__(self, model, prefill_engine, *, capacity, planner_backend="cpu"):
        self.model = model
        self.prefill_engine = prefill_engine
        self.capacity = int(capacity)
        if planner_backend not in ("cpu", "gpu"):
            raise ValueError("planner_backend must be cpu or gpu")
        self.planner = planner_backend
        self.dis = [None] * len(model.layers)
        self.gdn = [None] * len(model.layers)

    def reset(self):
        self.dis = [None] * len(self.model.layers)
        self.gdn = [None] * len(self.model.layers)

    def _gdn(self, module, x, index):
        state = self.gdn[index]
        conv = (None, None, None) if state is None else state["conv"]
        q, cq = _conv(x, module.gdn_q_conv, conv[0])
        k, ck = _conv(x, module.gdn_k_conv, conv[1])
        v, cv = _conv(x, module.gdn_v_conv, conv[2])
        batch, length, _ = x.shape
        q = q.reshape(batch, length, module.heads, module.head_dim)
        k = k.reshape_as(q)
        v = v.reshape(batch, length, module.heads, module.value_dim)
        beta = module.gdn_b_proj(x).sigmoid()
        g = -module.gdn_A_log.float().exp() * F.softplus(
            module.gdn_a_proj(x).float() + module.gdn_dt_bias.float())
        if state is None:
            out, final = chunk_gated_delta_rule(q=q, k=k, v=v, g=g, beta=beta, initial_state=None,
                                                output_final_state=True, use_qk_l2norm_in_kernel=True)
        else:
            out, final = fused_recurrent_gated_delta_rule(
                q=q, k=k, v=v, g=g, beta=beta, initial_state=state["state"],
                output_final_state=True, use_qk_l2norm_in_kernel=True)
        self.gdn[index] = dict(state=final, conv=(cq, ck, cv))
        return out

    def _dism(self, module, x, index):
        state = self.dis[index]
        conv = (None, None, None) if state is None else state["conv"]
        q, cq = _conv(x, module.q_conv, conv[0])
        k, ck = _conv(x, module.k_conv, conv[1])
        v, cv = _conv(x, module.v_conv, conv[2])
        batch, length, _ = x.shape
        q = q.reshape(batch, length, module.heads, module.head_dim)
        k = k.reshape_as(q)
        v = v.reshape(batch, length, module.heads, module.value_dim)
        sq = module._activate_readout(_linear(x, module.sq_proj))
        sk = module._activate_readout(_linear(x, module.sk_proj), is_key=True)
        iq = torch.einsum("bnhd,hvd->bnhv", q, module.q_vocab).argmax(-1).transpose(1, 2).int().contiguous()
        ik = torch.einsum("bnhd,hvd->bnhv", k, module.k_vocab).argmax(-1).transpose(1, 2).int().contiguous()
        tau = F.softplus(module.log_sel_tau)
        if state is None:
            cache = HardDismDecoder(batch, module.heads, sq.shape[-1], v.shape[-1], self.capacity, tau,
                                    cache_dtype=torch.bfloat16, planner_backend=self.planner)
            raw = self.prefill_engine(iq, ik, sq, sk, v, tau)
            cache.prime(iq, ik, sk, v)
            self.dis[index] = dict(cache=cache, conv=(cq, ck, cv))
        else:
            raw = state["cache"].append(iq, ik, sq, sk, v)
            self.dis[index] = dict(cache=state["cache"], conv=(cq, ck, cv))
        return raw

    def _branch(self, module, x, index):
        if getattr(module, "is_pure_gdn", False):
            raw = self._gdn(module, x, index)
        else:
            raw = self._dism(module, x, index)
            if hasattr(module, "gdn_raw"):
                raw = raw + self._gdn(module, x, index)
        gate = module.g_proj_up(module.g_proj_down(x)).reshape_as(raw)
        out = raw * torch.rsqrt(raw.square().mean(-1, keepdim=True) + module.norm.eps)
        out = out * module.norm.weight * F.silu(gate)
        return _linear(out.flatten(2), module.o_proj)

    @torch.inference_mode()
    def __call__(self, tokens):
        """tokens: int64 [1, T]. Returns softcapped logits [1, T, vocab]."""
        with torch.autocast("cuda", dtype=torch.bfloat16):
            x = self.model.embedding(tokens)
            for index, layer in enumerate(self.model.layers):
                if not layer.post_norm:
                    raise NotImplementedError("hard inference implements post-norm DismBlock only")
                branch = self._branch(layer.attn, x, index)
                x = layer.attn_norm(x + branch)
                x = layer.mlp_norm(x + layer.mlp(x))
            logits = self.model.lm_head(self.model.norm(x)).float()
        cap = self.model.config.softcap
        return cap * torch.tanh(logits / cap)


def sample_next_token(logits, *, temperature=1.0, top_k=0, top_p=1.0, repetition_penalty=1.0,
                      seen=(), generator=None):
    """Sample one token id from ``logits`` of shape ``[..., vocab]``."""
    logits = logits.clone()
    if repetition_penalty and repetition_penalty != 1.0 and seen:
        index = torch.tensor(sorted(seen), device=logits.device, dtype=torch.long)
        values = logits[..., index]
        logits[..., index] = torch.where(values > 0, values / repetition_penalty,
                                         values * repetition_penalty)
    if temperature <= 0:
        return logits.argmax(-1, keepdim=True)
    logits = logits / temperature
    if top_k and top_k > 0:
        threshold = torch.topk(logits, min(top_k, logits.shape[-1]), dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < threshold, float("-inf"))
    if top_p and top_p < 1.0:
        values, index = logits.sort(dim=-1, descending=True)
        probability = values.softmax(-1)
        remove = probability.cumsum(-1) - probability > top_p
        values = values.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, index, values)
    return torch.multinomial(logits.softmax(-1), 1, generator=generator)


@torch.inference_mode()
def decode(engine, logits, *, max_new_tokens, temperature=1.0, top_k=0, top_p=1.0,
           repetition_penalty=1.0, eos_token_id=None, generator=None, seen=(), on_token=None):
    """Decode from an already-prefilled engine and its first-position logits.

    ``logits`` is the ``[..., vocab]`` output for the last prompt position and
    ``engine`` must already hold the prompt state. Returns generated token ids.
    """
    generated, seen = [], set(seen)
    for _ in range(int(max_new_tokens)):
        token = sample_next_token(logits, temperature=temperature, top_k=top_k, top_p=top_p,
                                  repetition_penalty=repetition_penalty, seen=seen, generator=generator)
        token_id = int(token)
        if eos_token_id is not None and token_id == eos_token_id:
            break
        generated.append(token_id)
        seen.add(token_id)
        if on_token is not None:
            on_token(token_id)
        logits = engine(token)[:, -1]
    return generated


@torch.inference_mode()
def generate(engine, prompt_tokens, *, max_new_tokens, temperature=1.0, top_k=0, top_p=1.0,
             repetition_penalty=1.0, eos_token_id=None, generator=None, on_token=None):
    """Prefill ``prompt_tokens`` [1, N] then decode token by token.

    ``engine`` must be fresh (or reset). Returns the list of generated token ids.
    ``on_token(token_id)`` is called after each accepted token, for streaming.
    """
    logits = engine(prompt_tokens)[:, -1]
    seen = {int(t) for t in prompt_tokens.flatten().tolist()}
    return decode(engine, logits, max_new_tokens=max_new_tokens, temperature=temperature,
                  top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty,
                  eos_token_id=eos_token_id, generator=generator, seen=seen, on_token=on_token)


__all__ = ["find_checkpoint_step", "load_hard_model", "default_tokenizer_dir",
           "HardInferenceEngine", "sample_next_token", "decode", "generate"]
