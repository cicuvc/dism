"""Experimental hard-inference generation harness for the GDN-hybrid DISM LM.

Prefill (parallel SAM planning + Triton execution) primes the native hard cache,
then tokens are decoded one at a time with the SAM decoder plus a FLA GDN
recurrent state. This is a diagnostic for generation quality/feel, NOT a
supported production path: nanochat marks the GDN hybrid as generation=False and
cached hybrid decoding is not implemented upstream. Outer model math
(projections, short convs, gate/norm/MLP) is the model's own; only the DISM core
and the GDN recurrence are driven incrementally here.

Example:
    python -m scripts.hard_generate --temperature 0.8 --top-p 0.95 -n 200
    python -m scripts.hard_generate --greedy -n 100
    python -m scripts.hard_generate --preset science --prompt-file p.txt
"""
import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "dism_v3" / "python"), str(REPO / "nanochat")]

import torch
from torch.nn import functional as F

from nanochat.models import build_model, ModelSpec
from nanochat.tokenizer import HuggingFaceTokenizer
from flash_dism.decoding import _linear
from flash_dism.inference import HardDismPrefill, HardDismDecoder
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule

DEFAULT_CHECKPOINT = Path(
    "/media/cicuvc/c63abdf1-0e56-4153-9228-95df5a2f239b/cicuvc/dism-runs"
    "/dism-simlpajama-72m-2500m/base_checkpoints/seqtrue72m")

PRESETS = {
    "solar": ("The solar system consists of the Sun and the objects that orbit it. "
              "The largest planet is"),
    "science": ("The history of science is the study of the development of science, "
                "including both the natural and social sciences. Science is a body of "
                "empirical, theoretical, and practical knowledge about the natural world. "
                "The scientific method"),
    "story": ("Once upon a time, a young girl found a mysterious book in the library. "
              "When she opened it,"),
    "code": ("def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n    if n < 2:\n        return n\n    return"),
    "chat": ("User: What is the capital of France?\nAssistant:"),
}


def _conv(x, layer, state):
    """Cached causal short conv in BF16, matching the production kernel dtype."""
    dtype = torch.bfloat16
    xb = x.to(dtype)
    projected = F.linear(xb, layer.weight.to(dtype))
    hs = layer.kernel_size - 1
    if state is None:
        state = projected.new_zeros(xb.shape[0], hs, layer.out_channels)
    if (state.shape != (xb.shape[0], hs, layer.out_channels)
            or state.device != xb.device or state.dtype != dtype):
        raise ValueError("convolution cache shape/device/dtype changed")
    history = torch.cat((state, projected), dim=1)
    weight = layer.conv_weight.to(dtype)
    values = sum(history[:, hs - lag:hs - lag + xb.shape[1]] * weight[lag]
                 for lag in range(layer.kernel_size))
    updated = history[:, -hs:].clone() if hs else history[:, :0].clone()
    return F.silu(values), updated


class HardEngine:
    """Post-norm DismBlock + SAM DISM cache + FLA GDN recurrent state."""

    def __init__(self, model, prefill_engine, *, capacity, planner_backend="cpu"):
        self.model = model
        self.prefill_engine = prefill_engine
        self.capacity = capacity
        self.planner = planner_backend
        self.dis = [None] * len(model.layers)
        self.gdn = [None] * len(model.layers)

    def _gdn(self, a, x, i):
        st = self.gdn[i]
        conv = (None, None, None) if st is None else st["conv"]
        q, cq = _conv(x, a.gdn_q_conv, conv[0])
        k, ck = _conv(x, a.gdn_k_conv, conv[1])
        v, cv = _conv(x, a.gdn_v_conv, conv[2])
        b, t, _ = x.shape
        q = q.reshape(b, t, a.heads, a.head_dim)
        k = k.reshape_as(q)
        v = v.reshape(b, t, a.heads, a.value_dim)
        beta = a.gdn_b_proj(x).sigmoid()
        g = -a.gdn_A_log.float().exp() * F.softplus(a.gdn_a_proj(x).float() + a.gdn_dt_bias.float())
        if st is None:
            out, hs = chunk_gated_delta_rule(q=q, k=k, v=v, g=g, beta=beta, initial_state=None,
                                             output_final_state=True, use_qk_l2norm_in_kernel=True)
        else:
            out, hs = fused_recurrent_gated_delta_rule(
                q=q, k=k, v=v, g=g, beta=beta, initial_state=st["state"],
                output_final_state=True, use_qk_l2norm_in_kernel=True)
        self.gdn[i] = dict(state=hs, conv=(cq, ck, cv))
        return out

    def _branch(self, a, x, i):
        if getattr(a, "is_pure_gdn", False):
            raw = self._gdn(a, x, i)
        else:
            st = self.dis[i]
            conv = (None, None, None) if st is None else st["conv"]
            q, cq = _conv(x, a.q_conv, conv[0])
            k, ck = _conv(x, a.k_conv, conv[1])
            v, cv = _conv(x, a.v_conv, conv[2])
            b, t, _ = x.shape
            q = q.reshape(b, t, a.heads, a.head_dim)
            k = k.reshape_as(q)
            v = v.reshape(b, t, a.heads, a.value_dim)
            sq = a._activate_readout(_linear(x, a.sq_proj))
            sk = a._activate_readout(_linear(x, a.sk_proj), is_key=True)
            iq = torch.einsum("bnhd,hvd->bnhv", q, a.q_vocab).argmax(-1).transpose(1, 2).int().contiguous()
            ik = torch.einsum("bnhd,hvd->bnhv", k, a.k_vocab).argmax(-1).transpose(1, 2).int().contiguous()
            tau = F.softplus(a.log_sel_tau)
            if st is None:
                cache = HardDismDecoder(b, a.heads, sq.shape[-1], v.shape[-1], self.capacity, tau,
                                        cache_dtype=torch.bfloat16, planner_backend=self.planner)
                raw = self.prefill_engine(iq, ik, sq, sk, v, tau)
                cache.prime(iq, ik, sk, v)
                self.dis[i] = dict(cache=cache, conv=(cq, ck, cv))
            else:
                raw = st["cache"].append(iq, ik, sq, sk, v)
                self.dis[i] = dict(cache=st["cache"], conv=(cq, ck, cv))
            if hasattr(a, "gdn_raw"):
                raw = raw + self._gdn(a, x, i)
        gate = a.g_proj_up(a.g_proj_down(x)).reshape_as(raw)
        out = raw * torch.rsqrt(raw.square().mean(-1, keepdim=True) + a.norm.eps)
        out = out * a.norm.weight * F.silu(gate)
        return _linear(out.flatten(2), a.o_proj)

    @torch.inference_mode()
    def __call__(self, tokens):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            x = self.model.embedding(tokens)
            for i, layer in enumerate(self.model.layers):
                if not layer.post_norm:
                    raise NotImplementedError("harness implements post-norm DismBlock only")
                branch = self._branch(layer.attn, x, i)
                x = layer.attn_norm(x + branch)
                x = layer.mlp_norm(x + layer.mlp(x))
            logits = self.model.lm_head(self.model.norm(x)).float()
        cap = self.model.config.softcap
        return cap * torch.tanh(logits / cap)


def sample(logits, *, temperature, top_k, top_p, repetition_penalty, seen, generator):
    logits = logits.clone()
    if repetition_penalty and repetition_penalty != 1.0 and seen:
        idx = torch.tensor(sorted(seen), device=logits.device, dtype=torch.long)
        vals = logits[..., idx]
        logits[..., idx] = torch.where(vals > 0, vals / repetition_penalty, vals * repetition_penalty)
    if temperature <= 0:
        return logits.argmax(-1, keepdim=True)
    logits = logits / temperature
    if top_k and top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.shape[-1]), dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p and top_p < 1.0:
        values, ids = logits.sort(dim=-1, descending=True)
        probs = values.softmax(-1)
        remove = probs.cumsum(-1) - probs > top_p
        values = values.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, ids, values)
    probs = logits.softmax(-1)
    return torch.multinomial(probs, 1, generator=generator)


def load(checkpoint, step, device="cuda"):
    metas = sorted(checkpoint.glob("meta_*.json"))
    if not metas:
        raise FileNotFoundError(f"no meta_*.json in {checkpoint}")
    if step is None:
        step = max(int(p.stem.split("_")[1]) for p in metas)
    meta = json.loads((checkpoint / f"meta_{step:06d}.json").read_text())
    with torch.device("meta"):
        model = build_model(ModelSpec.from_dict(meta["model_spec"]))
    model.to_empty(device="cpu")
    model.load_state_dict(torch.load(checkpoint / f"model_{step:06d}.pt",
                                     map_location="cpu", weights_only=True), strict=True)
    return model.to(device).eval(), meta, step


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--tokenizer", type=Path, default=None,
                        help="default: <run>/tokenizer, i.e. checkpoint.parents[1]/tokenizer")
    parser.add_argument("--prompt", type=str, default=None, help="overrides --preset")
    parser.add_argument("--prompt-file", type=Path, default=None, help="read prompt text from a file")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="science")
    parser.add_argument("-n", "--max-new-tokens", type=int, default=200)
    parser.add_argument("-t", "--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=0, help="0 disables")
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--greedy", action="store_true", help="temperature 0")
    parser.add_argument("--planner", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--mma-precision", choices=["bf16", "tf32x3"], default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stream", action="store_true", help="print tokens as they are produced")
    parser.add_argument("--raw", action="store_true", help="print only the continuation")
    args = parser.parse_args()

    if args.prompt_file is not None:
        prompt = args.prompt_file.read_text()
    elif args.prompt is not None:
        prompt = args.prompt
    else:
        prompt = PRESETS[args.preset]

    tokenizer_dir = args.tokenizer or (args.checkpoint.parents[1] / "tokenizer")
    tokenizer = HuggingFaceTokenizer.from_pretrained(str(tokenizer_dir))
    model, meta, step = load(args.checkpoint, args.step, args.device)
    bos = tokenizer.get_bos_token_id()
    eos = tokenizer.tokenizer.eos_token_id

    ids = [bos] + tokenizer.encode(prompt)
    ids = ids[: model.config.sequence_len - args.max_new_tokens]
    tokens = torch.tensor([ids], device=args.device, dtype=torch.long)
    temperature = 0.0 if args.greedy else args.temperature
    generator = torch.Generator(device=args.device).manual_seed(args.seed)

    with HardDismPrefill(workers=args.workers, mma_precision=args.mma_precision) as prefill:
        engine = HardEngine(model, prefill, capacity=len(ids) + args.max_new_tokens + 8,
                            planner_backend=args.planner)
        started = time.perf_counter()
        logits = engine(tokens)[:, -1]
        torch.cuda.synchronize()
        prefill_ms = (time.perf_counter() - started) * 1000
        if not args.raw:
            print("=" * 100)
            print(f"checkpoint step {step}  arch {meta['model_spec']['architecture']}  "
                  f"hard-inference harness (cached hybrid decoding is not a supported upstream path)")
            print(f"sampling: temperature={temperature} top_k={args.top_k} top_p={args.top_p} "
                  f"repetition_penalty={args.repetition_penalty} seed={args.seed}")
            print(f"prefill {len(ids)} tokens in {prefill_ms:.1f} ms")
            print("-" * 100)

        generated, seen = [], set(ids)
        streamed = ""
        started = time.perf_counter()
        for _ in range(args.max_new_tokens):
            token = sample(logits, temperature=temperature, top_k=args.top_k, top_p=args.top_p,
                           repetition_penalty=args.repetition_penalty, seen=seen, generator=generator)
            token_id = int(token)
            if token_id == eos:
                break
            generated.append(token_id)
            seen.add(token_id)
            if args.stream:
                now = tokenizer.decode(generated)
                if len(now) > len(streamed):
                    print(now[len(streamed):], end="", flush=True)
                    streamed = now
            logits = engine(token)[:, -1]
        torch.cuda.synchronize()
        decode_s = time.perf_counter() - started

    text = tokenizer.decode(generated)
    if args.raw:
        print(text)
        return
    if not args.stream:
        print(text)
    print("-" * 100)
    print(f"decoded {len(generated)} tokens in {decode_s:.2f} s "
          f"({len(generated)/max(decode_s, 1e-9):.1f} tok/s)")
    print("=" * 100)


if __name__ == "__main__":
    main()
