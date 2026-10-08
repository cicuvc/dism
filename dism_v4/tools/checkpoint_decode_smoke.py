"""Opt-in trained v3 hybrid LM integration of the v4 native hard decoder.

Keep checkpoint and production module code untouched. Projections/conv/SWA/MLP
use the existing FP32 Torch inference math; only the DISM core is replaced.
Prefill uses the existing exact Torch path and primes the native cache. This is
a correctness/generation harness, not an optimized end-to-end inference engine.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
NANO = Path('/home/cicuvc/cs/projects/deltastar/nanochat')
sys.path[:0] = [str(ROOT), str(ROOT / 'dism_v3/python'), str(NANO)]

import torch
from torch.nn import functional as F
from nanochat.models import build_model, ModelSpec
from nanochat.tokenizer import HuggingFaceTokenizer
from flash_dism.decoding import forward_torch, _convolve, _linear
from flash_dism.reference.dism_decode_ref import dism_wrapper_decode
from dism_v4.decoding import HardDismDecoder


class Runner:
    def __init__(self, model, capacity, backend='native', dtype=torch.float32, interval=32, planner='cpu'):
        self.model, self.capacity, self.backend = model, capacity, backend
        self.dtype, self.interval = dtype, interval
        self.planner=planner
        self.states = [None] * len(model.layers)

    def attention(self, module, x, index):
        state = self.states[index]
        direction = torch.ones((x.shape[0], module.heads), device=x.device, dtype=torch.bool)
        if self.backend == 'reference':
            out, recurrent, conv = forward_torch(module, x, state, direction=direction, hard_prob=1.)
            self.states[index] = dict(recurrent_state=recurrent, conv_state=conv)
            return out

        conv = (None, None, None) if state is None else state['conv']
        q, cq = _convolve(x, module.q_conv, conv[0])
        k, ck = _convolve(x, module.k_conv, conv[1])
        v, cv = _convolve(x, module.v_conv, conv[2])
        b, n, _ = x.shape
        q = q.reshape(b, n, module.heads, module.head_dim)
        k = k.reshape_as(q)
        v = v.reshape(b, n, module.heads, module.value_dim)
        sq = module._activate_readout(_linear(x, module.sq_proj))
        sk = module._activate_readout(_linear(x, module.sk_proj), is_key=True)
        iq = torch.einsum('bnhd,hvd->bnhv', q, module.q_vocab).argmax(-1).transpose(1, 2).int()
        ik = torch.einsum('bnhd,hvd->bnhv', k, module.k_vocab).argmax(-1).transpose(1, 2).int()
        if state is None:
            tau = F.softplus(module.log_sel_tau)
            cache = HardDismDecoder(b, module.heads, sq.shape[-1], v.shape[-1],
                self.capacity, tau, cache_dtype=self.dtype, rebuild_interval=self.interval,
                planner_backend=self.planner,
                sample_interval=32, materialize_threshold=64, rebuild_chunk=128)
            # Prefill output is deliberately identical to the existing reference.
            out, _ = dism_wrapper_decode(q, k, sq, sk, module.q_vocab, module.k_vocab,
                                        v, tau, direction=direction, hard_prob=1.)
            cache.prime(iq, ik, sk.to(self.dtype), v.to(self.dtype))
            extra, old_v = (), v[:, :0]
        else:
            cache, extra, old_v = state['cache'], state['extra'], state['v']
            out = cache.append(iq, ik, sq.to(self.dtype), sk.to(self.dtype), v.to(self.dtype))
        # Only the bounded SWA value window is duplicated, not full DISM history.
        window_v = torch.cat((old_v, v), dim=1)
        proxy = SimpleNamespace(length=cache.position, v=window_v)
        out, extra = module._combine_torch(out, x, proxy, extra)
        keep = module.window_size - 1
        self.states[index] = dict(cache=cache, conv=(cq, ck, cv), extra=extra,
                                 v=window_v[:, -keep:].clone() if keep else window_v[:, :0])
        gate = _linear(_linear(x, module.g_proj_down), module.g_proj_up).reshape_as(out)
        out = out * torch.rsqrt(out.square().mean(-1, keepdim=True) + module.norm.eps)
        out = out * module.norm.weight * F.silu(gate)
        return _linear(out.flatten(2), module.o_proj)

    @torch.inference_mode()
    def __call__(self, tokens):
        x = self.model.embedding(tokens)
        for i, layer in enumerate(self.model.layers):
            x = x + self.attention(layer.attn, layer.attn_norm(x), i)
            x = x + layer.mlp(layer.mlp_norm(x))
        logits = self.model.lm_head(self.model.norm(x)).float()
        cap = self.model.config.softcap
        return cap * torch.tanh(logits / cap)


def sample(logits, temperature, top_p, generator):
    if temperature == 0:
        return logits.argmax(-1, keepdim=True)
    values, ids = (logits / temperature).sort(descending=True)
    probability = values.softmax(-1)
    remove = probability.cumsum(-1) - probability > top_p
    probability.masked_fill_(remove, 0.)
    selected = torch.multinomial(probability, 1, generator=generator)
    return ids.gather(-1, selected)


def main():
    parser = argparse.ArgumentParser()
    base = '/media/cicuvc/c63abdf1-0e56-4153-9228-95df5a2f239b/cicuvc/dism-runs'
    parser.add_argument('--checkpoint', type=Path, default=Path(base) / 'nanochat-hybrid125m-2500m/base_checkpoints/dism125m/model_019074.pt')
    parser.add_argument('--output', type=Path, default=ROOT / 'dism_v4/decoding/results/checkpoint_smoke.json')
    parser.add_argument('--steps', type=int, default=192)
    parser.add_argument('--prefill', type=int, default=37)
    parser.add_argument('--planner',choices=['cpu','gpu'],default='cpu')
    parser.add_argument('--new-tokens', type=int, default=128)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    meta = json.loads(args.checkpoint.with_name(args.checkpoint.name.replace('model_', 'meta_')).with_suffix('.json').read_text())
    with torch.device('meta'):
        model = build_model(ModelSpec.from_dict(meta['model_spec']))
    model.to_empty(device='cpu')
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=True), strict=True)
    model.cuda().eval()
    tokenizer = HuggingFaceTokenizer.from_pretrained(str(NANO / 'runs/dism72m_3k/tokenizer'))
    # Transformers renamed LlamaTokenizerFast to LlamaTokenizer. The saved
    # fingerprint includes the Python class name; verify the SAME serialized
    # backend under the historical class identity rather than waiving the check.
    historical_identity = 'transformers.models.llama.tokenization_llama_fast\0LlamaTokenizerFast\0'
    historical_fingerprint = hashlib.sha256((historical_identity +
        tokenizer.tokenizer.backend_tokenizer.to_str()).encode()).hexdigest()
    assert meta['tokenizer_spec']['fingerprint'] in (
        tokenizer.get_fingerprint(), historical_fingerprint), 'tokenizer mismatch'
    assert tokenizer.get_bos_token_id() == 1 and tokenizer.tokenizer.eos_token_id == 2
    prompts = [
        'The solar system consists of the Sun and the objects that orbit it. The largest planet is',
        'Photosynthesis is the process by which plants convert sunlight into chemical energy. During this process,',
        'Once upon a time, a young girl found a mysterious book in the library. When she opened it,',
    ]
    def encode(text):
        return torch.tensor([[1] + tokenizer.encode(text)], device='cuda', dtype=torch.long)

    report = dict(checkpoint=str(args.checkpoint), step=meta['step'], config=meta['model_config'],
                  planner=args.planner,
                  tokenizer_fingerprint=tokenizer.get_fingerprint(),
                  historical_tokenizer_fingerprint=historical_fingerprint,
                  protocol='full-hard, FP32 existing Torch model math; native DISM FP32 accumulation, optional BF16 payload; exact Torch prefill + native prime; no production monkeypatch',
                  parity={}, generations=[])
    with args.checkpoint.open('rb') as f:
        report['checkpoint_sha256'] = hashlib.file_digest(f, 'sha256').hexdigest()
    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))

    with torch.inference_mode():
        forced = encode((' '.join(prompts) + ' ') * (1 + (args.prefill+args.steps)//50))
        prefix = forced[:, :args.prefill]
        for dtype in (torch.float32, torch.bfloat16):
            reference = Runner(model, forced.shape[1]+1, 'reference')
            native = Runner(model, forced.shape[1]+1, dtype=dtype,planner=args.planner)
            expected, actual = reference(prefix), native(prefix)
            torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
            errors, rels, cosines, agrees, kls = [], [], [], [], []
            started = time.perf_counter()
            for t in range(args.prefill, args.prefill+args.steps):
                token = forced[:, t:t+1]
                expected, actual = reference(token)[:, -1], native(token)[:, -1]
                assert torch.isfinite(actual).all()
                errors.append((expected-actual).abs().max().item())
                rels.append(((expected-actual).norm()/expected.norm()).item())
                cosines.append(F.cosine_similarity(expected, actual).item())
                agrees.append(int(expected.argmax() == actual.argmax()))
                kls.append((expected.softmax(-1)*(expected.log_softmax(-1)-actual.log_softmax(-1))).sum().item())
            key = str(dtype)
            report['parity'][key] = dict(steps=args.steps, prefill=args.prefill, rebuild_interval=32,
                max_abs=max(errors), relative_l2_max=max(rels), cosine_min=min(cosines),
                top1_agreement=sum(agrees)/len(agrees), kl_mean=sum(kls)/len(kls),
                seconds=time.perf_counter()-started, layer_cache=[s['cache'].memory_stats() for s in native.states])
            print('PARITY', key, report['parity'][key], flush=True)
            save()
            if dtype == torch.float32:
                assert max(rels) < 1e-3 and min(cosines) > .99999, 'FP32 end-to-end parity failed'
            del reference, native

        for i, prompt in enumerate(prompts):
            tokens = encode(prompt)
            runner = Runner(model, tokens.shape[1]+args.new_tokens+1,planner=args.planner)
            logits = runner(tokens)[:, -1]
            generator = torch.Generator(device='cuda').manual_seed(20261008+i)
            generated = []
            trajectory = []
            torch.cuda.synchronize()
            started = time.perf_counter()
            for _ in range(args.new_tokens):
                trajectory.append(logits)
                token = sample(logits, .8, .9, generator)
                generated.append(token.item())
                if token.item() == 2:
                    break
                logits = runner(token)[:, -1]
                assert torch.isfinite(logits).all()
            torch.cuda.synchronize()
            seconds = time.perf_counter()-started
            # Replay exactly the sampled token stream through the old cache.
            # This is outside the native generation timing interval.
            reference = Runner(model, runner.capacity, 'reference')
            expected = reference(tokens)[:, -1]
            generator_ref = torch.Generator(device='cuda').manual_seed(20261008+i)
            trajectory_error, sample_agree = 0., 0
            for j, token_id in enumerate(generated):
                trajectory_error = max(trajectory_error, (trajectory[j]-expected).abs().max().item())
                sample_agree += int(sample(expected, .8, .9, generator_ref).item() == token_id)
                if j+1 < len(generated):
                    expected = reference(tokens.new_tensor([[token_id]]))[:, -1]
            assert trajectory_error < .001, 'FP32 generated trajectory differs from reference'
            item = dict(prompt=prompt, continuation=tokenizer.decode(generated), tokens=generated,
                        temperature=.8, top_p=.9, seed=20261008+i, seconds=seconds,
                        tokens_per_second=len(generated)/seconds,
                        reference_max_abs=trajectory_error,
                        reference_same_seed_sample_agreement=sample_agree/len(generated))
            report['generations'].append(item)
            print('GENERATION', json.dumps(item), flush=True)
            save()
        report['status'] = 'passed'
        save()


if __name__ == '__main__':
    main()
