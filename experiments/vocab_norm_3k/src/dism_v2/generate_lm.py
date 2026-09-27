"""Basic local LM generation: original CUDA prefix baseline or CPU hard cursors."""
import argparse
import json
import os
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from .hard_decode_cpu import HardDismCPUCache


class IncrementalHybrid:
    """Batch-one, pure-hard inference; GPU layers + CPU DISM KV/cursors.

    FP32 label GEMM and exact CPU recurrence are not bitwise equivalent to
    training's BF16/tanh tile path. Keep the original prefix backend as control.
    """
    def __init__(self, model):
        if model.config.architecture != 'hybrid' or model.training:
            raise ValueError('Requires eval-mode original hybrid model')
        self.model = model
        self.position = 0
        self.convs = [{} for _ in model.blocks]
        self.swa = [None for _ in model.blocks]
        self.dism = [HardDismCPUCache(F.softplus(b.dism.log_sel_tau.detach().float()).cpu(),
                                     value_dim=model.config.head_dim)
                     for b in model.blocks]

    @torch.inference_mode()
    def step(self, token):
        from flash_attn import flash_attn_func
        c = self.model.config
        if token.shape != (1, 1) or self.position >= c.context:
            raise ValueError('Expected one token within configured context')
        x = self.model.embedding(token)
        for index, block in enumerate(self.model.blocks):
            h = block.norm1(x)
            d = block.dism
            def project(name, proj, conv):
                y, state = conv(proj(h), cache=self.convs[index].get(name), output_final_state=True)
                self.convs[index][name] = state
                return y.reshape(1, c.heads, c.head_dim)
            q = project('q', d.q_proj, d.qd_conv)
            k = project('k', d.k_proj, d.kd_conv)
            v = project('v', d.v_proj, d.v_dism)
            qvoc, kvoc = d.expanded_vocabularies(q.dtype)
            with torch.autocast('cuda', enabled=False):
                qi = torch.einsum('bhd,hvd->bhv', q.float(), qvoc.float()).argmax(-1)
                ki = torch.einsum('bhd,hvd->bhv', k.float(), kvoc.float()).argmax(-1)
            out = self.dism[index].step(qi.cpu(), ki.cpu(), v.cpu()).to(device=x.device, dtype=v.dtype)
            gate = d.g_proj_up(d.g_proj_down(h))
            dout = d.o_proj(d.norm(out.reshape(1, 1, c.width), gate))
            s = block.swa
            qs, ks, vs = s.qkv(h).reshape(1, 1, 3, c.heads, c.head_dim).unbind(2)
            def rope(t):
                co = s.cos[self.position].to(t.dtype)
                si = s.sin[self.position].to(t.dtype)
                a, b = t[..., 0::2], t[..., 1::2]
                return torch.stack((a * co - b * si, a * si + b * co), -1).flatten(-2)
            qs, ks = rope(qs), rope(ks)
            old = self.swa[index]
            if old is not None:
                ks, vs = torch.cat((old[0], ks), 1), torch.cat((old[1], vs), 1)
            ks, vs = ks[:, -c.window:].contiguous(), vs[:, -c.window:].contiguous()
            self.swa[index] = ks, vs
            # All cached keys precede or equal the single current query.
            sout = flash_attn_func(qs, ks, vs, causal=False, dropout_p=0.)
            x = x + dout + s.out(sout.reshape(1, 1, c.width))
            g, value = block.up(block.norm2(x)).chunk(2, -1)
            x = x + block.down(F.silu(g) * value)
        self.position += 1
        return self.model.lm_head(self.model.final_norm(x))[:, -1].float()


def cap_logits(logits, cap):
    return logits if cap is None else cap * torch.tanh(logits / cap)


def sample(logits, temperature, top_p, generator):
    if not torch.isfinite(logits).all():
        raise FloatingPointError('Nonfinite generation logits')
    if temperature == 0:
        return logits.argmax(-1, keepdim=True)
    scores, ids = (logits / temperature).sort(descending=True, dim=-1)
    probs = scores.softmax(-1)
    discard = probs.cumsum(-1) - probs >= top_p
    probs = probs.masked_fill(discard, 0)
    return ids.gather(-1, torch.multinomial(probs, 1, generator=generator))


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--backend', choices=['prefix', 'cpu-cache'], default='prefix')
    p.add_argument('--prompt', action='append')
    p.add_argument('--max-new-tokens', type=int, default=96)
    p.add_argument('--temperature', type=float, default=.8)
    p.add_argument('--top-p', type=float, default=.9)
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--compare-tokens', type=int, default=0)
    a = p.parse_args()
    if a.output.exists() or a.temperature < 0 or not 0 < a.top_p <= 1 or a.max_new_tokens <= 0:
        raise ValueError('Require a new report path and valid sampling parameters')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    saved = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    config, step = saved['config'], saved['step']
    os.environ['DISM_TILE_LSE'] = config['tile_lse']
    os.environ['DISM_BWD_OPT'] = str(config['backward_opt'])
    from .lm_model import DecoderLM, LMConfig
    from .train_lm import load_tokenizer
    model = DecoderLM(LMConfig(**config['model']))
    model.load_state_dict(saved['model'], strict=True)
    del saved
    model.cuda().eval()
    tokenizer = load_tokenizer(config['tokenizer'])
    rng = torch.Generator(device='cuda').manual_seed(a.seed)
    direction_rng = torch.Generator(device='cuda').manual_seed(779)
    prompts = a.prompt or ['The solar system consists of', 'Photosynthesis is the process by which',
                            'Once upon a time, in a small village,',
                            'To calculate the area of a circle,']
    report = dict(checkpoint=str(a.checkpoint), step=step, backend=a.backend, hard_prob=1.,
                  softcap=model.config.softcap, temperature=a.temperature, top_p=a.top_p,
                  seed=a.seed, samples=[])
    def prefix(ids):
        n = ids.shape[1]
        # Installed FLA's stateless T=1 conv update omits current x without
        # cache. A causal, ignored extra position selects the normal conv path.
        padded = torch.cat((ids, ids[:, -1:]), 1) if n == 1 else ids
        return model.lm_head(model.forward_features(padded, 1., direction_rng)[:, n-1]).float()
    if a.compare_tokens:
        ids = tokenizer(' '.join(prompts), add_special_tokens=False, return_tensors='pt').input_ids.cuda()
        ids = ids[:, :a.compare_tokens]
        cache = IncrementalHybrid(model)
        checks = []
        with torch.autocast('cuda', dtype=torch.bfloat16):
            for i in range(ids.shape[1]):
                x = cap_logits(cache.step(ids[:, i:i+1]), model.config.softcap)
                y = cap_logits(prefix(ids[:, :i+1]), model.config.softcap)
                checks.append(dict(position=i, max_abs=(x-y).abs().max().item(),
                                   rms=(x-y).square().mean().sqrt().item(),
                                   cosine=F.cosine_similarity(x, y).item(),
                                   top1_equal=bool((x.argmax(-1)==y.argmax(-1)).item())))
        report['incremental_comparison'] = checks
        print(json.dumps(dict(event='comparison', checks=checks)), flush=True)
    for prompt in prompts:
        ids = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').input_ids.cuda()
        if not ids.numel() or ids.shape[1] + a.max_new_tokens > model.config.context:
            raise ValueError('Prompt must be nonempty and fit context with continuation')
        initial = ids.shape[1]
        cache = IncrementalHybrid(model) if a.backend == 'cpu-cache' else None
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            if cache:
                for i in range(initial):
                    logits = cache.step(ids[:, i:i+1])
            for i in range(a.max_new_tokens):
                if cache is None:
                    logits = prefix(ids)
                nxt = sample(cap_logits(logits, model.config.softcap), a.temperature, a.top_p, rng)
                ids = torch.cat((ids, nxt), 1)
                if nxt.item() == tokenizer.eos_token_id:
                    break
                if cache and i + 1 < a.max_new_tokens:
                    logits = cache.step(nxt)
        torch.cuda.synchronize()
        result = dict(prompt=prompt, continuation=tokenizer.decode(ids[0, initial:].tolist()),
                      token_ids=ids[0, initial:].tolist(), new_tokens=ids.shape[1]-initial,
                      eos=ids[0, -1].item()==tokenizer.eos_token_id,
                      seconds_including_prefill=time.perf_counter()-start)
        report['samples'].append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
