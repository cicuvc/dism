"""Offline hard-attention diagnostics from actual production forward labels."""
import argparse
import json
import math
import os
from pathlib import Path

import torch
from torch.nn import functional as F


def hard_attention(qi, ki, tau):
    """Exact natural-log reference recurrence; diagnostic CPU dense allocation."""
    heads, n = qi.shape
    w = torch.full((heads, n, n), -torch.inf)
    for i in range(n):
        row = tau[:, None].expand(heads, i+1).clone()
        if i:
            row[:, 1:] += F.softplus(w[:, i-1, :i])
        w[:, i, :i+1] = row.masked_fill(qi[:, i:i+1] != ki[:, :i+1], -torch.inf)
    z = torch.logaddexp(torch.logsumexp(w, -1), torch.zeros(heads, n))
    return w, torch.exp(w-z[..., None]), torch.exp(-z), z


def plot_needle_comparison(output, tokenizer):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 10), constrained_layout=True)
    for ax, name in zip(axes, ('niah_success', 'niah_failure')):
        data = torch.load(output/f'{name}.pt', weights_only=True)
        lo, hi = data['needle_span']
        weights = torch.cat([d['last_attention'][:,lo:hi] for d in data['layers']])
        im=ax.imshow(weights, aspect='auto', vmin=0, vmax=1, cmap='magma', interpolation='nearest')
        ax.set_xticks(range(hi-lo), [repr(tokenizer.decode([i])) for i in data['token_ids'][lo:hi]], rotation=70)
        ax.set_yticks(range(60), [f'L{l+1}H{h+1}' for l in range(15) for h in range(4)], fontsize=6)
        ax.set(title=name, xlabel='Needle token', ylabel='Layer/head')
    fig.colorbar(im, ax=axes, label='Final-query attention probability', shrink=.8)
    fig.suptitle('Same answer yellow: retrieval location versus successful output')
    fig.savefig(output/'needle_comparison.png', dpi=180)
    plt.close(fig)


def plots(output, name, data, needle):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    layers = [0, 4, 9, 14]
    n = data[0]['length']
    fig, axes = plt.subplots(4, 4, figsize=(13, 12), constrained_layout=True)
    for r, layer in enumerate(layers):
        for head in range(4):
            ax = axes[r, head]
            a = data[layer]['binned_attention'][head]
            im = ax.imshow(torch.log10(a.clamp_min(1e-5)), origin='upper', extent=(0,n,n,0),
                           vmin=-5, vmax=0, cmap='magma', aspect='auto')
            ax.set_title(f'Layer {layer+1}, head {head+1}')
            if needle:
                ax.axvline(needle[0], color='cyan', linewidth=.7)
            if r==3: ax.set_xlabel('Key position')
            if head==0: ax.set_ylabel('Query position')
    fig.colorbar(im, ax=axes, label='log10(attention mass per key bin), query-averaged', shrink=.7)
    fig.suptitle(name + ': exact hard recurrence from CUDA-selected labels')
    fig.savefig(output / f'{name}_attention.png', dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    metrics = [('local_mass', 'Mass at distance 0..127', 0,1),
               ('far_mass', 'Mass at distance >=512', 0,1),
               ('fallback', 'Zero-value fallback mass',0,1),
               ('q_effective_labels', 'Q effective label count',1,512),
               ('k_effective_labels', 'K effective label count',1,512),
               ('mean_max_weight', 'Mean largest attention weight',0,1)]
    for ax, (key, title, lo, hi) in zip(axes.flat, metrics):
        v = np.array([[h[key] for h in layer['heads']] for layer in data])
        im = ax.imshow(v, aspect='auto', vmin=lo, vmax=hi, cmap='viridis')
        ax.set(title=title, xlabel='Head', ylabel='Layer', xticks=range(4), xticklabels=range(1,5),
               yticks=range(15), yticklabels=range(1,16))
        fig.colorbar(im, ax=ax, shrink=.8)
    fig.suptitle(name + ': mass statistics use queries at positions >=512 (zero-based)')
    fig.savefig(output / f'{name}_heads.png', dpi=150)
    plt.close(fig)
    last = torch.cat([d['last_attention'] for d in data])
    fig, ax = plt.subplots(figsize=(13, 7), constrained_layout=True)
    im = ax.imshow(torch.log10(last.clamp_min(1e-5)), aspect='auto', vmin=-5, vmax=0, cmap='magma',
                   extent=(0,n,60,0))
    ax.set(xlabel='Key position', ylabel='Layer/head (four rows per layer)',
           title=name + ': final-query attention, all 60 heads')
    ax.set_yticks(np.arange(15)*4+2, [str(i) for i in range(1,16)])
    if needle:
        ax.axvspan(*needle, color='cyan', alpha=.35, label='Needle token span')
        ax.legend()
    fig.colorbar(im, ax=ax, label='log10 attention probability')
    fig.savefig(output / f'{name}_last_query.png', dpi=150)
    plt.close(fig)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Choose a new output directory')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    saved = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    cfg, step = saved['config'], saved['step']
    os.environ['DISM_TILE_LSE'] = cfg['tile_lse']
    os.environ['DISM_BWD_OPT'] = str(cfg['backward_opt'])
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    from .lm_model import DecoderLM, LMConfig
    from .train_lm import load_tokenizer
    from .lm_data import PackedStream, split_files
    from .eval_lm_niah import build_prompt
    from . import autograd
    model = DecoderLM(LMConfig(**cfg['model'])).cuda().eval()
    model.load_state_dict(saved['model'], strict=True)
    del saved
    tok = load_tokenizer(cfg['tokenizer'])
    enc = lambda s: tok.encode(s, add_special_tokens=False)
    _, files = split_files(cfg['data'])
    bg = PackedStream(files, tok, context=8192, batch_size=2, repeat=False).next_batch()[0].tolist()
    samples = [('natural', bg[0][:1024], None)]
    for index, key in enumerate(('silver telescope', 'wooden compass')):
        prefix = f'The secret color of the {key} is'
        needle = enc('\n\n'+prefix+' yellow.\n\n')
        ids, at = build_prompt(bg[index], 2048, needle, enc('\n\n'+prefix), .9)
        samples.append(('niah_'+('success' if index==0 else 'failure'), ids, (at,at+len(needle))))
    a.output.mkdir(parents=True)
    reports = {}
    for name, ids, needle in samples:
        captures = []
        original = autograd.forward
        def traced(*args, **kwargs):
            result = original(*args, **kwargs)
            _, _, v, _, tau, qi, ki = args
            captures.append(dict(qi=qi[0].cpu(), ki=ki[0].cpu(), tau=tau.cpu(),
                                 v=v[0].cpu(), out=result[0][0].cpu(), norm=result[1][0].cpu()))
            return result
        x = torch.tensor([ids], device='cuda')
        def run():
            with torch.autocast('cuda', dtype=torch.bfloat16):
                return model.forward_features(x, 1., torch.Generator(device='cuda').manual_seed(779))
        try:
            autograd.forward = traced
            features = run()
        finally:
            autograd.forward = original
        # Tracing must not alter the actual production computation.
        torch.testing.assert_close(features, run(), atol=0, rtol=0)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model.lm_head(features[:, -1]).float()
        prediction = tok.decode([logits.argmax(-1).item()])
        assert len(captures)==15
        n = len(ids)
        distance = torch.arange(n)[:,None]-torch.arange(n)[None,:]
        valid = torch.arange(n)>=512
        data = []
        for layer, cap in enumerate(captures):
            w, weights, fallback, z = hard_attention(cap['qi'], cap['ki'], cap['tau'])
            # End-to-end diagnostics: compare reconstructed last16 O rows to CUDA.
            predicted = weights[:, -16:] @ cap['v'].float()
            actual = cap['out'][:, -16:].float()
            error = (predicted-actual).norm()/actual.norm().clamp_min(1e-8)
            normalizer_error = (z - cap['norm']*math.log(2)).abs().max().item()
            heads=[]
            for head in range(4):
                phead = weights[head]
                stats = dict(tau=cap['tau'][head].item(),
                    local_mass=(phead*(distance>=0)*(distance<128))[valid].sum(-1).mean().item(),
                    far_mass=(phead*(distance>=512))[valid].sum(-1).mean().item(),
                    fallback=fallback[head, valid].mean().item(),
                    mean_max_weight=phead[valid].max(-1).values.mean().item(),
                    mean_match_count=torch.isfinite(w[head, valid]).sum(-1).float().mean().item(),
                    last_fallback=fallback[head,-1].item(),
                    last_needle_mass=phead[-1,needle[0]:needle[1]].sum().item() if needle else None)
                for kind in ('qi','ki'):
                    counts = torch.bincount(cap[kind][head].long(), minlength=512).float()
                    probs = counts/counts.sum()
                    pre = kind[0]
                    stats[pre+'_effective_labels']=torch.exp(-(probs*probs.clamp_min(1e-30).log()).sum()).item()
                    stats[pre+'_max_label_fraction']=probs.max().item()
                values, indices = phead[-1].topk(5)
                stats['last_top_keys']=[dict(position=j, weight=pv, token=tok.decode([ids[j]]),
                    context=tok.decode(ids[max(0,j-4):min(n,j+5)]))
                    for pv,j in zip(values.tolist(),indices.tolist()) if pv>0]
                heads.append(stats)
            block=n//256
            binned=weights.reshape(4,256,block,256,block).sum(-1).mean(2)
            data.append(dict(layer=layer+1,length=n,heads=heads,
                reconstruction_last16_relative_l2=error.item(), log_normalizer_max_abs_error=normalizer_error,
                binned_attention=binned, last_attention=weights[:,-1].clone(),
                qi=cap['qi'],ki=cap['ki'],tau=cap['tau']))
            print(json.dumps(dict(sample=name,layer=layer+1,relative_l2=error.item(),norm_error=normalizer_error)),flush=True)
        torch.save(dict(token_ids=ids,needle_span=needle,layers=data),a.output/f'{name}.pt')
        (a.output/f'{name}_text.txt').write_text(tok.decode(ids))
        plots(a.output,name,data,needle)
        reports[name]=dict(length=n,needle_span=needle,greedy_prediction=prediction,
            tracing_bitwise_equal=True,layers=[{k:v for k,v in d.items() if k not in
                ('binned_attention','last_attention','qi','ki','tau')} for d in data])
    (a.output/'report.json').write_text(json.dumps(dict(checkpoint=str(a.checkpoint),step=step,
        hard_prob=1., production_tile_lse=cfg['tile_lse'],samples=reports,
        interpretation='Exact full-LSE recurrence reconstructed from actual CUDA labels; not bitwise tile approximation.',
        score_semantics='raw logM=tau on matching labels, -inf otherwise; W=logM+softplus(diagonal predecessor); P=exp(W)/(1+sum exp(W))',
        selection='One natural prefix and two outcome-selected pilot NIAH cases, not aggregate causal evidence.'),indent=2)+'\n')
    plot_needle_comparison(a.output, tok)


if __name__ == '__main__':
    main()
