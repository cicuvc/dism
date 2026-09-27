"""Matched frozen-sequence GDN vs GDN/DISM per-position evaluation."""
import os
for key, value in dict(DISM_TILE_LSE='tanh_finite', DISM_BWD_OPT='13',
                       DISM_ROW_BITSET='0', DISM_OUTPUT_Q_ALIAS='kv',
                       DISM_OUTPUT_TMA='0').items():
    os.environ.setdefault(key, value)
import hashlib
import json
from pathlib import Path
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


@torch.inference_mode()
def main():
    from dism_v2.eval_lm_positions import load_checkpoint, evaluate_context
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    root = Path('/media/cicuvc/c63abdf1-0e56-4153-9228-95df5a2f239b/cicuvc/dism-lm-runs')
    study = root / 'activation-study-3k-20260910'
    out = study / 'gdn-position-256-20260911'
    out.mkdir(exist_ok=False)
    bundle = torch.load(root / 'swa-eval-bundle-20260910.pt', map_location='cpu', weights_only=False)
    x, y = (bundle[k][:256, :2048].long().contiguous() for k in ('x', 'y'))
    assert x.shape == y.shape == (256, 2048)
    report = dict(sequences=256, context=2048,
        token_sha256=hashlib.sha256(x.numpy().tobytes()+y.numpy().tobytes()).hexdigest(),
        models={}, caveats=['Paired sequence SE/95% intervals are sampling uncertainty, not training-seed uncertainty.',
          'Packed sequences may be correlated; one training seed.',
          'FFN widths 1428 (mixed) and 1392 (pure) compensate for attention parameter differences.',
          'Later-position advantage alone does not prove long-range retrieval causality.'])
    losses = {}
    for arm, probability in [('gdn_dism_384_matched', 1.), ('gdn_only_384_matched', 0.)]:
        model, meta = load_checkpoint(study / arm / 'latest.pt')
        assert meta['step'] == 3000
        report['models'][arm] = meta
        if len(report['models']) == 2:
            a, b = list(report['models'].values())
            for key in ('batch', 'micro_batch', 'steps', 'lr', 'weight_decay', 'warmup', 'seed', 'data', 'tokenizer'):
                assert a['config'][key] == b['config'][key], key
            for key in ('width', 'layers', 'heads', 'head_dim', 'context', 'softcap'):
                assert a['model'][key] == b['model'][key], key
            assert abs(a['parameters'] - b['parameters']) / a['parameters'] < .001
        pieces = []
        for i in range(0, 256, 2):
            value = evaluate_context(model, x[i:i+2].cuda(), y[i:i+2].cuda(), probability, 779)
            assert torch.isfinite(value).all()
            pieces.append(value)
            if (i+2) % 32 == 0:
                print(arm, i+2, float(torch.cat(pieces).mean()), flush=True)
        losses[arm] = torch.cat(pieces)
        torch.save(losses[arm], out / (arm + '.pt'))
        del model
        torch.cuda.empty_cache()
    mixed, pure = [losses[k].double() for k in ('gdn_dism_384_matched', 'gdn_only_384_matched')]
    delta = mixed - pure
    def stats(lo, hi):
        paired = delta[:, lo:hi].mean(1)
        mean, se = float(paired.mean()), float(paired.std() / len(paired)**.5)
        return dict(start=lo, end_exclusive=hi, mixed=float(mixed[:, lo:hi].mean()),
                    pure=float(pure[:, lo:hi].mean()), delta=mean, paired_se=se,
                    ci95=[mean-1.96*se, mean+1.96*se])
    report['overall'] = stats(0, 2048)
    report['segments'] = [stats(a,b) for a,b in [(0,128),(128,256),(256,512),(512,1024),(1024,2048)]]
    report['bins64'] = [stats(i,i+64) for i in range(0,2048,64)]
    seq_trend = delta[:,1024:].mean(1)-delta[:,:512].mean(1)
    report['late_minus_early_delta'] = dict(mean=float(seq_trend.mean()), paired_se=float(seq_trend.std()/16))
    report['per_position'] = dict(mixed=mixed.mean(0).tolist(), pure=pure.mean(0).tolist(),
        delta=delta.mean(0).tolist(), paired_se=(delta.std(0)/16).tolist())
    (out/'report.json').write_text(json.dumps(report, indent=2))
    bins=report['bins64']; positions=[(s['start']+s['end_exclusive']-1)/2 for s in bins]
    fig, axes=plt.subplots(2,1,figsize=(10,7),sharex=True,layout='constrained')
    axes[0].plot(positions,[s['mixed'] for s in bins],label='9 GDN + 3 DISM (72.188M)')
    axes[0].plot(positions,[s['pure'] for s in bins],label='12 GDN (72.184M)')
    axes[0].set_ylabel('NLL (nats), 64-token bins'); axes[0].legend()
    axes[0].set_title('3000 steps, 256 paired held-out sequences, context 2048')
    axes[1].plot(positions,[s['delta'] for s in bins],color='tab:purple',label='Mixed minus pure GDN')
    axes[1].fill_between(positions,[s['ci95'][0] for s in bins],[s['ci95'][1] for s in bins],alpha=.2,color='tab:purple',label='Pointwise paired-sequence 95% interval')
    axes[1].axhline(0,color='black',linewidth=.8);axes[1].set_ylabel('Delta NLL (negative favors DISM)')
    axes[1].set_xlabel('Token position (zero-based)');axes[1].legend()
    for ax in axes: ax.grid(alpha=.2)
    fig.savefig(out/'per_position.png',dpi=180)
    print(json.dumps({k:report[k] for k in ('overall','segments','late_minus_early_delta')},indent=2),flush=True)
    print(out,flush=True)


if __name__ == '__main__':
    main()
