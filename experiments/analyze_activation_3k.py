"""Paired checkpoint validation and CPU-only plots for the first three arms."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import torch

MODES = ('baseline', 'vocab_silu', 'no_qk_silu')


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    a.output.mkdir(exist_ok=False,parents=True)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    os.environ['DISM_TILE_LSE']='tanh_finite'
    os.environ['DISM_BWD_OPT']='13'
    from dism_v2.eval_lm_positions import load_checkpoint,evaluate_context
    bundle=torch.load(a.bundle,map_location='cpu',weights_only=False)
    x,y=bundle['x'][:256,:2048].long(),bundle['y'][:256,:2048].long()
    losses,logs,usage,configs={},{},{},{}
    for mode in MODES:
        model,meta=load_checkpoint(a.root/mode/'latest.pt')
        assert meta['step']==3000 and meta['parameters']==49678876
        assert meta['model']['dism_activation']==mode
        configs[mode]=meta['config']
        logs[mode]=[json.loads(s) for s in (a.root/mode/'metrics.jsonl').read_text().splitlines()]
        usage[mode]=json.loads((a.root/mode/'vocab-load-final/report.json').read_text())
        losses[mode]={}
        for prob in (1.,.5,0.):
            parts=[]
            for start in range(0,len(x),2):
                v=evaluate_context(model,x[start:start+2].cuda(),y[start:start+2].cuda(),prob,779)
                assert torch.isfinite(v).all()
                parts.append(v)
            losses[mode][prob]=torch.cat(parts)
            print(json.dumps(dict(mode=mode,hard_prob=prob,nll=float(losses[mode][prob].double().mean()))),flush=True)
        del model
        torch.cuda.empty_cache()
    for key in ('seed','steps','warmup','lr','weight_decay','batch','micro_batch','data','tokenizer','parameters','tile_lse','backward_opt'):
        assert all(configs[m][key]==configs['baseline'][key] for m in MODES),key
    assert len({usage[m]['packed_input_sha256'] for m in MODES})==1
    summary={}
    for mode in MODES:
        rows=usage[mode]['rows']
        summary[mode]=dict(validation=[r for r in logs[mode] if 'val/loss' in r],
            vocab={side:{key:statistics.mean(r[side][key] for r in rows)
                         for key in ('effective_vocab','unused','top1_mass','top8_mass')}
                   for side in ('q','k')},
            alignment={key:statistics.mean(r['alignment'][key] for r in rows)
                       for key in ('overlap','q_mass_on_unused_k','independent_label_match_probability')},paired={})
        for prob,values in losses[mode].items():
            delta=(values-losses['baseline'][prob]).double().mean(-1)
            softdelta=(values-losses[mode][1.]).double().mean(-1)
            summary[mode]['paired'][str(prob)]=dict(nll=float(values.double().mean()),
                delta=float(delta.mean()),se=float(delta.std()/len(x)**.5),
                delta_vs_own_hard=float(softdelta.mean()),se_vs_own_hard=float(softdelta.std()/len(x)**.5),
                position_delta=[float(t) for t in (values-losses['baseline'][prob]).double().reshape(256,8,256).mean((0,2))])
    report=dict(sequences=256,context=2048,head_chunk=256,micro_batch=2,seed=779,
        token_sha256=hashlib.sha256(x.numpy().tobytes()+y.numpy().tobytes()).hexdigest(),
        summary=summary,caveats=['Single training seed; paired sequence SE is not between-training-seed uncertainty.',
        'Packed texts have correlations; sequence intervals are descriptive.',
        'Final-checkpoint soft/mixed evaluation is an inference intervention, not retraining.',
        'RNG aligned across models within each probability; probability changes alter RNG consumption/global directions.',
        'Sample differs from original100-effective-batch validation; do not compare absolute NLL across samples.'])
    torch.save(losses,a.output/'paired_losses.pt')
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(2,2,figsize=(12,9),constrained_layout=True)
    for mode in MODES:
        train={r['step']:r for r in logs[mode] if 'train/loss' in r and r['step']%10==0}
        steps=sorted(train); yy=np.array([train[s]['train/loss'] for s in steps])
        axes[0,0].plot(np.array(steps)[10:],np.convolve(yy,np.ones(11)/11,mode='valid'),label=mode)
        val=summary[mode]['validation']
        axes[0,1].plot([r['step'] for r in val],[r['val/loss'] for r in val],marker='o',label=mode)
        delta=(losses[mode][1.]-losses['baseline'][1.]).double().reshape(256,32,64).mean(-1)
        mean,se=delta.mean(0).numpy(),(delta.std(0)/16).numpy()
        xx=np.arange(32)*64+32.5
        line,=axes[1,0].plot(xx,mean,label=mode)
        axes[1,0].fill_between(xx,mean-1.96*se,mean+1.96*se,alpha=.12,color=line.get_color())
        axes[1,1].plot([0,.5,1],[summary[mode]['paired'][str(p)]['nll'] for p in (0.,.5,1.)],marker='o',label=mode)
    axes[0,0].set(title='Train loss: 11 logged-point trailing mean',xlabel='Update',ylabel='NLL',ylim=(3.7,6))
    axes[0,1].set(title='Original fixed-prefix validation',xlabel='Update',ylabel='NLL')
    axes[1,0].set(title='Paired pure-hard NLL minus baseline (95% bands)',xlabel='Target position',ylabel='Delta NLL')
    axes[1,0].axhline(0,color='black',linewidth=.7)
    axes[1,1].set(title='Final-checkpoint soft/hard intervention',xlabel='hard_prob',ylabel='NLL')
    for ax in axes.flat: ax.legend(fontsize=8)
    fig.savefig(a.output/'comparison.png',dpi=170)
    plt.close(fig)


if __name__=='__main__':main()
