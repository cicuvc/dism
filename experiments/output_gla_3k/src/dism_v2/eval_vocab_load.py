"""Dataset-wide hard Q/K vocabulary occupancy from actual CUDA embedding labels."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time

import torch


def load_metrics(counts):
    counts = counts.double()
    vocab = counts.numel()
    total = counts.sum()
    if total <= 0:
        raise ValueError('Empty label histogram')
    p = counts/total
    ranked = p.sort(descending=True).values
    sorted_counts = counts.sort().values
    gini = (2 * (torch.arange(1, vocab+1)*sorted_counts).sum() / (vocab*total) - (vocab+1)/vocab).item()
    return dict(tokens=int(total), unused=int((counts==0).sum()),
        at_most_10=int((counts<=10).sum()),
        below_1pct_uniform=int((counts<total/vocab*.01).sum()),
        below_10pct_uniform=int((counts<total/vocab*.1).sum()),
        effective_vocab=torch.exp(-(p*p.clamp_min(1e-300).log()).sum()).item(),
        top1_mass=ranked[0].item(), top8_mass=ranked[:8].sum().item(),
        labels_for_90pct=int(torch.searchsorted(ranked.cumsum(0), .9))+1,
        gini=gini, max_to_uniform=(p.max()*vocab).item())


def alignment(q, k):
    p, r = q.double()/q.sum(), k.double()/k.sum()
    m = (p+r)/2
    js = .5*((p*(p.clamp_min(1e-300)/m.clamp_min(1e-300)).log()).sum()+
             (r*(r.clamp_min(1e-300)/m.clamp_min(1e-300)).log()).sum())
    return dict(overlap=torch.minimum(p,r).sum().item(), js_nats=js.item(),
                independent_label_match_probability=(p*r).sum().item(),
                q_mass_on_unused_k=p[k==0].sum().item(),
                q_mass_on_rare_k=p[k<k.sum()/k.numel()*.01].sum().item(),
                k_mass_on_unused_q=r[q==0].sum().item())


def plot(output, hist):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    p = hist.double()/hist.sum(-1, keepdim=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 9), constrained_layout=True)
    for s, ax in enumerate(axes):
        arr=p[:,s].reshape(-1,p.shape[-1])
        im=ax.imshow(torch.log10(arr.clamp_min(1e-7)), aspect='auto', vmin=-7, vmax=0,
                     cmap='magma', interpolation='nearest')
        ax.set(title=('Q' if s==0 else 'K')+' label frequency (same label IDs)', xlabel='Vocabulary ID', ylabel='Layer (4 head rows each)')
        ax.set_yticks(np.arange(hist.shape[0])*hist.shape[2]+1.5,range(1,hist.shape[0]+1))
    fig.colorbar(im,ax=axes,label='log10 empirical frequency; zero clipped to1e-7',shrink=.8)
    fig.savefig(output/'label_load.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),constrained_layout=True)
    for s, name in enumerate(('Q','K')):
        ranked=p[:,s].reshape(-1,p.shape[-1]).sort(descending=True).values.numpy()
        median=np.median(ranked,0)
        axes[0].plot(np.arange(1,ranked.shape[-1]+1),median,label=name)
        axes[0].fill_between(np.arange(1,ranked.shape[-1]+1),*np.quantile(ranked,[.1,.9],axis=0),alpha=.15)
        cumulative=ranked.cumsum(-1)
        axes[1].plot(np.arange(1,ranked.shape[-1]+1),np.median(cumulative,0),label=name)
    axes[0].axhline(1/p.shape[-1],color='gray',linestyle='--',label='Uniform')
    axes[0].set(xscale='log',yscale='log',ylim=(1e-8,1),xlabel='Within-head frequency rank',ylabel='Frequency',title='Median and10–90% range across heads')
    axes[1].set(xscale='log',xlabel='Most-used labels',ylabel='Cumulative frequency',title='Median cumulative load across heads')
    for ax in axes:ax.legend()
    fig.savefig(output/'ranked_load.png',dpi=170);plt.close(fig)


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--sequences',type=int,default=6400)
    p.add_argument('--micro-batch',type=int,default=8)
    a=p.parse_args()
    if a.output.exists() or a.micro_batch<=0 or a.sequences<=0 or a.sequences%a.micro_batch:
        p.error('New output directory; positive sequence count divisible by micro-batch required')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    cfg,step=saved['config'],saved['step']
    os.environ['DISM_TILE_LSE']=cfg['tile_lse']
    os.environ['DISM_BWD_OPT']=str(cfg['backward_opt'])
    os.environ['TOKENIZERS_PARALLELISM']='false'
    from .lm_model import DecoderLM, LMConfig
    from .train_lm import load_tokenizer
    from .lm_data import PackedStream,split_files
    from . import embedding
    c=LMConfig(**cfg['model'])
    model=DecoderLM(c).cuda().eval()
    model.load_state_dict(saved['model'],strict=True)
    del saved
    tokenizer=load_tokenizer(cfg['tokenizer'])
    _,files=split_files(cfg['data'])
    stream=PackedStream(files,tokenizer,context=c.context,batch_size=a.micro_batch,repeat=False)
    hist=torch.zeros(c.layers,2,c.heads,c.qk_vocab,device='cuda',dtype=torch.int64)
    offset=torch.arange(c.heads,device='cuda')[None,:,None]*c.qk_vocab
    original=embedding.forward
    call=0
    def traced(*args,**kwargs):
        nonlocal call
        raw=original(*args,**kwargs)
        layer=call%c.layers
        for side,labels in enumerate((raw[7],raw[6])):
            ids=(labels.long()+offset).flatten()
            hist[layer,side].add_(torch.bincount(ids,minlength=c.heads*c.qk_vocab).reshape(c.heads,c.qk_vocab))
        call+=1
        return raw
    digest=hashlib.sha256()
    a.output.mkdir(parents=True)
    start=time.perf_counter()
    try:
        embedding.forward=traced
        for batch in range(a.sequences//a.micro_batch):
            x,_=stream.next_batch()
            digest.update(x.numpy().tobytes())
            gx=x.cuda()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                h=model.forward_features(gx,1.,torch.Generator(device='cuda').manual_seed(779))
            if not torch.isfinite(h).all():raise FloatingPointError('Nonfinite model features')
            if batch==0:
                embedding.forward=original
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    control=model.forward_features(gx,1.,torch.Generator(device='cuda').manual_seed(779))
                torch.testing.assert_close(h,control,atol=0,rtol=0)
                embedding.forward=traced
            if (batch+1)%100==0:
                print(json.dumps(dict(sequences=(batch+1)*a.micro_batch,total=a.sequences,seconds=time.perf_counter()-start)),flush=True)
    finally:
        embedding.forward=original
    hist=hist.cpu()
    assert call==c.layers*(a.sequences//a.micro_batch)
    assert (hist.sum(-1)==a.sequences*c.context).all()
    rows=[]
    for l in range(c.layers):
        for h in range(c.heads):
            rows.append(dict(layer=l+1,head=h+1,q=load_metrics(hist[l,0,h]),k=load_metrics(hist[l,1,h]),
                             alignment=alignment(hist[l,0,h],hist[l,1,h])))
    report=dict(checkpoint=str(a.checkpoint),step=step,hard_prob=1.,sequences=a.sequences,
                tokens_per_head_per_side=a.sequences*c.context,vocab=c.qk_vocab,
                layers=c.layers,heads=c.heads,context=c.context,seconds=time.perf_counter()-start,
                first_batch_tracing_bitwise_equal=True,packed_input_sha256=digest.hexdigest(),
                validation_files=files,rows=rows,
                caveats=['Observed hard top1 assignment, not soft probability mass or gradient usage.',
                         'Unused means zero in this validation sample, not globally dead.',
                         'Q/K marginal overlap is not causal per-query match coverage.',
                         'No uniform-use assumption is imposed on the model.'])
    torch.save(hist,a.output/'counts.pt')
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    with (a.output/'counts.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['layer','head','side','label','count','frequency'])
        for l in range(c.layers):
            for h in range(c.heads):
                for side in range(2):
                    for label,count in enumerate(hist[l,side,h].tolist()):
                        writer.writerow([l+1,h+1,('q','k')[side],label,count,count/(a.sequences*c.context)])
    plot(a.output,hist)
    print(json.dumps(dict(output=str(a.output),tokens_per_head=a.sequences*c.context,seconds=report['seconds'])),flush=True)


if __name__=='__main__':main()
