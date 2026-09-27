import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root=Path('/media/cicuvc/c63abdf1-0e56-4153-9228-95df5a2f239b/cicuvc/dism-lm-runs/gdn-dism-vocab-silu-384-72m-30k-20260911')
    out=root/'document-boundary-32-canonical'
    r=torch.load(out/'results.pt',weights_only=False)
    x=r['losses'].double();alone=r['alone'].double();assert x.shape==(32,2,3,5,512)
    torch.testing.assert_close(x[:,:,:,4],alone[:,None,None].expand(-1,2,3,-1),atol=0,rtol=0)
    def stats(delta):
        seq=delta.reshape(32,-1).mean(1);m=float(seq.mean());se=float(seq.std()/32**.5)
        return dict(mean=m,se=se,ci95=[m-1.96*se,m+1.96*se])
    report=dict(arms={},alone=float(alone.mean()),mass={},partition_check='both reset == B alone exactly, every token',
        caveats=['32 target documents, two prefixes and three lengths are repeated measures; SE clusters by target document.',
          'Document sampling excludes short documents (<1537 tokens), first512 B predictions only.',
          'Full-hard inference on existing checkpoint does not test whether boundary-aware training learns different circuits.',
          'Canonical split of tokenwise Linear/LayerNorm controls BF16 shape effects in all arms.',
          'Mass uses actual CUDA labels with exact hard recurrence, not tanh-approximate kernel weights; eight target documents only.'])
    for i,mode in enumerate(r['modes']):
        delta=x[:,:,:,i]-x[:,:,:,0]
        report['arms'][mode]=dict(nll=float(x[:,:,:,i].mean()),delta=stats(delta),
            segments={f'{a}:{b}':stats(delta[...,a:b]) for a,b in [(0,32),(32,128),(128,512)]},
            by_prefix={length:stats(delta[:,:,j]) for j,length in enumerate(r['lengths'])},
            prefix_swap_per_token_abs=float((x[:,0,:,i]-x[:,1,:,i]).abs().mean()))
    for layer in (4,8,12):
        a=np.stack([m['values'] for m in r['masses'] if m['layer']==layer])
        report['mass'][layer]=dict(cross_document=float(a[...,0].mean()),
            cross_document_chain_ge4=float(a[...,1].mean()),cross_label_match_rate=float(a[...,2].mean()),
            first32=float(a[:,:,:32,0].mean()),last256=float(a[:,:,256:,0].mean()),
            per_head=a[...,0].mean((0,2)).tolist())
    old=torch.load(root/'document-boundary-32/results.pt',weights_only=False)['losses'].double()
    report['canonical_vs_original_intact_abs']=float((x[:,:,:,0]-old[:,:,:,0]).abs().mean())
    report['canonical_vs_original_intact_mean']=float((x[:,:,:,0]-old[:,:,:,0]).mean())
    (out/'summary.json').write_text(json.dumps(report,indent=2))
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    bins=np.arange(16)*32+15.5
    for i in range(1,5):
        seq=(x[:,:,:,i]-x[:,:,:,0]).mean((1,2)).reshape(32,16,32).mean(2)
        mean=seq.mean(0).numpy();se=(seq.std(0)/32**.5).numpy()
        axes[0].plot(bins,mean,label=r['modes'][i]);axes[0].fill_between(bins,mean-1.96*se,mean+1.96*se,alpha=.12)
    axes[0].axhline(0,color='black',lw=.7);axes[0].set(xlabel='Position in B (predicting next token)',ylabel='NLL change vs intact',title='Boundary intervention (32 documents)')
    for layer in (4,8,12):
        a=np.stack([m['values'] for m in r['masses'] if m['layer']==layer])
        axes[1].plot(bins,a[...,0].mean((0,1)).reshape(16,32).mean(1),label=f'Layer {layer}')
    axes[1].set(xlabel='Position in B',ylabel='Attention probability on previous document',title='Hard reference weights, prefix1536 (8 docs)',ylim=(0,1))
    for ax in axes:ax.legend();ax.grid(alpha=.2)
    fig.savefig(out/'boundary.png',dpi=180)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
