"""Report direction, scale and energy of selective-STE gradient disagreements."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[2]/'dism_v3/build/selector_surrogate'
names=['baseline','ln_scheduled','ln_fixed05'];keys=['q','k','q_vocab','k_vocab']
summary=[];rows=[]
fig,axes=plt.subplots(3,4,figsize=(17,10),constrained_layout=True)
for ni,name in enumerate(names):
 records=json.loads((root/name/'results.json').read_text())['records']
 for n in [2048,8192]:
  rec=[r for r in records if r['context']==n]
  for ki,key in enumerate(keys):
   cos=np.array([r['selectors'][key]['cosine'] for r in rec]);s=np.array([r['selectors'][key]['soft_rms'] for r in rec]);h=np.array([r['selectors'][key]['hard_rms'] for r in rec])
   active=(s>0)&(h>0);neg=(cos<0)&active;low=(cos<.5)&active
   entry=dict(model=name,context=n,parameter=key,cells=cos.size,zero_cells=int((~active).sum()),
      cosine_median=float(np.median(cos[active])),cosine_p10=float(np.quantile(cos[active],.1)),
      pooled_cosine=float((cos*s*h).sum()/np.sqrt((s*s).sum()*(h*h).sum())),
      negative_count=int(neg.sum()),below_half_count=int(low.sum()),
      negative_soft_energy_fraction=float((s[neg]**2).sum()/max((s*s).sum(),1e-300)),
      negative_surrogate_energy_fraction=float((h[neg]**2).sum()/max((h*h).sum(),1e-300)),
      below_half_soft_energy_fraction=float((s[low]**2).sum()/max((s*s).sum(),1e-300)),
      rms_ratio_pooled=float(np.sqrt((h*h).sum()/max((s*s).sum(),1e-300))),
      rms_ratio_p90=float(np.quantile((h/s)[active],.9)),
      soft_rms_median=float(np.median(s)),surrogate_rms_median=float(np.median(h)))
   summary.append(entry)
   if n==8192:
    matrix=np.array([np.mean([r['selectors'][key]['cosine'] for r in rec if r['layer']==layer],0) for layer in range(6,12)])
    im=axes[ni,ki].imshow(matrix,vmin=-1,vmax=1,cmap='coolwarm',aspect='auto');axes[ni,ki].set_title(name+' / '+key);axes[ni,ki].set_yticks(range(6),range(6,12));axes[ni,ki].set_xlabel('Head (0-based)');axes[ni,ki].set_ylabel('Layer (0-based)');fig.colorbar(im,ax=axes[ni,ki])
   for r in rec:
    for head in range(12):
     rows.append(dict(model=name,context=n,document=r['document'],layer=r['layer'],head=head,parameter=key,
         upstream_cosine=r['upstream']['cosine'][head],**{k:v[head] for k,v in r['selectors'][key].items()}))
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
with (root/'heads.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
fig.savefig(root/'selector_cosine_8k.png',dpi=150)
for r in summary:print(r)

fig,axes=plt.subplots(3,4,figsize=(17,10),constrained_layout=True)
for ni,name in enumerate(names):
 for ki,key in enumerate(keys):
  points=[r for r in rows if r['model']==name and r['context']==8192 and r['parameter']==key]
  ax=axes[ni,ki]
  ax.scatter([r['cosine'] for r in points],[max(r['hard_soft_norm_ratio'],1e-8) for r in points],c=[r['layer'] for r in points],s=12,alpha=.6,cmap='viridis')
  ax.set_yscale('log');ax.axhline(1,color='gray',lw=.8);ax.axvline(0,color='gray',lw=.8)
  ax.set_xlim(-1.05,1.05);ax.set_title(name+' / '+key);ax.set_xlabel('Gradient cosine');ax.set_ylabel('Surrogate / soft norm')
fig.savefig(root/'selector_direction_scale_8k.png',dpi=150)
