"""Summarize per-head checkpoint probes, preserving document-level variation."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[2]/'dism_v3/build/soft_hard_heads'
names=['baseline','ln_scheduled','ln_fixed05']
summary=[];headrows=[]
fig,axes=plt.subplots(3,3,figsize=(15,10),constrained_layout=True)
for row,name in enumerate(names):
 records=json.loads((root/name/'results.json').read_text())['records']
 for length in [2048,8192]:
  rec=[r for r in records if r['context']==length]
  loss={r['document']:(r['soft_loss'],r['hard_loss']) for r in rec}
  entry=dict(model=name,context=length,soft_loss=np.mean([v[0] for v in loss.values()]),hard_loss=np.mean([v[1] for v in loss.values()]))
  for kind in ['output','gradient']:
   for metric in ['cosine','relative_l2','hard_soft_norm_ratio','soft_rms','hard_rms']:
    v=np.array([r[kind][metric] for r in rec]);entry[kind+'_'+metric+'_median']=float(np.median(v));entry[kind+'_'+metric+'_mean']=float(np.mean(v))
  summary.append(entry)
  for layer in sorted({r['layer'] for r in rec}):
   lr=[r for r in rec if r['layer']==layer]
   for head in range(12):
    hr=dict(model=name,context=length,layer=layer,head=head)
    for kind in ['output','gradient']:
     for metric in ['cosine','relative_l2','hard_soft_norm_ratio','soft_rms','hard_rms']:
      hr[kind+'_'+metric]=float(np.mean([r[kind][metric][head] for r in lr]))
    headrows.append(hr)
  if length==8192:
   for col,(kind,metric) in enumerate([('output','cosine'),('gradient','cosine'),('output','hard_soft_norm_ratio')]):
    matrix=np.array([[np.mean([r[kind][metric][h] for r in rec if r['layer']==l]) for h in range(12)] for l in range(6,12)])
    if col==2:matrix=np.log10(np.maximum(matrix,1e-10))
    im=axes[row,col].imshow(matrix,aspect='auto',vmin=(-4 if col==2 else 0),vmax=(4 if col==2 else 1),cmap='coolwarm' if col==2 else 'viridis')
    axes[row,col].set_title(name+' | '+kind+' '+('log10 norm ratio' if col==2 else metric));axes[row,col].set_yticks(range(6),range(6,12));axes[row,col].set_xlabel('Head (0-based)');axes[row,col].set_ylabel('Layer (0-based)');fig.colorbar(im,ax=axes[row,col])
fig.savefig(root/'heads_8k.png',dpi=160)
(root/'summary.json').write_text(json.dumps(summary,indent=2))
with (root/'heads.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(headrows[0]));w.writeheader();w.writerows(headrows)
print(json.dumps(summary,indent=2))
