"""Plot measured cotangent decomposition and write a compact diagnostic table."""
from pathlib import Path
import json,csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[2]/'dism_v3/build/selector_decomposition'
names=['baseline_L7_H1','baseline_L9_H0','ln_scheduled_L6_H0','ln_scheduled_L9_H0','baseline_L7_H4','ln_scheduled_L6_H1']
rows=[]
fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
for ax,key in zip(axes,['q','q_vocab']):
 original=[];projected=[]
 for name in names:
  d=json.loads((root/name/'results.json').read_text());v=d['vjp'][key]
  original.append(v['hard']['ratio']);projected.append(v['hard_soft_tangent']['ratio'])
  rows.append(dict(case=name,parameter=key,original_ratio=v['hard']['ratio'],projected_ratio=v['hard_soft_tangent']['ratio'],parallel_ratio=v['parallel']['ratio'],residual_ratio=v['residual']['ratio'],soft_projection_ratio=v['soft_soft_tangent']['ratio'],linearity_error=v['linearity_error']))
 x=list(range(len(names)))
 ax.bar([v-.18 for v in x],original,width=.36,label='Original surrogate')
 ax.bar([v+.18 for v in x],projected,width=.36,label='Project onto soft norm tangent')
 ax.axhline(1,color='gray',lw=1);ax.set_yscale('log');ax.set_xticks(x,names,rotation=45,ha='right');ax.set_title(key);ax.set_ylabel('Gradient norm / original soft gradient norm');ax.legend()
fig.savefig(root/'norm_projection.png',dpi=160)
with (root/'summary.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
