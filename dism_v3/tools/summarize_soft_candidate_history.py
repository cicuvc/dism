"""Temporal fixed-panel agreement, intervention utility and soft-gradient scale."""
import json,csv,itertools
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[2]/'dism_v3/build/soft_candidate_history'
summary=[];rows=[];threshold_rows=[]
for step in [12000,14000,22000,22888]:
 path=root/str(step)/'results.json'
 if not path.exists():continue
 data=json.loads(path.read_text());records=data['records']
 for n in [2048,8192]:
  rec=[r for r in records if r['context']==n]
  if not rec:continue
  for mode in ['all_soft','local_soft']:
   for cutoff in [0., .01, .1]:
    comparisons=[]
    for r in rec:
     key='all_soft' if mode=='all_soft' else f"local_soft_{r['layer']}"
     panel=[c for c in r['candidates'] if c['in_fixed_panel']]
     for a,b in itertools.combinations(panel,2):
      delta=a['delta_sum']-b['delta_sum'];gradient=a[key]['prob_direction']-b[key]['prob_direction']
      if abs(delta)>cutoff and gradient!=0:comparisons.append(delta*gradient>0)
    threshold_rows.append(dict(step=step,context=n,mode=mode,cutoff=cutoff,pairs=len(comparisons),accuracy=sum(comparisons)/len(comparisons) if comparisons else None))
   for metric in ['prob_direction','logit_direction']:
    agree=disagree=signok=signbad=0;chosen=[];oracle=[];random=[];gnorm=[];errors=[]
    for r in rec:
     key='all_soft' if mode=='all_soft' else f"local_soft_{r['layer']}"
     cs=r['candidates'];panel=[c for c in cs if c['in_fixed_panel']]
     for a,b in itertools.combinations(panel,2):
      product=(a['delta_sum']-b['delta_sum'])*(a[key][metric]-b[key][metric]);agree+=product>0;disagree+=product<0
     for c in cs:
      product=c['delta_sum']*c[key][metric];signok+=product>0;signbad+=product<0
      rows.append(dict(step=step,context=n,document=r['document'],layer=r['layer'],head=r['head'],side=r['side'],position=r['position'],symbol=c['symbol'],in_fixed_panel=c['in_fixed_panel'],mode=mode,metric=metric,hard_delta_sum=c['delta_sum'],soft_direction=c[key][metric],soft_probability=c[key]['probability']))
     chosen.append(min(cs,key=lambda c:c[key][metric])['delta_sum'])
     oracle.append(min(c['delta_sum'] for c in cs));random.append(np.mean([c['delta_sum'] for c in cs]))
     gnorm.append(cs[0][key]['logit_gradient_norm']);errors.append(cs[0][key]['oracle_output_relative_error'])
    summary.append(dict(step=step,hard_probability=min(.95,.95*step/22887),context=n,mode=mode,metric=metric,nodes=len(rec),pair_agree=int(agree),pair_disagree=int(disagree),pair_accuracy=agree/(agree+disagree) if agree+disagree else None,sign_accuracy=signok/(signok+signbad) if signok+signbad else None,
      selected_improves=sum(v<0 for v in chosen),selected_worsens=sum(v>0 for v in chosen),selected_unchanged=sum(v==0 for v in chosen),mean_selected_delta_sum=float(np.mean(chosen)),mean_oracle_delta_sum=float(np.mean(oracle)),mean_uniform_delta_sum=float(np.mean(random)),median_soft_logit_gradnorm=float(np.median(gnorm)),max_oracle_output_relative_error=max(errors)))
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(root/'pair_threshold_sensitivity.json').write_text(json.dumps(threshold_rows,indent=2)+'\n')
with (root/'candidate_scores.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
for x in summary:
 if x['metric']=='prob_direction':print(x)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
for col,n in enumerate([2048,8192]):
 for mode,label in [('all_soft','All soft'),('local_soft','Target layer soft')]:
  selected=[r for r in summary if r['context']==n and r['mode']==mode and r['metric']=='prob_direction']
  steps=[r['step'] for r in selected]
  axes[0,col].plot(steps,[r['pair_accuracy'] for r in selected],marker='o',label=label)
  axes[1,col].plot(steps,[r['mean_selected_delta_sum'] for r in selected],marker='o',label=label)
 for field,label in [('mean_uniform_delta_sum','Uniform incl. no-op'),('mean_oracle_delta_sum','Hindsight best')]:
  axes[1,col].plot(steps,[r[field] for r in selected],marker='.',linestyle='--',label=label)
 axes[0,col].axhline(.5,color='gray',linestyle=':');axes[0,col].set_ylim(.2,.8)
 axes[0,col].set_title(f'{n} tokens ({32 if n==2048 else 16} nodes)')
 axes[0,col].set_ylabel('Candidate pair agreement');axes[0,col].legend(fontsize=8)
 axes[1,col].axhline(0,color='gray',linestyle=':');axes[1,col].set_ylabel('Selected hard delta summed NLL')
 axes[1,col].set_xlabel('Training step');axes[1,col].legend(fontsize=8)
 for ax in axes[:,col]:
  ax.set_xticks(steps,['12K','14K','22K','Final']);ax.grid(alpha=.2)
fig.suptitle('LN + sigmoid scheduled run: fixed final-checkpoint candidate panel\nSmall correlated sample; negative NLL change is better')
fig.savefig(root/'history.png',dpi=160)
plt.close(fig)

alignment=[]
for step in [12000,14000,22000,22888]:
 path=root/str(step)/'output_alignment.json'
 if not path.exists():continue
 records=json.loads(path.read_text())['records']
 for n in [2048,8192]:
  selected=[r for r in records if r['context']==n]
  for kind in ['end_to_end','same_soft_inputs','same_hard_inputs']:
   alignment.append(dict(step=step,context=n,kind=kind,
    median_head_cosine=float(np.median([v for r in selected for v in r[kind]['cosine']])),
    mean_layer_pooled_cosine=float(np.mean([r[kind]['pooled_cosine'] for r in selected])),
    max_probe_loss_difference=max(abs(r[f'{mode}_loss']-r[f'plain_{mode}_loss']) for r in selected for mode in ['hard','soft'])))
if alignment:
 (root/'output_alignment_summary.json').write_text(json.dumps(alignment,indent=2)+'\n')
 fig,axes=plt.subplots(1,2,figsize=(9,3.5),layout='constrained')
 for ax,n in zip(axes,[2048,8192]):
  for kind,label in [('end_to_end','End to end'),('same_soft_inputs','Same soft inputs'),('same_hard_inputs','Same hard inputs')]:
   selected=[r for r in alignment if r['context']==n and r['kind']==kind]
   ax.plot([r['step'] for r in selected],[r['median_head_cosine'] for r in selected],marker='o',label=label)
  ax.set_title(f'{n} tokens');ax.set_xlabel('Training step');ax.set_ylabel('Median raw DISM head cosine')
  ax.set_xticks([12000,14000,22000,22888],['12K','14K','22K','Final']);ax.legend(fontsize=8);ax.grid(alpha=.2)
 fig.savefig(root/'output_alignment.png',dpi=160)
 plt.close(fig)
