"""Summarize reproducible hard-label counterfactual gains and confidence rankings."""
from pathlib import Path
import csv,json,itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[2]/'dism_v3/build/hard_symbol_candidates'
summary=[];flat=[]
fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
for ax,name in zip(axes,['baseline','ln_scheduled','ln_fixed05']):
 data=json.loads((root/name/'results.json').read_text());r=data['records']
 assert (root/name/'complete').exists() and len(r)==48
 assert 'all other hard labels frozen' in data['protocol']
 def aggregate(rec):
  candidates=[c for x in rec for c in x['candidates']]
  best=[min(x['candidates'],key=lambda c:c['delta_nll_sum']) for x in rec]
  changes=np.array([c['delta_nll_sum'] for c in candidates]);b=np.array([c['delta_nll_sum'] for c in best])
  good=[c for c in best if c['delta_nll_sum']<0]
  concordant=discordant=0;ranked_nodes=0;top_hits=0
  for x in rec:
   cs=x['candidates']
   if max(c['delta_nll_sum'] for c in cs)>min(c['delta_nll_sum'] for c in cs):
    ranked_nodes+=1
    top_hits+=max(cs,key=lambda c:c['probability'])['delta_nll_sum']==min(c['delta_nll_sum'] for c in cs)
   for a,b in itertools.combinations(cs,2):
    relation=(a['probability']-b['probability'])*(a['delta_nll_sum']-b['delta_nll_sum'])
    concordant+=relation<0;discordant+=relation>0
  return dict(nodes=len(rec),candidates=len(candidates),beneficial_candidates=int((changes<0).sum()),harmful_candidates=int((changes>0).sum()),zero_candidates=int((changes==0).sum()),
    nodes_with_better_candidate=len(good),nodes_with_gain_gt_01=sum(c['delta_nll_sum']<-.1 for c in best),
    best_gain_median_given_benefit=float(np.median([-c['delta_nll_sum'] for c in good])) if good else None,
    max_gain=float(max(0,-changes.min())),max_damage=float(max(0,changes.max())),
    best_benefit_adds_matches=sum(c['new_direct_matches']>c['old_direct_matches'] for c in good),
    best_benefit_removes_matches=sum(c['new_direct_matches']<c['old_direct_matches'] for c in good),
    best_benefit_no_matches=sum(c['new_direct_matches']==0 for c in good),
    confidence_concordant_pairs=int(concordant),confidence_discordant_pairs=int(discordant),
    confidence_pair_accuracy=float(concordant/(concordant+discordant)) if concordant+discordant else None,
    confidence_top_hits=int(top_hits),ranked_nodes=ranked_nodes)
 item=dict(model=name,all=aggregate(r),by_context={str(n):aggregate([x for x in r if x['context']==n]) for n in [2048,8192]},
           by_selection={s:aggregate([x for x in r if x['selection']==s]) for s in ['fixed','small_margin']})
 summary.append(item)
 for x in r:
  for c in x['candidates']:
   flat.append(dict(model=name,context=x['context'],document=x['document'],layer=x['layer'],head=x['head'],side=x['side'],position=x['position'],selection=x['selection'],original_symbol=x['original_symbol'],candidate=c['symbol'],probability=c['probability'],original_probability=x['original_probability'],delta_nll_sum=c['delta_nll_sum'],delta_nll=c['delta_nll'],old_matches=c['old_direct_matches'],new_matches=c['new_direct_matches'],raw_rms_change=c['raw_output_change']['rms'],changed_loss_tokens=c['changed_loss_tokens']))
 for side,color in [('q','tab:blue'),('k','tab:orange')]:
  points=[(i,c['delta_nll_sum']) for i,x in enumerate(r) if x['side']==side for c in x['candidates']]
  ax.scatter([v[0] for v in points],[v[1] for v in points],s=16,alpha=.65,label=side,color=color)
 ax.axhline(0,color='gray');ax.set_title(name);ax.set_xlabel('Node index');ax.set_ylabel('Change in total target NLL (lower is better)');ax.legend()
fig.savefig(root/'candidate_effects.png',dpi=160)
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
with (root/'candidates.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
print(json.dumps(summary,indent=2))
