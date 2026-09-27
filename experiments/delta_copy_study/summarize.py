"""Summarize paired gate studies, retaining all seeds and position curves."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
summary={}
for name in ('results-v2','results-copy-bias0','results-abba-bias0'):
    arms={}
    for method in ('temperature','random','shared'):
        seeds=[]
        for seed in (0,1):
            path=root/name/f'{method}-seed{seed}'/'evaluation.jsonl'
            if not path.exists():continue
            snapshots=[json.loads(x) for x in path.read_text().splitlines()]
            if snapshots[-1]['step']!=1000:continue
            rows=snapshots[-1]['values'];hard,soft,unit,native,mixed_hard=rows
            seeds.append(dict(seed=seed,hard_loss=hard['loss'],hard_accuracy=hard['accuracy'],
                hard_after8_accuracy=hard['after8_accuracy'],keep_fraction=hard['keep_fraction'],
                delta_only_gap=hard['loss']-soft['loss'],unit_soft_gap=hard['loss']-unit['loss'],
                total_gap=hard['loss']-native['loss'],
                early_delta_gaps={s['step']:s['values'][0]['loss']-s['values'][1]['loss'] for s in snapshots},
                segments={k:hard[k] for k in ('first_segment_accuracy','second_segment_accuracy',
                         'first_segment_after8','second_segment_after8') if k in hard},
                positions={k:hard[k] for k in ('position_loss','position_accuracy','position_keep') if k in hard}))
        if len(seeds)==2:
            arms[method]=dict(seeds=seeds,mean={k:float(np.mean([s[k] for s in seeds])) for k in
                ('hard_loss','hard_accuracy','hard_after8_accuracy','keep_fraction','delta_only_gap','unit_soft_gap','total_gap')})
    summary[name]=arms
(root/'summary.json').write_text(json.dumps(summary,indent=2))
arms=summary['results-abba-bias0']
if len(arms)==3:
    fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True,layout='constrained')
    for method,arm in arms.items():
        acc=np.array([s['positions']['position_accuracy'] for s in arm['seeds']])
        keep=np.array([s['positions']['position_keep'] for s in arm['seeds']])
        axes[0].plot(range(64),acc.mean(0),label=method)
        axes[1].plot(range(64),keep.mean(0),label=method)
    axes[0].set(ylabel='Hard inference accuracy',title='ABBA: predict final B (0-31), A (32-63); mean of 2 seeds')
    axes[1].set(ylabel='Gate keep fraction',xlabel='Target position in supervised BA')
    for ax in axes:ax.axvline(31.5,color='black',ls='--');ax.grid(alpha=.2);ax.legend()
    fig.savefig(root/'abba-positions.png',dpi=180)
for name,arms in summary.items():
    print(name)
    for method,arm in arms.items():print(method,arm['mean'])
