"""Summarize saved generation and paired synthetic NIAH, no model execution."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root=Path('/media/cicuvc/c63abdf1-0e56-4153-9228-95df5a2f239b/cicuvc/dism-lm-runs')
    run=root/'gdn-dism-vocab-silu-384-72m-30k-20260911'
    raw=json.loads((run/'niah-256-final.json').read_text())
    assert raw['step']==30000 and raw['model']['gdn_dism']
    initial=json.loads((run/'niah-256.json').read_text())
    assert initial['rows']==raw['rows'], 'Deterministic NIAH replay differs'
    summary=dict(lengths={},controls={},deterministic_replay=True)
    def stats(rows):
        n=len(rows)
        return dict(n=n,exact=sum(r['exact_match'] for r in rows),
            candidate=sum(r['candidate_correct'] for r in rows),
            nll=sum(r['target_nll'] for r in rows)/n)
    colors=raw['candidate_colors']
    historical=root/'dism-swa50m-softcap30-20260909-offline/niah-256.json'
    if historical.exists():
        old=json.loads(historical.read_text())
        before=[r['prompt_sha256'] for r in old['rows'] if r['condition']=='needle']
        after=[r['prompt_sha256'] for r in raw['rows'] if r['condition']=='needle']
        summary['same_prompts_as_original_50m']=before==after
    for length in (2048,8192):
        rows=[r for r in raw['rows'] if r['condition']=='needle' and r['length']==length]
        shifts=[]
        for r in rows:
            a,b=colors.index(r['target']),colors.index(r['alternative'])
            x,y=r['candidate_logits'],r['counterfactual']['candidate_logits']
            shifts.append((x[a]-x[b])-(y[a]-y[b]))
        summary['lengths'][length]=dict(main=stats(rows),
            absent=stats([r['absent'] for r in rows]),
            changed=stats([r['counterfactual'] for r in rows]),
            gain_over_absent=sum(r['nll_gain_over_absent'] for r in rows)/len(rows),
            positive_margin_shifts=sum(s>0 for s in shifts),
            depths={d:stats([r for r in rows if r['depth']==d]) for d in (.1,.35,.65,.9)})
    for condition in ('near','no_information'):
        summary['controls'][condition]=stats([r for r in raw['rows'] if r['condition']==condition])
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    for ax,metric,title in zip(axes,['exact','candidate'],['Full-vocabulary top-1','Eight-color top-1']):
        for length in (2048,8192):
            d=summary['lengths'][length]['depths']
            ax.plot([10,35,65,90],[100*s[metric]/s['n'] for s in d.values()],marker='o',label=f'N={length}')
        if metric=='candidate':ax.axhline(12.5,ls='--',color='gray',label='Uniform chance (12.5%)')
        ax.set(title=title,xlabel='Needle insertion depth (%)',ylabel='Accuracy (%)',ylim=(0,100))
        ax.grid(alpha=.2);ax.legend()
    fig.suptitle('72M GDN/DISM, 30k steps: 32 cases per length/depth')
    fig.savefig(run/'niah-depth.png',dpi=180)
    generation=json.loads((run/'generation-t08-p09.json').read_text())
    summary['generation']=dict(prompts=len(generation['samples']),
        total_new_tokens=sum(r['new_tokens'] for r in generation['samples']),
        eos=sum(r['eos'] for r in generation['samples']),temperature=generation['temperature'],
        top_p=generation['top_p'],backend=generation['backend'])
    (run/'inference-summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
