"""CPU-only paired analysis of A100 SWA results and frozen local hybrid results."""
import argparse
import json
from pathlib import Path
import torch


def summarize(values):
    v=values.double().mean(-1)
    return dict(mean=v.mean().item(),se=v.std().item()/len(v)**.5)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hybrid-run',type=Path,required=True)
    p.add_argument('--swa-results',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('New output directory required')
    hreport=json.loads((a.hybrid_run/'length-prefill-8192/report.json').read_text())
    sreport=json.loads((a.swa_results/'report.json').read_text())
    assert hreport['step']==sreport['step']==30000
    assert hreport['packed_xy_sha256']==sreport['paired_xy_sha256']
    h=torch.load(a.hybrid_run/'length-prefill-8192/per_token.pt',weights_only=True)['losses']
    s=torch.load(a.swa_results/'position_losses.pt',weights_only=True)
    hn=json.loads((a.hybrid_run/'niah-256.json').read_text())['rows']
    sn=sreport['niah']
    hid={(r['trial'],r['condition'],r['length'],r['depth']):r for r in hn}
    sid={(r['trial'],r['condition'],r['length'],r['depth']):r for r in sn}
    assert hid.keys()==sid.keys()
    for key,r in hid.items():
        if r['condition']=='needle':assert r['prompt_sha256']==sid[key]['prompt_sha256']
    summary=dict(position={},niah={},source_swa=str(a.swa_results),source_hybrid=str(a.hybrid_run),
                 hardware_note='Hybrid on RTX5090; SWA on A100. Same BF16 head / FP32 softcap CE formula; not bitwise hardware controlled.')
    for n in h:
        assert h[n].shape==s[n].shape
        summary['position'][n]=dict(hybrid=summarize(h[n]),swa=summarize(s[n]),
            hybrid_minus_swa=summarize(h[n]-s[n]),
            first_block_hybrid=summarize(h[n][:,:n]),first_block_swa=summarize(s[n][:,:n]))
    for name,rows in [('hybrid',hn),('swa',sn)]:
        def score(sub):
            return dict(samples=len(sub),exact=sum(r['exact_match'] for r in sub),
                        candidate_correct=sum(r['candidate_correct'] for r in sub),
                        mean_nll=sum(r['target_nll'] for r in sub)/len(sub))
        result={c:score([r for r in rows if r['condition']==c]) for c in ('near','no_information','needle')}
        for n in (2048,8192):
            selected=[r for r in rows if r['condition']=='needle' and r['length']==n]
            result[n]=score(selected)
            result[n]['by_depth']={d:score([r for r in selected if r['depth']==d]) for d in (.1,.35,.65,.9)}
        far=[r for r in rows if r['condition']=='needle' and r['tokens_after_needle']>1905]
        result['beyond_swa_receptive_field']=score(far)
        shifts=[];max_changes=[]
        for r in far:
            target=r['target'] if isinstance(r['target'],int) else sreport['candidate_colors'].index(r['target'])
            alt=r['alternative'] if isinstance(r['alternative'],int) else sreport['candidate_colors'].index(r['alternative'])
            x=torch.tensor(r['candidate_logits']);y=torch.tensor(r['counterfactual']['candidate_logits'])
            shifts.append(float((x[target]-x[alt])-(y[target]-y[alt])))
            max_changes.append(float((x-y).abs().max()))
        result['beyond_swa_receptive_field']['counterfactual_mean_max_color_logit_change']=sum(max_changes)/len(max_changes)
        result['beyond_swa_receptive_field']['positive_binding_shifts']=sum(x>0 for x in shifts)
        summary['niah'][name]=result
    a.output.mkdir(parents=True)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(3,1,figsize=(11,11),constrained_layout=True)
    def curve(ax,values,label):
        bins=values.double().reshape(len(values),-1,64).mean(-1)
        mean=bins.mean(0).numpy();error=(1.96*bins.std(0)/len(values)**.5).numpy()
        pos=np.arange(len(mean))*64+32.5
        line,=ax.plot(pos,mean,label=label)
        ax.fill_between(pos,mean-error,mean+error,color=line.get_color(),alpha=.15)
    for index,n in enumerate((2048,8192)):
        curve(axes[index],h[n][:,:n], 'Hybrid')
        curve(axes[index],s[n][:,:n], 'SWA-only')
        axes[index].set(title=f'Prefill {n}: same-prefix NLL (64-position bins)',ylabel='NLL (nats)',xlabel='Target position')
    curve(axes[2],h[8192]-s[8192],'Hybrid minus SWA, full8192')
    axes[2].axhline(0,color='black',linewidth=.7)
    axes[2].set(title='Paired difference: positive means SWA has lower NLL',ylabel='Delta NLL',xlabel='Target position')
    for ax in axes:
        ax.legend();ax.axvline(2048,color='gray',linestyle='--');ax.spines[['top','right']].set_visible(False)
    fig.savefig(a.output/'paired_positions.png',dpi=180);fig.savefig(a.output/'paired_positions.svg');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
    depths=(.1,.35,.65,.9)
    for ax,n in zip(axes,(2048,8192)):
        for index,(name,color) in enumerate((('hybrid','tab:blue'),('swa','tab:orange'))):
            cells=summary['niah'][name][n]['by_depth']
            v=[cells[d]['exact']/cells[d]['samples'] for d in depths]
            ax.bar(np.arange(4)+(index-.5)*.35,v,width=.35,label=name,color=color)
        ax.set(xticks=range(4),xticklabels=['10%','35%','65%','90%'],ylim=(0,1),
               xlabel='Needle insertion fraction',ylabel='Exact match',title=f'Context {n},32 cases per cell')
        ax.legend();ax.spines[['top','right']].set_visible(False)
    fig.savefig(a.output/'paired_niah.png',dpi=180);plt.close(fig)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
