"""Summarize single-query collision interventions with trial-cluster bootstrap."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from niah_circuit_probe import atomic_json
from summarize_niah_circuit import estimate


def summarize(directories, require_complete=True, row_summary=None, expected_trials=None):
    expected_trials = list(range(16)) if expected_trials is None else expected_trials
    output=dict(sign='benefit = baseline NLL minus intervention NLL; positive is improvement',
                units='five digit queries averaged per trial; paired trial bootstrap; pointwise intervals',arms={})
    hashes=set()
    for arm in ('old','restored','gqa2'):
        samples=[];histograms={};protocol=None
        for directory in directories:
            path=directory/f'{arm}.json'
            if not path.exists():
                if require_complete:raise FileNotFoundError(path)
                continue
            d=json.loads(path.read_text());hashes.add(d['suite_sha256'])
            if require_complete:assert d['buffers_unchanged']
            current=(d['halo'],d['recent'],d['paths'],d['modes'])
            if protocol is not None:assert current==protocol
            protocol=current
            samples+=d['samples']
            for key,h in d['histograms'].items():
                total=histograms.setdefault(key,{'q':np.zeros(512),'k':np.zeros(512)})
                for side,counts in h.items():total[side]+=counts
        if not samples:continue
        assert len({s['trial'] for s in samples})==len(samples)
        if require_complete:assert sorted(s['trial'] for s in samples)==sorted(expected_trials)
        samples.sort(key=lambda x:x['trial'])
        records=defaultdict(list);rows=defaultdict(list)
        errors=[]
        for s in samples:
            for r in s['statistics']:rows[(r['layer'],r['slot'])].append(r)
            for r in s['interventions']:
                records[(r['layer'],r['slot'])].append(dict(r,trial=s['trial'],
                    baseline_hit=s['baseline']['token_hits'][r['target_offset']]))
                for mode,m in r['measurements'].items():
                    if 'error' in m:errors.append(dict(trial=s['trial'],layer=r['layer'],slot=r['slot'],mode=mode,error=m['error']))
        arm_result=dict(trials=len(samples),errors=errors,heads=[],interventions=[],
                        peak_allocated=max(s['peak_allocated'] for s in samples),
                        max_reconstruction_relative=max(r['reconstruction_rel_l2'] for rs in rows.values() for r in rs))
        for (layer,slot),rs in sorted(rows.items()):
            h=histograms[f'L{layer:02d}_slot{slot:02d}']
            usage={}
            for side,counts in h.items():
                probs=counts/counts.sum();nz=probs[probs>0]
                usage[side]=dict(coverage=float((counts>0).mean()),max_share=float(probs.max()),
                                 entropy_effective=float(np.exp(-(nz*np.log(nz)).sum())),
                                 collision_effective=float(1/(probs@probs)))
            def avg(key):return float(np.mean([r[key] for r in rs]))
            curve=[]
            for i in range(6):
                ns=sum(r['curve'][i]['source'] for r in rs);nd=sum(r['curve'][i]['candidate'] for r in rs)
                curve.append(dict(length=rs[0]['curve'][i]['length'],source=ns,candidate=nd,
                                  source_fraction=ns/(ns+nd) if ns+nd else None))
            head=dict(layer=layer,slot=slot,usage=usage,curve=curve,
                                           source_gate=avg('source_gate'),source_digit_gate=avg('source_digit_gate'),
                                           source_mass=avg('source_mass'),candidate_mass=avg('candidate_mass'),
                                           protected_mass=avg('protected_mass'),candidate_edges=avg('candidate_edges'),
                                           k_symbol_share=avg('k_symbol_share'))
            if row_summary is not None:
                assert row_summary['suite_sha256']==d['suite_sha256']
                prior=next(h for h in row_summary['arms'][arm]['heads'] if h['context']==16384 and h['depth']==.05 and h['layer']==layer and h['slot']==slot)
                tau=prior['tau']
                # fallback_mass = 1 / unscaled denominator; exact counts for
                # chain lengths 1 and 2 follow from adjacent survival counts.
                short=[]
                for r in rs:
                    n1=r['curve'][0]['candidate']-r['curve'][1]['candidate']
                    n2=r['curve'][1]['candidate']-r['curve'][2]['candidate']
                    short.append(r['fallback_mass']*(n1*np.exp(tau)+n2*(np.exp(tau)+np.exp(2*tau))))
                head.update(tau=tau,short_candidate_mass=float(np.mean(short)),
                            short_share_of_candidate_mass=float(np.mean(short)/max(avg('candidate_mass'),1e-300)))
            arm_result['heads'].append(head)
        for (layer,slot),rs in sorted(records.items()):
            cohorts={}
            for name,select in [('all',lambda r:True),('first8',lambda r:r['trial']<8),('last8',lambda r:r['trial']>=8)]:
                subset=[r for r in rs if select(r)]
                if not subset:continue
                effects={}
                modes = list(protocol[3])
                if all(mode in modes for mode in ('full', 'random0', 'random1')):
                    modes.append('full_minus_random')
                for mode in modes:
                    bytrial=defaultdict(list)
                    accuracy=defaultdict(list)
                    invalid=0
                    for r in subset:
                        m=r['measurements']
                        needed=['full','random0','random1'] if mode=='full_minus_random' else [mode]
                        if any('error' in m[k] for k in needed):invalid+=1;continue
                        value=m['full']['benefit']-(m['random0']['benefit']+m['random1']['benefit'])/2 if mode=='full_minus_random' else m[mode]['benefit']
                        bytrial[r['trial']].append(value)
                        hit=float(m['full']['hit'])-(float(m['random0']['hit'])+float(m['random1']['hit']))/2 if mode=='full_minus_random' else float(m[mode]['hit'])-float(r['baseline_hit'])
                        accuracy[r['trial']].append(hit)
                    # Do not hide numerical failures by averaging incomplete trials.
                    effects[mode]=dict(invalid_queries=invalid)
                    if not invalid:
                        assert all(len(x)==5 for x in bytrial.values())
                        effects[mode].update(estimate([np.mean(x) for x in bytrial.values()]))
                        effects[mode]['accuracy_delta']=estimate([np.mean(x) for x in accuracy.values()])
                cohorts[name]=effects
            per_digit={}
            for offset in range(1,6):
                per_digit[str(offset)]={}
                for mode in protocol[3]:
                    selected=[r['measurements'][mode] for r in rs if r['target_offset']==offset]
                    if all('error' not in r for r in selected):per_digit[str(offset)][mode]=estimate([r['benefit'] for r in selected])
            arm_result['interventions'].append(dict(layer=layer,slot=slot,cohorts=cohorts,per_digit=per_digit,
                source_mass=float(np.mean([r['diagnostics']['source_mass'] for r in rs])),
                candidate_mass=float(np.mean([r['diagnostics']['candidate_mass'] for r in rs])),
                amplification_p50_p95_max=np.exp(np.quantile([r['diagnostics']['log_amplification'] for r in rs],[.5,.95,1.])).tolist(),
                max_random_mass_error=max(abs(c['mass']-r['diagnostics']['candidate_mass']) for r in rs for c in r['diagnostics']['random_controls'])))
        output['arms'][arm]=arm_result
    assert len(hashes)==1
    output['suite_sha256']=hashes.pop()
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directories',type=Path,nargs='+',required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--partial',action='store_true')
    p.add_argument('--expected-trials',type=int,nargs='+',help='Complete trial IDs; defaults to 0..15')
    p.add_argument('--row-summary',type=Path)
    a=p.parse_args()
    row_summary=json.loads(a.row_summary.read_text()) if a.row_summary else None
    result=summarize(a.directories,not a.partial,row_summary,a.expected_trials)
    atomic_json(a.out,result)
    for arm,d in result['arms'].items():
        print(arm,'trials',d['trials'],'errors',len(d['errors']))
        for r in d['interventions']:
            print(r['layer'],r['slot'],{k:round(v.get('mean',float('nan')),4) for k,v in r['cohorts']['all'].items()})


if __name__=='__main__':main()
