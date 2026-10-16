"""Paired trial-level NIAH tau-scaling effects; positive means lower NLL."""
import argparse
import json
from pathlib import Path

import numpy as np

from niah_circuit_probe import atomic_json, digest
from summarize_niah_circuit import estimate


def summarize(directory, partial=False):
    suite=json.loads((directory/'inputs.json').read_text())
    result=dict(suite_sha256=digest(suite),sign='baseline NLL - intervention NLL; positive improves',
        uncertainty='paired trial bootstrap, 10000 resamples, seed 1729; pointwise CI',arms={})
    for arm in ('old','restored','gqa2'):
        path=directory/f'{arm}.json'
        if partial and not path.exists():continue
        raw=json.loads(path.read_text())
        assert raw['suite_sha256']==result['suite_sha256']
        if not partial:
            assert raw['buffers_unchanged']
            assert len(raw['cells'])==len(suite['cells'])
        cells=[]
        for cell in raw['cells']:
            m=cell['measurements'];base=m['baseline']
            if not partial:assert len(m)==9
            record=dict(context=cell['context'],depth=cell['depth'],baseline={},effects={},comparisons={})
            vectors={}
            for condition in ('present','absent'):
                b=np.array([x['token_nll'] for x in base[condition]])
                assert b.shape==(suite['trials'],6)
                record['baseline'][condition]=dict(digit_nll=estimate(b[:,1:].mean(1)),
                    six_token_nll=estimate(b.mean(1)),digit_accuracy=estimate([np.mean(x['token_hits'][1:]) for x in base[condition]]))
            for name,measurement in m.items():
                effects={};vectors[name]={}
                for condition in ('present','absent'):
                    assert [x['trial'] for x in measurement[condition]]==[x['trial'] for x in base[condition]]
                    loss=np.array([x['token_nll'] for x in measurement[condition]])
                    original=np.array([x['token_nll'] for x in base[condition]])
                    assert np.isfinite(loss).all()
                    benefit=original-loss;vectors[name][condition]=benefit[:,1:].mean(1)
                    effects[condition]=dict(digit_benefit=estimate(vectors[name][condition]),
                        six_token_benefit=estimate(benefit.mean(1)),
                        per_digit=[estimate(benefit[:,i]) for i in range(1,6)],
                        digit_accuracy_delta=estimate([np.mean(x['token_hits'][1:])-np.mean(y['token_hits'][1:])
                            for x,y in zip(measurement[condition],base[condition])]))
                effects['present_minus_absent_benefit']=estimate(vectors[name]['present']-vectors[name]['absent'])
                record['effects'][name]=effects
            for scope in ('selected','all'):
                for variant in ('mul','add'):
                    schedule=f'{scope}_log_{variant}';fixed=f'{scope}_fixed_{variant}'
                    if schedule in vectors and fixed in vectors:
                        record['comparisons'][f'{schedule}_minus_fixed']={condition:estimate(vectors[schedule][condition]-vectors[fixed][condition])
                            for condition in ('present','absent')}
            cells.append(record)
        result['arms'][arm]=dict(cells=cells,peak_allocated=max(m['peak_allocated'] for c in raw['cells'] for m in c['measurements'].values()))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--partial',action='store_true')
    a=p.parse_args();result=summarize(a.directory,a.partial);atomic_json(a.out,result)
    for arm,d in result['arms'].items():
        for c in d['cells']:
            print(arm,c['context'],c['depth'],{k:round(v['present']['digit_benefit']['mean'],4) for k,v in c['effects'].items()})


if __name__=='__main__':main()
