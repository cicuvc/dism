"""Paired validation summaries for frozen K-price experiments."""
import argparse
import json
from pathlib import Path

import numpy as np

from niah_circuit_probe import atomic_json


def summarize(result):
    base = result['arms']['baseline']
    counts = np.array([b['tokens'] for b in base['batches']])
    baseline_nats = np.array([b['nats'] for b in base['batches']])
    rng = np.random.default_rng(1729)
    draws = rng.integers(len(counts),size=(10000,len(counts)))
    out = dict(checkpoint=result['checkpoint'], valid_tokens=base['valid_tokens'],
               uncertainty='paired packed-row bootstrap, 10000 resamples, seed 1729; exploratory pointwise intervals', arms={})
    for name, arm in result['arms'].items():
        assert [b['tokens'] for b in arm['batches']] == counts.tolist()
        delta = np.array([b['nats'] for b in arm['batches']])-baseline_nats
        bootstrap = delta[draws].sum(1)/counts[draws].sum(1)
        heads = arm['heads']
        changed = [h for h in heads if arm.get('only') is None or [h['layer'],h['head']] == arm['only']]
        chains=[]
        for length in range(3):
            original=sum(h['sampled_causal_chains'][length]['original'] for h in changed)
            modified=sum(h['sampled_causal_chains'][length]['modified'] for h in changed)
            intersection=sum(h['sampled_causal_chains'][length]['intersection'] for h in changed)
            chains.append(dict(length=length+1,original=original,modified=modified,intersection=intersection,
                               retention=intersection/original if original else None))
        out['arms'][name]=dict(nll=arm['nll'],bpb=arm['bpb'],delta_nll=float(delta.sum()/counts.sum()),
            delta_nll_ci95=np.quantile(bootstrap,[.025,.975]).tolist(),
            mean_active=float(np.mean([h['k']['active'] for h in heads])),
            mean_effective=float(np.mean([h['k']['effective'] for h in heads])),
            min_active=min(h['k']['active'] for h in heads),min_effective=min(h['k']['effective'] for h in heads),
            mean_max_frequency=float(np.mean([h['k']['max_frequency'] for h in heads])),
            mean_changed_fraction=float(np.mean([h['changed_fraction'] for h in changed])),
            mean_score_regret=float(np.mean([h['mean_score_regret'] for h in changed])),
            max_fp32_label_disagreement=max(h['fp32_label_disagreement'] for h in heads),
            selected_head_chain_counts=chains,peak_allocated_gib=arm['peak_allocated']/2**30)
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    a=parser.parse_args()
    result=json.loads((a.directory/'results.json').read_text())
    assert result['buffers_unchanged']
    summary=summarize(result)
    atomic_json(a.directory/'summary.json',summary)
    for name,arm in summary['arms'].items():
        print(name,json.dumps(arm))


if __name__=='__main__':
    main()
