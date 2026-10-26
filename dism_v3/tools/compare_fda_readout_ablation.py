"""Pair complete FDA readout-ablation results by document and prompt identity."""
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    args=parser.parse_args();root=args.output.resolve()
    manifest=json.loads((root/'manifest.json').read_text());records=[]
    for job in manifest['jobs']:
        path=root/job['key']/'results.json'
        if not path.exists() or not (path.parent/'provenance.json').exists():continue
        found=json.loads(path.read_text())
        reference_path=Path(manifest['reference_top4_results'][str(job['context'])])
        reference=json.loads(reference_path.read_text())
        old={s['doc_id']:s for s in reference['samples']['fda']}
        new={s['doc_id']:s for s in found['samples']['fda']}
        assert len(old)==len(reference['samples']['fda'])
        assert len(new)==len(found['samples']['fda']) and old.keys()==new.keys()
        flips=[];counts={'both_correct':0,'both_wrong':0,'top2_only':0,'top4_only':0}
        changed=0
        for key,a in old.items():
            b=new[key]
            for identity in ('doc_hash','prompt_hash','target_hash','target','filter'):
                assert a[identity]==b[identity],(key,identity)
            x,y=int(a['contains']),int(b['contains'])
            name=('both_correct' if x else 'both_wrong') if x==y else ('top2_only' if y else 'top4_only')
            counts[name]+=1
            changed+=int(a['filtered_resps']!=b['filtered_resps'])
            if x!=y:
                flips.append(dict(doc_id=key,direction=name,target=a['target'],
                                  top4=a['filtered_resps'],top2=b['filtered_resps']))
        n=len(new);old_correct=counts['both_correct']+counts['top4_only'];new_correct=counts['both_correct']+counts['top2_only']
        assert abs(old_correct/n-reference['results']['fda']['contains,none'])<1e-12
        assert abs(new_correct/n-found['results']['fda']['contains,none'])<1e-12
        records.append(dict(context=job['context'],count=n,top4_correct=old_correct,
                            top2_correct=new_correct,top4_score=old_correct/n,top2_score=new_correct/n,
                            delta_percentage_points=(new_correct-old_correct)*100/n,
                            changed_generations=changed,**counts,flips=flips,
                            reference=str(reference_path),result=str(path)))
    result=dict(status='finished' if len(records)==len(manifest['jobs']) else 'partial',records=records)
    (root/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    for row in records:print(json.dumps({k:v for k,v in row.items() if k!='flips'},indent=2))


if __name__=='__main__':main()
