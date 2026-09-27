"""Separate-process OUTPUT timing: saved82f9305 vs Q/KV and Q/K reuse."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--baseline-binary',required=True)
    parser.add_argument('--output',type=Path,default=Path('/tmp/dism-q-alias.json'))
    args=parser.parse_args()
    cases=[(1024,.5,d,r) for r in range(3) for d in ('q_from_k','k_from_q')]
    cases += [(1024,0.,d,0) for d in ('q_from_k','k_from_q')]
    cases += [(n,.5,'q_from_k',0) for n in (65,257)]
    rows=[]
    for ix,(n,prob,direction,rep) in enumerate(cases):
        variants=('baseline','kv','k') if ix%2==0 else ('k','kv','baseline')
        for variant in variants:
            env=dict(os.environ,DISM_OUTPUT_Q_ALIAS='none' if variant=='baseline' else variant)
            cmd=[sys.executable,'-m','dism_v2.benchmark_persistent_summary','--kernel','output',
                 '--n',str(n),'--d','64','--dv','64','--hard-prob',str(prob),'--direction',direction]
            if variant=='baseline': cmd += ['--baseline-binary',args.baseline_binary]
            p=subprocess.run(cmd,env=env,text=True,capture_output=True,timeout=180)
            if p.returncode: raise RuntimeError(p.stdout+'\n'+p.stderr)
            row=json.loads(p.stdout.strip().splitlines()[-1]);row.update(variant=variant,round=rep)
            rows.append(row)
            print(n,prob,direction,rep,variant,row['median_us'],flush=True)
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(dict(baseline_commit='82f9305',
                scope='OUTPUT GPU launch in ordinary forward stream, no concurrent GPU tests; '
                      '20 warmups/30 CUPTI samples; clocks unlocked',measurements=rows),indent=2)+'\n')
    for n,prob,direction in dict.fromkeys((n,p,d) for n,p,d,_ in cases):
        print(n,prob,direction,{v:statistics.median(r['median_us'] for r in rows
              if (r['n'],r['hard_prob'],r['direction'],r['variant'])==(n,prob,direction,v))
              for v in ('baseline','kv','k')})


if __name__=='__main__': main()
