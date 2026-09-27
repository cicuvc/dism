"""Compare summary GPU launch times in ordinary forward streams, not training TPS."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys


def legacy_worker():
    import torch
    import tt_dism as old
    torch.set_default_device('cuda')
    torch.manual_seed(0)
    with torch.no_grad():
        q,k,v=[torch.randn(64,4,1024,64,dtype=torch.bfloat16) for _ in range(3)]
        beta=torch.randn(64,4,1024,dtype=torch.float32)
        tau=torch.full((4,),3.,dtype=torch.float32)
        # Same inputs as the original wrapper; preprocessing is outside timing.
        q,k=[torch.softmax(x,-1).bfloat16().contiguous() for x in (q,k)]
        def run(): return old.parallel_attn_fwd(q,k,v,beta,tau)
        for _ in range(20): result=run()
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as prof:
            for _ in range(30): result=run()
            torch.cuda.synchronize()
        times=[e.device_time_total for e in prof.events()
               if e.device_type==torch.autograd.DeviceType.CUDA and e.name=='perprocess_kernel_hh']
        assert len(times)==30,len(times)
        assert torch.isfinite(result[0]).all()
    print(json.dumps(dict(backend='legacy_triton',gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,triton=old.triton.__version__,b=64,h=4,n=1024,
        n_headdim=64,n_vocab=64,query_chunk=32,diagonal_chunk=32,
        samples_us=times,median_us=statistics.median(times))))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--legacy-worker',action='store_true')
    parser.add_argument('--rounds',type=int,default=3)
    parser.add_argument('--output',type=Path,default=Path('/tmp/dism-summary-comparison.json'))
    args=parser.parse_args()
    if args.legacy_worker:
        legacy_worker()
        return
    if args.rounds<1: parser.error('positive rounds required')
    variants=[('legacy_triton',None,None),('cuda_soft_q',0.,'q_from_k'),
              ('cuda_soft_k',0.,'k_from_q'),('cuda_mixed_q',.5,'q_from_k'),
              ('cuda_mixed_k',.5,'k_from_q')]
    rows=[]
    for rep in range(args.rounds):
        for name,prob,direction in (variants if rep%2==0 else list(reversed(variants))):
            if prob is None:
                cmd=[sys.executable,'-m','dism_v2.benchmark_summary_comparison','--legacy-worker']
            else:
                cmd=[sys.executable,'-m','dism_v2.benchmark_persistent_summary',
                     '--hard-prob',str(prob),'--direction',direction]
            result=subprocess.run(cmd,text=True,capture_output=True,check=True,timeout=180)
            row=json.loads(result.stdout.strip().splitlines()[-1])
            row.update(variant=name,round=rep)
            rows.append(row)
            print(name,rep,row['median_us'],flush=True)
    medians={name:statistics.median(r['median_us'] for r in rows if r['variant']==name)
             for name,_,_ in variants}
    data=dict(scope='CUPTI summary launch duration in normal forward stream; '
        '20 warmups/30 samples per process; no per-launch sync or graphs; '
        'different algorithms, no end-to-end throughput claim',
        tile_lse=os.environ.get('DISM_TILE_LSE','full'),
        lineinfo=os.environ.get('DISM_LINEINFO','0'),rounds=args.rounds,
        median_of_round_medians_us=medians,
        relative_to_triton={name:medians['legacy_triton']/value for name,value in medians.items()},
        measurements=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps({k:v for k,v in data.items() if k!='measurements'}),flush=True)


if __name__=='__main__': main()
