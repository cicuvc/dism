"""Opt-in sqrt-block summary program: structural parallelism and GPU timings.

Synthetic projected inputs, not model throughput. Preparation is reported
separately. Both paths reuse allocations; the baseline calls unchanged production
summary kernels with CPU-generated production metadata. Results belong in build/.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import subprocess
import time

import torch
from flash_dism.inference.build import load_summary_probe


def labels_for(pattern,h,n):
    gen=torch.Generator().manual_seed(452)
    if pattern=='repeat':return torch.zeros(h,n,dtype=torch.int32)
    if pattern=='periodic':return (torch.arange(n,dtype=torch.int32)[None,:].expand(h,-1)%7).contiguous()
    return torch.randint(0,512 if pattern=='random' else 4,(h,n),generator=gen,dtype=torch.int32)


def timed(fn,repeats):
    values=[]
    for _ in range(repeats):
        torch.cuda.synchronize();start=time.perf_counter();fn();torch.cuda.synchronize()
        values.append((time.perf_counter()-start)*1000)
    return dict(median_ms=statistics.median(values),samples_ms=values)


def graph_timed(fn,repeats):
    fn();torch.cuda.synchronize()
    # Prepared programs are stream-bound, so capture on the original owner stream.
    owner=torch.cuda.current_stream();g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g,stream=owner):fn()
    values=[]
    for _ in range(repeats):
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record();g.replay();b.record();b.synchronize();values.append(a.elapsed_time(b))
    return dict(median_ms=statistics.median(values),samples_ms=values)


def error(a,b):
    a=a.double();b=b.double();diff=a-b
    return dict(max_abs=diff.abs().max().item() if diff.numel() else 0.,
                relative_l2=(diff.norm()/b.norm().clamp_min(1e-30)).item())


def case(args,h,n,pattern,tau):
    mod=load_summary_probe();labels=labels_for(pattern,h,n)
    start=time.perf_counter();p=mod.SummaryProgram(labels,[tau]*h,32,64,args.block,args.tile)
    compile_ms=(time.perf_counter()-start)*1000
    torch.manual_seed(450)
    k=torch.randn(h,n,32,device='cuda',dtype=torch.bfloat16)*.2
    v=torch.randn(h,n,64,device='cuda',dtype=torch.bfloat16)*.2
    prep=timed(lambda:p.prepare(k),1)
    stats=p.stats()
    result=dict(heads=h,n=n,pattern=pattern,tau=tau,stats=stats,compile_ms=compile_ms,prepare=prep)
    baseline=p.baseline(k,v).clone();torch.cuda.synchronize()
    result['baseline_eager']=timed(lambda:p.baseline(k,v),args.repeats)
    if stats['active_heads']:result['baseline_graph']=graph_timed(lambda:p.baseline(k,v),args.repeats)
    runs=[]
    for workers in args.workers:
        actual=p.run(k,v,workers).clone();torch.cuda.synchronize()
        assert actual.isfinite().all()
        runs.append(dict(workers=workers,error_vs_baseline=error(actual,baseline),
                         eager=timed(lambda:p.run(k,v,workers),args.repeats),
                         graph=graph_timed(lambda:p.run(k,v,workers),args.repeats) if stats['active_heads'] else None))
    result['runs']=runs
    if args.reference:
        ref=p.reference(k.float().cpu(),v.float().cpu())
        result['baseline_vs_fp64']=error(baseline.cpu(),ref)
        result['program_vs_fp64']=error(actual.cpu(),ref)
        assert result['program_vs_fp64']['relative_l2']<2e-6, result['program_vs_fp64']
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--heads',type=int,nargs='+',default=[12])
    p.add_argument('--lengths',type=int,nargs='+',default=[8192])
    p.add_argument('--patterns',nargs='+',choices=['random','repeat','periodic','branching'],default=['random','repeat','periodic','branching'])
    p.add_argument('--tau',type=float,nargs='+',default=[.5])
    p.add_argument('--block',type=int,default=0)
    p.add_argument('--tile',type=int,default=128)
    p.add_argument('--workers',type=int,nargs='+',default=[32,64,128])
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--reference',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(1)
    root=Path(__file__).resolve().parents[1]
    files=sorted((root/'python/flash_dism/inference/csrc/probe').glob('*'))+[Path(__file__)]
    result=dict(command=sys.argv,git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                gpu=torch.cuda.get_device_name(),sm_count=torch.cuda.get_device_properties(0).multi_processor_count,
                sources={str(x.relative_to(root)):hashlib.sha256(x.read_bytes()).hexdigest() for x in files},cases=[])
    for h in args.heads:
        for n in args.lengths:
            for pattern in args.patterns:
                for tau in args.tau:
                    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):r=case(args,h,n,pattern,tau)
                    stream.synchronize();result['cases'].append(r)
                    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
                    print(json.dumps(r),flush=True)
if __name__=='__main__':main()
