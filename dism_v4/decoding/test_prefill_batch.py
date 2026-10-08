"""Batch/head CPU planning + combined GPU grid integration regression."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
import triton
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from dism_v4.decoding import HardDismPrefill, HardPrefillPlan
from dism_v4.decoding.prefill_triton import TritonPrefillPlan


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    torch.manual_seed(638)
    rng=np.random.default_rng(638)
    cases=[]
    for precision in ("bf16","tf32x3"):
        with HardDismPrefill(mma_precision=precision) as engine:
            assert engine.workers==8
            for case in range(2 if args.smoke else 6):
                b,h,n=2,3,(1,65,257)[case%3]
                r,dv=(16,32) if case%2 else (32,64)
                labels=rng.integers(0,7,(2,b,h,n),dtype=np.int32)
                labels[:,0,0,:]=0
                labels[0,1,0,:]=100 # A fully inactive head among active heads.
                if case==5:labels[0,:,:,:]=100 # Entire batch with no streams.
                q,k=[torch.from_numpy(x).cuda() for x in labels]
                reset=torch.tensor(rng.random((b,h,n))<.1,device="cuda")
                tau=torch.tensor([[.7,0.,np.log(64.)],[1.,-.7,1e-9]],dtype=torch.float64)
                sq,sk=[torch.randn(b,n,h,r,device="cuda",dtype=torch.bfloat16) for _ in range(2)]
                v=torch.randn(b,n,h,dv,device="cuda",dtype=torch.bfloat16)
                prepared=engine.prepare(q,k,tau,reset=reset)
                out=prepared.execute(sq,sk,v)
                expected=torch.empty_like(out)
                ref=np.empty(out.shape,dtype=np.float64)
                for bi in range(b):
                    for hi in range(h):
                        p=HardPrefillPlan(labels[0,bi,hi],labels[1,bi,hi],float(tau[bi,hi]),reset=reset[bi,hi].cpu().numpy())
                        inputs=[x[bi,:,hi].contiguous() for x in (sq,sk,v)]
                        single=TritonPrefillPlan(p,mma_precision=precision)
                        expected[bi,:,hi]=single.execute(*inputs)
                        ref[bi,:,hi]=p.execute(*(x.float().cpu().numpy() for x in inputs),dtype=np.float64)
                torch.testing.assert_close(out,expected,rtol=2e-5,atol=2e-5)
                if precision=="tf32x3":
                    np.testing.assert_allclose(out.cpu().numpy(),ref,rtol=3e-4,atol=3e-4)
                # CPU labels, [H] tau and one-shot API; identical all-head tau.
                out2=engine(q.cpu(),k.cpu(),sq,sk,v,tau[0],reset=reset.cpu())
                direct=engine.prepare(q,k,tau[0],reset=reset).execute(sq,sk,v)
                torch.testing.assert_close(out2,direct,rtol=2e-5,atol=2e-5)
                other=torch.cuda.Stream()
                with torch.cuda.stream(other):
                    try:prepared.execute(sq,sk,v)
                    except ValueError:pass
                    else:raise AssertionError("cross-stream use accepted")
                cases.append(dict(precision=precision,case=case,
                    max_abs_vs_fp64=float(np.max(np.abs(out.cpu().numpy()-ref))),
                    **prepared.statistics()))
    timing=[]
    if not args.smoke:
        b,h,n=2,8,4096
        prob=np.arange(1,513,dtype=float)**-1.2;prob/=prob.sum()
        q,k=[torch.tensor(rng.choice(512,(b,h,n),p=prob),dtype=torch.int32,device="cuda") for _ in range(2)]
        tau=np.full(h,np.log(64.))
        sq,sk=[torch.randn(b,n,h,32,device="cuda",dtype=torch.bfloat16) for _ in range(2)]
        v=torch.randn(b,n,h,64,device="cuda",dtype=torch.bfloat16)
        for workers in (1,8):
            with HardDismPrefill(workers=workers) as engine:
                engine(q,k,sq,sk,v,tau);torch.cuda.synchronize()
                samples=[]
                for _ in range(3):
                    t=time.perf_counter();prepared=engine.prepare(q,k,tau)
                    prepared.execute(sq,sk,v);torch.cuda.synchronize()
                    samples.append(1000*(time.perf_counter()-t))
                gpu_ms=triton.testing.do_bench(lambda:prepared.execute(sq,sk,v),warmup=100,rep=200)
                timing.append(dict(workers=workers,total_ms=float(np.median(samples)),samples_ms=samples,
                                   gpu_ms=gpu_ms,**prepared.statistics()))
    report=dict(status="PASS",cases=cases,timing=timing)
    if args.output:args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(dict(status="PASS",cases=len(cases),timing=timing),indent=2))


if __name__=="__main__":main()
