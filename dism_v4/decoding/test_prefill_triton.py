"""GPU numerical/codegen/timing checks, independent of training extension."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
import triton
from prefill import HardPrefillPlan
from prefill_triton import TritonPrefillPlan

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from test_prefill_reference import dense_exact


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    rng = np.random.default_rng(6182)
    records = []
    shapes = [(3,5),(16,32),(32,64),(64,32),(32,128)]
    for c in ((32,) if args.smoke else (16,32,64)):
        for case in range(4 if args.smoke else 20):
            n = (1,33,257,1024)[case % 4]
            r,dv = shapes[case % len(shapes)]
            keys, queries = rng.integers(0,7,(2,n))
            if case % 4 in (0,1): keys[:]=queries[:]=0
            if case % 7 == 2: queries+=100
            reset = rng.random(n)<.07
            if case % 3: reset[:]=False
            tau = (0.,.7,np.log(64.),-.7)[case % 4]
            p = HardPrefillPlan(queries,keys,tau,reset=reset)
            g = TritonPrefillPlan(p,chunk_size=c,mma_precision="tf32x3")
            dtype = torch.bfloat16 if case % 2 else torch.float32
            sq,sk = [torch.tensor(rng.normal(size=(n,r)),device="cuda",dtype=dtype) for _ in range(2)]
            v = torch.tensor(rng.normal(size=(n,dv)),device="cuda",dtype=dtype)
            arrays = [x.float().cpu().numpy().astype(np.float64) for x in (sq,sk,v)]
            expected = p.execute(*arrays,dtype=np.float64)
            exact,_ = dense_exact(queries,keys,reset,tau,*arrays)
            np.testing.assert_allclose(expected,exact,rtol=2e-9,atol=2e-9)
            try:
                out = g.execute(sq,sk,v).cpu().numpy()
            except Exception:
                print(f"FAIL launch: chunk={c}, N={n}, R={r}, DV={dv}, case={case}", flush=True)
                raise
            np.testing.assert_allclose(out,expected,rtol=3e-4,atol=3e-4)
            k = g.last_kernel
            records.append(dict(n=n,r=r,dv=dv,c=c,case=case,max_abs=float(np.max(np.abs(out-expected))),
                                regs=k.n_regs if k else 0,spills=k.n_spills if k else 0,
                                **g.statistics()))
    # High log score and multi-chunk streams: no global-pivot underflow.
    for tau in (.7,np.log(64.),1000.):
        n=4096 if not args.smoke else 257
        p=HardPrefillPlan(np.zeros(n,dtype=np.int32),np.zeros(n,dtype=np.int32),tau)
        g=TritonPrefillPlan(p,chunk_size=32,mma_precision="tf32x3")
        sq,sk,v=[torch.randn((n,32),device="cuda") for _ in range(3)]
        expected=p.execute(*(x.cpu().numpy() for x in (sq,sk,v)),dtype=np.float64)
        out=g.execute(sq,sk,v).cpu().numpy()
        np.testing.assert_allclose(out,expected,rtol=3e-4,atol=3e-4)
        records.append(dict(stress_tau=tau,n=n,max_abs=float(np.max(np.abs(out-expected)))))
    timing=[]
    if not args.smoke:
        for pattern in ("random","repeat"):
            n=4096
            q,k=rng.integers(0,16,(2,n))
            if pattern=="repeat": q[:]=k[:]=0
            p=HardPrefillPlan(q,k,np.log(64.))
            sq,sk=[torch.randn(n,32,device="cuda") for _ in range(2)]
            v=torch.randn(n,64,device="cuda")
            for c in (16,32,64):
                torch.cuda.synchronize(); start=time.perf_counter()
                g=TritonPrefillPlan(p,chunk_size=c,mma_precision="tf32x3")
                torch.cuda.synchronize(); prepare_ms=(time.perf_counter()-start)*1000
                g.execute(sq,sk,v)
                ms=triton.testing.do_bench(lambda:g.execute(sq,sk,v),warmup=100,rep=200)
                timing.append(dict(pattern=pattern,c=c,prepare_ms=prepare_ms,execute_ms=ms,
                                   regs=g.last_kernel.n_regs,spills=g.last_kernel.n_spills,**g.statistics()))
    report=dict(status="PASS",cases=records,timing=timing)
    if args.output: args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__": main()
