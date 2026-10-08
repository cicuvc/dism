"""Compare actual BF16 GEMM intermediates with FP64 and tf32x3 paths."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import triton
from prefill import HardPrefillPlan
from prefill_triton import TritonPrefillPlan


def metrics(out, ref):
    x,y=out.astype(np.float64),ref.astype(np.float64)
    err=x-y
    nx,ny=np.linalg.norm(x),np.linalg.norm(y)
    row_y=np.linalg.norm(y,axis=1)
    valid=row_y>1e-10
    row_x=np.linalg.norm(x[valid],axis=1)
    cos=(x[valid]*y[valid]).sum(1)/np.maximum(row_x*row_y[valid],1e-300)
    ratios=row_x/row_y[valid]
    return dict(max_abs=float(np.max(np.abs(err))),relative_l2=float(np.linalg.norm(err)/ny),
                cosine=float((x*y).sum()/(nx*ny)),norm_ratio=float(nx/ny),
                signed_projection_bias=float((err*y).sum()/(ny*ny)),
                mean_error=float(err.mean()),row_cosine_min=float(cos.min()),
                row_cosine_p01=float(np.quantile(cos,.01)),
                row_norm_ratio_p01=float(np.quantile(ratios,.01)),
                row_norm_ratio_p99=float(np.quantile(ratios,.99)))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path)
    parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args()
    rng=np.random.default_rng(5273)
    records=[]
    for case in range(4 if args.smoke else 32):
        n=257 if case<16 else 4096
        r,dv=((16,32),(16,64),(32,32),(32,64))[case%4]
        tau=(0.,.7,np.log(64.),1e-9)[(case//4)%4]
        repeat=case%2==0
        q,k=rng.integers(0,16,(2,n))
        if repeat:q[:]=k[:]=0
        reset=np.zeros(n,dtype=bool)
        if case%5==0:reset[::37]=True
        p=HardPrefillPlan(q,k,tau,reset=reset)
        sq,sk=[torch.tensor(rng.normal(size=(n,r)),device="cuda",dtype=torch.float32) for _ in range(2)]
        if case%3==0:
            sq=torch.nn.functional.silu(sq);sk=torch.nn.functional.silu(sk)
        sq=sq.bfloat16();sk=sk.bfloat16()
        v=torch.tensor(rng.normal(size=(n,dv)),device="cuda",dtype=torch.bfloat16)
        ref=p.execute(*(x.float().cpu().numpy() for x in (sq,sk,v)),dtype=np.float64)
        for c in (16,32,64):
            gpu=TritonPrefillPlan(p,chunk_size=c)
            assert gpu.mma_precision=="bf16"
            out=gpu.execute(sq,sk,v).cpu().numpy()
            assert np.isfinite(out).all()
            gpu.mma_precision="tf32x3"
            exact=gpu.execute(sq,sk,v).cpu().numpy()
            np.testing.assert_allclose(exact,ref,rtol=3e-4,atol=3e-4)
            gpu.mma_precision="bf16"
            gpu.execute(sq,sk,v)
            record=dict(case=case,n=n,r=r,dv=dv,tau=tau,repeat=repeat,c=c,
                        silu=case%3==0,**metrics(out,ref),
                        tf32x3_relative_l2=metrics(exact,ref)["relative_l2"],
                        regs=gpu.last_kernel.n_regs,spills=gpu.last_kernel.n_spills)
            if case>=30:
                record["bf16_ms"]=triton.testing.do_bench(lambda:gpu.execute(sq,sk,v),warmup=50,rep=100)
                gpu.mma_precision="tf32x3"
                record["tf32x3_ms"]=triton.testing.do_bench(lambda:gpu.execute(sq,sk,v),warmup=50,rep=100)
            records.append(record)
    summary={}
    for c in (16,32,64):
        group=[x for x in records if x["c"]==c]
        summary[c]=dict(cases=len(group),min_cosine=min(x["cosine"] for x in group),
                        max_relative_l2=max(x["relative_l2"] for x in group),
                        norm_ratio_range=[min(x["norm_ratio"] for x in group),max(x["norm_ratio"] for x in group)],
                        projection_bias_range=[min(x["signed_projection_bias"] for x in group),max(x["signed_projection_bias"] for x in group)],
                        mean_projection_bias=float(np.mean([x["signed_projection_bias"] for x in group])),
                        worst_row_cosine=min(x["row_cosine_min"] for x in group))
    report=dict(status="MEASURED",summary=summary,cases=records)
    if args.output:args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
