"""One-shot prefill vs cached vector execution; one sequence and one head.

Excludes model projections/vocabulary selection and final decoding-cache prime.
Includes GPU label D2H, CPU planning, native chunk packing/H2D, and output zero.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
import triton
from prefill import HardPrefillPlan, load_prefill
from prefill_triton import TritonPrefillPlan


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths",type=int,nargs="+",default=[1024,4096,16384,65536])
    parser.add_argument("--repeats",type=int,default=5)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    rng=np.random.default_rng(7219)
    torch.manual_seed(7219)
    load_prefill()
    records=[]
    for pattern in ("uniform512","zipf512","repeat"):
        for n in args.lengths:
            if pattern=="uniform512": q,k=rng.integers(0,512,(2,n),dtype=np.int32)
            elif pattern=="zipf512":
                prob=np.arange(1,513,dtype=np.float64)**-1.2;prob/=prob.sum()
                q,k=rng.choice(512,size=(2,n),p=prob).astype(np.int32)
            else:q=k=np.zeros(n,dtype=np.int32)
            labels=torch.tensor(np.stack((q,k,np.zeros(n,dtype=np.int32))),device="cuda")
            sq,sk=[torch.randn(n,32,device="cuda",dtype=torch.bfloat16) for _ in range(2)]
            v=torch.randn(n,64,device="cuda",dtype=torch.bfloat16)
            # Prime JIT, CPU allocator and GPU allocations; exclude compilation.
            p=HardPrefillPlan(q,k,np.log(64.))
            g=TritonPrefillPlan(p)
            for _ in range(3):g.execute(sq,sk,v)
            torch.cuda.synchronize()
            samples=[]
            for _ in range(args.repeats):
                del p,g
                torch.cuda.synchronize()
                t0=time.perf_counter()
                host=labels.cpu().numpy()
                t1=time.perf_counter()
                p=HardPrefillPlan(host[0],host[1],np.log(64.),reset=host[2].astype(bool))
                t2=time.perf_counter()
                g=TritonPrefillPlan(p)
                torch.cuda.synchronize()
                t3=time.perf_counter()
                out=g.execute(sq,sk,v)
                torch.cuda.synchronize()
                t4=time.perf_counter()
                samples.append(dict(d2h_ms=(t1-t0)*1000,plan_ms=(t2-t1)*1000,
                                    pack_upload_ms=(t3-t2)*1000,execute_wall_ms=(t4-t3)*1000,
                                    total_ms=(t4-t0)*1000))
            assert torch.isfinite(out).all()
            gpu_ms=triton.testing.do_bench(lambda:g.execute(sq,sk,v),warmup=100,rep=200)
            median={key:float(np.median([x[key] for x in samples])) for key in samples[0]}
            record=dict(pattern=pattern,n=n,r=32,dv=64,tau=float(np.log(64.)),
                        **median,gpu_cached_ms=gpu_ms,
                        end_to_end_tokens_s=n/(median["total_ms"]*.001),
                        vector_tokens_s=n/(gpu_ms*.001),
                        program=p.statistics(),gpu=g.statistics(),samples=samples)
            records.append(record)
            print(json.dumps({k:v for k,v in record.items() if k not in ("samples","program","gpu")}),flush=True)
            del out,sq,sk,v,labels,p,g
    report=dict(device=torch.cuda.get_device_name(),torch=torch.__version__,triton=triton.__version__,
                batch=1,heads=1,chunk_size=16,mma_precision="bf16",input_dtype="bf16",repeats=args.repeats,
                excludes="model projections, vocab interpolation, final decoding cache prime; no compile time",results=records)
    args.output.write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":main()
