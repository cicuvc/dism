"""Time the unchanged tt_dism_contrun.py after untimed JIT warmup.

The original 10000-iteration loop keeps randn_like(dO), backward(), grad
accumulation and tqdm. No extra per-iteration synchronization or CUDA graphs.
Only wrap tqdm iteration with timing so imports/input construction are excluded.
"""
import gc
import json
from pathlib import Path
import runpy
import time

import torch
import tqdm
from tt_dism import ParallelSoftDiscreteAttention


def warmup():
    torch.set_default_device('cuda:0')
    beta=torch.randn(64,4,1024,requires_grad=True)
    tau=torch.full((4,),3.,requires_grad=True)
    v=torch.randn(64,4,1024,64,dtype=torch.bfloat16,requires_grad=True)
    q,k=[torch.randn_like(v,requires_grad=True) for _ in range(2)]
    for _ in range(20):
        out=ParallelSoftDiscreteAttention.apply(q,k,v,beta,tau)
        out.backward(torch.randn_like(out))
    torch.cuda.synchronize()
    assert torch.isfinite(out).all()
    assert all(torch.isfinite(x.grad).all() for x in (q,k,v,beta,tau))


def main():
    torch.manual_seed(0)
    warmup()
    gc.collect()
    torch.cuda.synchronize()
    torch.manual_seed(0)
    original=tqdm.tqdm
    measurements=[]
    def timed(iterable,*args,**kwargs):
        count=0
        stream=torch.cuda.current_stream()
        start,end=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0=time.perf_counter()
        start.record(stream)
        for item in original(iterable,*args,**kwargs):
            yield item
            count+=1
        end.record(stream)
        submitted=time.perf_counter()
        end.synchronize()
        wall=time.perf_counter()-t0
        event_ms=start.elapsed_time(end)
        measurements.append(dict(iterations=count,wall_seconds=wall,event_ms=event_ms,
            host_submit_seconds=submitted-t0,final_drain_seconds=time.perf_counter()-submitted,
            wall_ms_per_iteration=wall*1000/count,event_ms_per_iteration=event_ms/count,
            iterations_per_second=count/wall,input_tokens_per_second=count*64*1024/wall,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),stream=int(stream.cuda_stream)))
    tqdm.tqdm=timed
    try:
        try:
            runpy.run_path(str(Path(__file__).resolve().parents[1]/'tt_dism_contrun.py'),run_name='__main__')
        except SystemExit as e:
            if e.code not in (None,0):
                raise
    finally:
        tqdm.tqdm=original
    assert len(measurements)==1 and measurements[0]['iterations']==10000
    print(json.dumps(dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,
        batch=64,heads=4,n=1024,n_headdim=64,n_vocab=64,warmup=20,seed=0,
        scope='Original script loop: forward + randn_like(dO) + backward with accumulated leaf grads; one default stream, no graph, no optimizer/model. Imports, input construction, warmup excluded. Event interval includes GPU idle gaps, not a sum of kernel durations.',
        result=measurements[0])),flush=True)


if __name__=='__main__':
    main()
