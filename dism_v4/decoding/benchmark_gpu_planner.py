"""Paired ordinary-step benchmark, same tokens for CPU and GPU planners."""
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from dism_v4.decoding.gpu_runtime import GpuPlannerCache
from dism_v4.decoding.runtime import NativeDecodeCache

@torch.inference_mode()
def main():
    torch.set_num_threads(2);torch.manual_seed(173)
    output=Path(__file__).parent/'results/gpu_planner_benchmark.json'
    report=[]
    for n in (2048,16384):
        for pattern in ('zipf','repeat'):
            h,r,d,steps=12,32,64,128
            labels=torch.zeros(n+steps,3,h,device='cuda',dtype=torch.int32)
            if pattern=='zipf':
                probs=torch.arange(1,513,device='cuda').float().pow(-1.2)
                labels[:,:2]=torch.multinomial(probs,(n+steps)*2*h,True).reshape(n+steps,2,h).int()
            sk=torch.randn(n+steps,h,r,device='cuda',dtype=torch.bfloat16)
            sq=torch.randn_like(sk);v=torch.randn(n+steps,h,d,device='cuda',dtype=torch.bfloat16)
            options=dict(rebuild_interval=512,sample_interval=32 if n==2048 else 128,
                         materialize_threshold=64 if n==2048 else 160,rebuild_chunk=128)
            row=dict(n=n,pattern=pattern,heads=h,r=r,dv=d,steps=steps,options=options)
            def create(cls):
                c=cls(h,r,d,n+steps,[.7]*h,**options)
                c.prime(labels[:n],sk[:n].transpose(0,1).contiguous(),v[:n].transpose(0,1).contiguous())
                torch.cuda.synchronize();return c
            expected=[]
            for name,cls in [('cpu',NativeDecodeCache),('gpu',GpuPlannerCache)]:
                cache=create(cls);elapsed=[]
                for i in range(n,n+steps):
                    torch.cuda.synchronize();started=time.perf_counter_ns()
                    result=cache.step(labels[i],sk[i],sq[i],v[i])
                    torch.cuda.synchronize();elapsed.append((time.perf_counter_ns()-started)/1000)
                    if name=='cpu':expected.append(result.clone())
                    else:torch.testing.assert_close(result,expected[i-n],atol=3e-4,rtol=3e-4)
                row[name+'_wall_median_us']=float(np.median(elapsed[16:]))
                row[name+'_wall_p95_us']=float(np.percentile(elapsed[16:],95))
                if name=='gpu':assert cache.check_status()==n+steps
            # Capture a fixed 128-token sequence. Each replayed kernel advances
            # its position on device; no CPU metadata update between kernels.
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                cache=create(GpuPlannerCache)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream):
                    for i in range(n,n+steps):last=cache.step(labels[i],sk[i],sq[i],v[i])
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record();graph.replay();end.record();end.synchronize()
                row['graph_sequence_us_per_step']=start.elapsed_time(end)*1000/steps
                assert cache.check_status()==n+steps
                torch.testing.assert_close(last,expected[-1],atol=3e-4,rtol=3e-4)
            # Device timeline: identify kernel latency, verify no transfer on step.
            cache=create(GpuPlannerCache)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as prof:
                for i in range(n,n+32):cache.step(labels[i],sk[i],sq[i],v[i])
                torch.cuda.synchronize()
            path=output.with_name(f'gpu_planner_{n}_{pattern}_trace.json')
            prof.export_chrome_trace(str(path))
            events=json.loads(path.read_text())['traceEvents']
            kernels=[e['dur'] for e in events if e.get('cat')=='kernel']
            assert len(kernels)==32 and not any(e.get('cat')=='gpu_memcpy' for e in events)
            row['kernel_median_us']=float(np.median(kernels))
            row['kernel_p95_us']=float(np.percentile(kernels,95))
            row['ordinary_step_memcpy_count']=0
            report.append(row);output.write_text(json.dumps(report,indent=2))
            print(json.dumps(row),flush=True)

if __name__=='__main__':main()
