"""GPU-planned decode: ordinary steps, rebuild and interval tradeoff.

Operator-only synthetic resident inputs, no model/checkpoint. 'graph' captures
one whole interval with precomputed inputs to measure the device-side floor;
it is NOT autoregressive model throughput. Capture time is reported separately.
All modes rebuild outside capture and check the device position/error flags.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from flash_dism.inference import HardDismDecoder


def summary(values):
    return dict(median=float(np.median(values)),min=min(values),max=max(values),samples=values)


def timed(fn):
    torch.cuda.synchronize()
    start=time.perf_counter()
    value=fn()
    torch.cuda.synchronize()
    return value,(time.perf_counter()-start)*1000


@torch.no_grad()
def case(args,b,n,pattern,interval,mode):
    h,r,d=args.heads,32,64
    count=args.repeats*interval
    rng=np.random.default_rng(120)
    labels=rng.integers(0,512,(n+count,3,b*h),dtype=np.int32)
    labels[:,2]=0
    if pattern=='repeat':labels[:,:2]=0
    labels=torch.from_numpy(labels).cuda()
    torch.manual_seed(120)
    keys=torch.randn(n+count,b*h,r,device='cuda',dtype=torch.bfloat16)*.2
    values=torch.randn(n+count,b*h,d,device='cuda',dtype=torch.bfloat16)*.2
    queries=torch.randn(count,b*h,r,device='cuda',dtype=torch.bfloat16)*.2
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    samples=[]
    with torch.cuda.stream(stream):
        capacity=n+(max(interval,4096) if args.fixed_context else count)
        decoder=HardDismDecoder(b,h,r,d,capacity,[.5]*h,planner_backend='gpu',rebuild_interval=interval,
                               rebuild_workers=args.rebuild_workers)
        prime_args=(labels[:n,1].reshape(n,b,h).permute(1,2,0),
                    labels[:n,0].reshape(n,b,h).permute(1,2,0),
                    keys[:n].reshape(n,b,h,r).permute(1,0,2,3).contiguous(),
                    values[:n].reshape(n,b,h,d).permute(1,0,2,3).contiguous())
        _,prime_ms=timed(lambda:decoder.prime(*prime_args))
        core=decoder.core
        raw=[(labels[n+i],keys[n+i],queries[i],values[n+i]) for i in range(count)]
        api=[(a[1].view(b,h,1),a[0].view(b,h,1),q.view(b,1,h,r),k.view(b,1,h,r),v.view(b,1,h,d))
             for a,k,q,v in raw]
        for repeat in range(args.repeats):
            if args.fixed_context and repeat:
                # Re-prime a fresh cache outside the measurement so every
                # interval starts at exactly N, independent of its length.
                graph=None
                decoder=HardDismDecoder(b,h,r,d,capacity,[.5]*h,planner_backend='gpu',rebuild_interval=interval,
                                       rebuild_workers=args.rebuild_workers)
                timed(lambda:decoder.prime(*prime_args))
                core=decoder.core
            begin=0 if args.fixed_context else repeat*interval
            capture_ms=0.
            if mode=='graph':
                graph=torch.cuda.CUDAGraph()
                start=time.perf_counter()
                with torch.cuda.graph(graph,stream=stream):
                    for i in range(begin,begin+interval):out=core.step(*raw[i])
                capture_ms=(time.perf_counter()-start)*1000
            torch.cuda.synchronize()
            e0,e1=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start=time.perf_counter();e0.record()
            if mode=='graph':graph.replay()
            else:
                for i in range(begin,begin+interval):
                    out=decoder.append(*api[i]) if mode=='api' else core.step(*raw[i])
            e1.record();e1.synchronize()
            wall_ms=(time.perf_counter()-start)*1000
            device_ms=e0.elapsed_time(e1)
            position=core.check_status()
            assert position==n+begin+interval
            assert out.isfinite().all().item()
            decoder.position=position
            # Save scalar diagnostics outside all timed intervals.
            state=core.snapshot[5].cpu()
            task_mean=float(state[:,5].float().mean())
            _,rebuild_ms=timed(core.rebuild)
            decoder.snapshot_position=position
            samples.append(dict(position=position,ordinary_wall_us=wall_ms*1000/interval,
                                ordinary_event_us=device_ms*1000/interval,rebuild_ms=rebuild_ms,
                                amortized_wall_us=(wall_ms+rebuild_ms)*1000/interval,
                                window_capture_ms=capture_ms,last_mean_tasks=task_mean))
        window_cache_stats=core.cpu.memory_stats()
        # One additional same-position rebuild broken into its public stages.
        # Incremental SAM work is zero here; use the timed real-window rebuild
        # above for end-to-end comparisons, not the sum of this diagnostic.
        position,status_ms=timed(core.check_status)
        _,refresh_ms=timed(lambda:core.cpu.native.refresh_gpu(core.snapshot[7],position,core.snapshot[6]))
        snapshot,export_ms=timed(core.cpu.native.gpu_snapshot)
        device,allocate_ms=timed(lambda:core.extension.GpuPlanner(snapshot,position))
        core.snapshot,core.device=snapshot,device
        assert core.check_status()==position
        snapshot_bytes=[t.numel()*t.element_size() for t in snapshot]
        # A one-node capture estimates core recapture overhead only; it is not
        # replayed at capacity and does not claim full-model capture cost.
        graph=torch.cuda.CUDAGraph();start=time.perf_counter()
        with torch.cuda.graph(graph,stream=stream):core.step(*raw[-1])
        single_capture_ms=(time.perf_counter()-start)*1000
        stats=core.cpu.memory_stats()
    torch.cuda.current_stream().wait_stream(stream)
    return dict(batch=b,heads=h,prompt=n,pattern=pattern,interval=interval,mode=mode,
                fixed_context=args.fixed_context,
                rebuild_workers=args.rebuild_workers,native_module=core.extension.__file__,
                prime_ms=prime_ms,samples=samples,
                summary={key:summary([s[key] for s in samples]) for key in samples[0] if key!='position'},
                rebuild_breakdown_ms=dict(status=status_ms,refresh=refresh_ms,export=export_ms,allocate=allocate_ms),
                single_node_capture_ms=single_capture_ms,snapshot_bytes=snapshot_bytes,cache_stats=stats,
                window_cache_stats=window_cache_stats)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batches',type=int,nargs='+',default=[1])
    p.add_argument('--prompts',type=int,nargs='+',default=[2048,8192])
    p.add_argument('--patterns',nargs='+',choices=['random','repeat'],default=['random','repeat'])
    p.add_argument('--intervals',type=int,nargs='+',default=[128])
    p.add_argument('--modes',nargs='+',choices=['api','raw','graph'],default=['api','raw','graph'])
    p.add_argument('--heads',type=int,default=12)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--fixed-context',action='store_true',help='restart from the same prompt for each sample')
    p.add_argument('--rebuild-workers',type=int,default=8)
    p.add_argument('--legacy-extension',type=Path,help='saved baseline extension from this worktree')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();root=Path(__file__).resolve().parents[1]
    if args.output.exists() or not args.output.resolve().is_relative_to(root/'build'):
        p.error('fresh output under this worktree dism_v3/build required')
    torch.set_num_threads(1)
    if args.legacy_extension:
        from check_decode_rebuild import legacy_extension
        legacy_extension(args.legacy_extension)
    sources=[Path(__file__),*sorted((root/'python/flash_dism/inference').glob('*.py')),
             *sorted((root/'python/flash_dism/inference/csrc').glob('*'))]
    report=dict(command=sys.argv,git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                gpu=torch.cuda.get_device_name(),torch_version=torch.__version__,
                sources={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources if f.is_file()},
                cases=[])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for b in args.batches:
        for n in args.prompts:
            for pattern in args.patterns:
                for interval in args.intervals:
                    for mode in args.modes:
                        print(f'START B{b} N{n} {pattern} I{interval} {mode}',flush=True)
                        result=case(args,b,n,pattern,interval,mode)
                        report['cases'].append(result)
                        args.output.write_text(json.dumps(report,indent=2)+'\n')
                        s=result['summary']
                        print(f'DONE ordinary={s["ordinary_wall_us"]["median"]:.2f} us '
                              f'event={s["ordinary_event_us"]["median"]:.2f} us '
                              f'rebuild={s["rebuild_ms"]["median"]:.2f} ms '
                              f'amortized={s["amortized_wall_us"]["median"]:.2f} us',flush=True)


if __name__=='__main__':main()
