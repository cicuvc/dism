"""Warm-cache embedding forward API latency (allocations/launch gaps included).

Run: python -m dism_v2.benchmark_embedding
The single-warp baseline is not a controlled bandwidth ablation: it uses
16 rows/CTA versus 64 in WS. No HBM bandwidth claim follows from its timing.
"""
import json
import statistics
import argparse
import torch
from .embedding import forward
from .emb_kernel import emb_fwd_wrapper


def latency(fn):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    samples=[]
    for _ in range(7):
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(30):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end)*1000/30)
    return statistics.median(samples)


def device_latency(fn):
    """Median actual CUDA kernel duration, excluding inter-launch host gaps.

    CUPTI/torch.profiler instrumentation is enabled. This measures hot inputs,
    not cold-cache bandwidth. One forward must launch exactly one kernel.
    """
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(30):
            fn()
        torch.cuda.synchronize()
    kernels=[e for e in prof.events() if e.device_type==torch.autograd.DeviceType.CUDA]
    assert len(kernels)==30, [e.name for e in kernels]
    return statistics.median(e.time_range.elapsed_us() for e in kernels)


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--device-only",action="store_true",
                        help="CUPTI kernel durations for fused BV64/BV128/Triton only")
    options=parser.parse_args()
    torch.manual_seed(941)
    print(json.dumps(dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,
                         cuda=torch.version.cuda,cache="warm repeated inputs",units="us",
                         scope="CUPTI kernel duration" if options.device_only else
                         "forward API; output allocation and host launch gaps included")))
    measure=device_latency if options.device_only else latency
    for b,h,n,v in [(1,4,65,129),(1,4,1024,1024),(1,4,4096,1024)]:
        for d in (32,64,128):
            q,k=[torch.randn(b,h,n,d,device="cuda",dtype=torch.bfloat16) for _ in range(2)]
            eq,ek=[torch.randn(h,v,d,device="cuda",dtype=torch.bfloat16) for _ in range(2)]
            args=(q,k,eq,ek,d**-.5)
            result=dict(b=b,h=h,n=n,v=v,d=d,
                cuda_ws_us=measure(lambda:forward(*args,warp_specialized=True)),
                triton_fused_us=measure(lambda:emb_fwd_wrapper(*args)))
            if not options.device_only:
                result["cuda_single_two_launches_us"]=measure(lambda:forward(*args,warp_specialized=False))
            if d!=128:
                result["cuda_ws_bv128_us"]=measure(lambda:forward(*args,block_v=128))
            print(json.dumps(result),flush=True)


if __name__=="__main__":
    main()
