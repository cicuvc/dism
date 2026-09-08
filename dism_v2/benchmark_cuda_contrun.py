"""Same-stream continuous CUDA voc_dism forward/backward, without a model.

DISM_TILE_LSE=tanh python -m dism_v2.benchmark_cuda_contrun
Like tt_dism_contrun.py: fresh random dO, backward(), accumulated leaf grads,
10000 iterations and no per-iteration synchronization. Includes CUDA embedding.
"""
import argparse
import json
import time

import torch
from tqdm import tqdm
from .autograd import voc_dism
from .kernel_config import TILE_LSE


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--steps',type=int,default=10000)
    parser.add_argument('--warmup',type=int,default=20)
    parser.add_argument('--vocab',type=int,default=512)
    args=parser.parse_args()
    if min(args.steps,args.warmup,args.vocab)<1:
        parser.error('positive steps/warmup/vocab required')
    torch.set_default_device('cuda:0')
    torch.manual_seed(0)
    q,k,v=[torch.randn(64,4,1024,64,dtype=torch.bfloat16,requires_grad=True) for _ in range(3)]
    tau=torch.full((4,),3.,dtype=torch.float32,requires_grad=True)
    qv,kv=[torch.randn(4,args.vocab,64,dtype=torch.bfloat16,requires_grad=True) for _ in range(2)]
    inputs=(q,k,v,tau,qv,kv)
    generator=torch.Generator(device='cuda').manual_seed(777)
    def step():
        out=voc_dism(*inputs,sm_scale=1.,hard_prob=.5,direction='random',generator=generator,
            embedding_backend='cuda',embedding_backward_backend='cuda')
        out.backward(torch.randn_like(out))
        return out
    for _ in range(args.warmup):
        out=step()
    torch.cuda.synchronize()
    assert torch.isfinite(out).all() and all(torch.isfinite(x.grad).all() for x in inputs)
    for x in inputs:
        x.grad=None
    generator.manual_seed(777)
    torch.manual_seed(0)
    stream=torch.cuda.current_stream()
    start,end=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0=time.perf_counter()
    start.record(stream)
    for _ in tqdm(range(args.steps)):
        out=step()
    end.record(stream)
    submitted=time.perf_counter()
    end.synchronize()
    finished=time.perf_counter()
    wall=finished-t0
    event_ms=start.elapsed_time(end)
    # Checks happen outside the timed interval; no changes to accumulation.
    finite=bool(torch.isfinite(out).all() and all(torch.isfinite(x.grad).all() for x in inputs))
    result=dict(iterations=args.steps,wall_seconds=wall,event_ms=event_ms,
        host_submit_seconds=submitted-t0,final_drain_seconds=finished-submitted,
        wall_ms_per_iteration=wall*1000/args.steps,event_ms_per_iteration=event_ms/args.steps,
        iterations_per_second=args.steps/wall,input_tokens_per_second=args.steps*64*1024/wall,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),stream=int(stream.cuda_stream),
        output_and_accumulated_grads_finite=finite)
    print(json.dumps(dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,
        tile_lse=TILE_LSE,batch=64,heads=4,n=1024,d=64,dv=64,vocab=args.vocab,hard_prob=.5,
        direction='random',scale=1.,tau=3.,warmup=args.warmup,seed=0,rng_seed=777,
        embedding_forward='cuda',embedding_backward='cuda_paired',
        scope='One CUDA voc_dism forward + randn_like(dO) + backward with accumulated leaf grads; includes CUDA embedding and all auxiliary kernels. One default stream, no graph/model/optimizer. Warmup and input construction excluded. Event interval includes GPU idle gaps.',
        result=result)),flush=True)
    if not finite:
        raise RuntimeError('nonfinite output or accumulated gradient')


if __name__=='__main__':
    main()
