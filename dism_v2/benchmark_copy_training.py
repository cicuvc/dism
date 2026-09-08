"""Complete copy-task training step benchmark; one backend per fresh process.

torch: original copy_task.voc_dism (eager Torch recurrence, Triton embedding).
torch_ref: eager dism_ref, including FP32 Torch interpolation/scores/scan and BF16 PV.
cuda: current paired WS CUDA core and embedding forward/backward.
Common projection/conv/FFN/optimizer code is unchanged; FFN uses existing compile.
"""
import argparse
import collections
import json
from pathlib import Path
import statistics
import sys
import time
import torch
from .autograd import voc_dism
from .dism_ref import voc_dism_ref
from .kernel_config import TILE_LSE


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--backend',choices=['cuda','torch','torch_ref'],required=True)
    p.add_argument('--batch',type=int,default=64)
    p.add_argument('--warmup',type=int,default=3)
    p.add_argument('--steps',type=int,default=5)
    p.add_argument('--hard-prob',type=float,default=.5)
    p.add_argument('--profile',action='store_true',help='CUPTI per-launch GPU times, not wall-clock throughput')
    a=p.parse_args()
    if min(a.batch,a.warmup,a.steps)<1 or not 0<=a.hard_prob<=1:
        p.error('positive batch/warmup/steps and hard-prob in [0,1] required')
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    try:
        from . import copy_task as task
    finally: sys.path.pop(0)
    original=task.voc_dism
    def attention(q,k,v,tau,qv,kv,hard=False,lmb=.5,gen=None,sm_scale=1.):
        if a.backend=='torch':
            return original(q,k,v,tau,qv,kv,hard=False,lmb=a.hard_prob,gen=gen,sm_scale=sm_scale)
        if a.backend=='torch_ref':
            with torch.autocast('cuda',enabled=False):
                return voc_dism_ref(q,k,v,tau,qv,kv,hard_prob=a.hard_prob,
                    direction='random',generator=gen,sm_scale=sm_scale).to(v.dtype)
        return voc_dism(q,k,v,tau,qv.bfloat16().contiguous(),kv.bfloat16().contiguous(),
            hard_prob=a.hard_prob,direction='random',generator=gen,sm_scale=sm_scale,
            embedding_backend='cuda',embedding_backward_backend='cuda')
    task.voc_dism=attention
    task.TOTAL_STEPS=1000
    # The original block hard-codes vocabulary 256: replace its attention factory
    # argument while retaining the exact block/model construction order.
    cls=task.DismMHAttentionV3
    task.DismMHAttentionV3=lambda dm,heads,vocab,hd,**kw:cls(dm,heads,512,hd,**kw)
    torch.set_default_device('cuda')
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32=False
    b,n,dm= a.batch,1024,256
    token=torch.nn.Embedding(128,dm)
    blocks=[task.DismTransformerBlock(dm) for _ in range(3)]
    model=torch.nn.Sequential(token,*blocks,torch.nn.RMSNorm(dm),torch.nn.Linear(dm,128))
    gen=torch.Generator(device='cuda').manual_seed(777)
    for block in blocks: block.attn.gen=gen
    data=torch.Generator(device='cuda').manual_seed(12345)
    opt=torch.optim.AdamW([
        dict(params=[x for x in model.parameters() if hasattr(x,'_no_weight_decay')],weight_decay=0.),
        dict(params=[x for x in model.parameters() if not hasattr(x,'_no_weight_decay')],weight_decay=.01)
    ],lr=.005)
    sched=task._build_cosine_warmup_scheduler(opt,1000,50,.1)
    meta=dict(backend=a.backend,tile_lse=TILE_LSE,batch=b,n=n,layers=3,heads=4,d=64,dv=64,qk_vocab=512,
        d_model=dm,hard_prob=a.hard_prob,sm_scale=1,parameters=sum(x.numel() for x in model.parameters()),
        gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,
        scope='wall-clock synchronized full step: data, zero_grad, forward, CE, backward, clipping, AdamW, scheduler; no eval/logging/compilation',
        warmup=a.warmup,steps=a.steps)
    if a.profile:
        meta['scope']='CUPTI per-launch GPU duration within full training steps; excludes host launch gaps'
    print(json.dumps(dict(kind='config',**meta)),flush=True)
    def step():
        x=torch.randint(0,127,(b,n//2),generator=data)
        tokens=torch.cat((x,x),-1)
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',torch.bfloat16): logits=model(tokens)
        pred=logits[:,n//2-1:-1,:]
        loss=torch.nn.functional.cross_entropy(pred.reshape(-1,128),tokens[:,n//2:].flatten())
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        opt.step();sched.step()
        return loss.detach(),norm
    try:
        for i in range(a.warmup):
            loss,norm=step();torch.cuda.synchronize()
            print(json.dumps(dict(kind='warmup',step=i,loss=float(loss))),flush=True)
        torch.cuda.reset_peak_memory_stats()
        if a.profile:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as prof:
                for _ in range(a.steps): step()
                torch.cuda.synchronize()
            groups=collections.defaultdict(list)
            for event in sorted(prof.events(),key=lambda e:e.time_range.start):
                if event.device_type==torch.autograd.DeviceType.CUDA and 'dism_v2::' in event.name:
                    groups[event.name].append(event.time_range.elapsed_us())
            for name,times in groups.items():
                assert len(times)==3*a.steps,(name,len(times))
                print(json.dumps(dict(kind='kernel',name=name,count=len(times),
                    median_us=statistics.median(times),mean_us=statistics.mean(times),
                    min_us=min(times),max_us=max(times),
                    occurrence_medians_us=[statistics.median(times[j::3]) for j in range(3)],
                    launch_times_us=times)),flush=True)
            return
        samples=[]
        for i in range(a.steps):
            torch.cuda.synchronize();t=time.perf_counter()
            loss,norm=step();torch.cuda.synchronize()
            seconds=time.perf_counter()-t;samples.append(seconds)
            print(json.dumps(dict(kind='sample',step=i,seconds=seconds,loss=float(loss),grad_norm=float(norm))),flush=True)
        med=statistics.median(samples)
        print(json.dumps(dict(kind='result',**meta,median_s=med,min_s=min(samples),max_s=max(samples),
            input_tokens_s=b*n/med,supervised_tokens_s=b*n/2/med,sequences_s=b/med,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())),flush=True)
    except torch.cuda.OutOfMemoryError as e:
        print(json.dumps(dict(kind='oom',**meta,error=str(e),
            peak_allocated_bytes=torch.cuda.max_memory_allocated())),flush=True)
        raise SystemExit(2)


if __name__=='__main__':main()
