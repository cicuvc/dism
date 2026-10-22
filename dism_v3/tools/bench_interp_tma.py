"""Compare separate dQ/dKV against joint dKV with FP32 TMA reduction of dQ."""
import argparse
from functools import partial
import json
from pathlib import Path
import statistics
import subprocess
import torch
import triton
from flash_dism.kernels.interp_true import emb_fwd_true
from flash_dism.kernels.interp_bwd_split import emb_bwd_split, _dq, _dkv
from flash_dism.kernels.interp_bwd_tma import _dkv_dq_tma


def capture(fn):
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs=fn()
    return graph,outputs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokens',type=int,default=16384)
    parser.add_argument('--rounds',type=int,default=7)
    parser.add_argument('--replays',type=int,default=100)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]/'build'
    if args.output.exists() or not args.output.resolve().is_relative_to(root):
        parser.error('output must be a fresh path under dism_v3/build')
    torch.set_num_threads(1)
    torch.manual_seed(1741)
    n,h,d,v=args.tokens,12,64,512
    def rand(shape):
        return torch.nn.functional.silu(torch.randn(shape,device='cuda')).bfloat16()
    q,k=[rand((1,n,h,d)) for _ in range(2)]
    eq,ek=[rand((h,v,d)) for _ in range(2)]
    tau=torch.full((h,),.5,device='cuda')
    saved=emb_fwd_true(q,k,eq,ek,1.,tau)
    doq=torch.randn_like(q)*.1
    dlq,dlk=[torch.randn(1,h,n,device='cuda')*.1 for _ in range(2)]
    values=(q,k,eq,ek,*saved[:4],doq,q,dlq,dlk,1.,tau)
    functions={name:partial(emb_bwd_split,*values,direction_true=True,fuse_dq_tma=fused)
               for name,fused in (('split',False),('tma',True))}
    expected=functions['split']()
    actual=functions['tma']()
    accuracy=[]
    for a,b in zip(actual,expected):
        x,y=a.double().flatten(),b.double().flatten()
        relative=((x-y).norm()/y.norm().clamp_min(1e-12)).item()
        accuracy.append(dict(relative_l2=relative,cosine=torch.nn.functional.cosine_similarity(x,y,dim=0).item(),
                             norm_ratio=(x.norm()/y.norm().clamp_min(1e-12)).item()))
        assert relative < 1e-4,accuracy
    selected=[]
    original_run=_dkv_dq_tma.fn.run
    def record_selected(*a,**kw):
        kernel=original_run(*a,**kw)
        selected.append(kernel)
        return kernel
    _dkv_dq_tma.fn.run=record_selected
    try:
        functions['tma']()
    finally:
        _dkv_dq_tma.fn.run=original_run
    graphs={name:capture(fn) for name,fn in functions.items()}
    times={name:[] for name in graphs}
    for index in range(args.rounds):
        order=list(graphs) if index%2==0 else list(reversed(graphs))
        for name in order:
            graph=graphs[name][0]
            for _ in range(10):graph.replay()
            start,stop=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
            start.record()
            for _ in range(args.replays):graph.replay()
            stop.record()
            stop.synchronize()
            times[name].append(start.elapsed_time(stop)/args.replays)
    report=dict(tokens=n,heads=h,dim=d,vocab=v,accuracy=accuracy,times_ms=times,
                median_ms={k:statistics.median(v) for k,v in times.items()},
                configs={t.fn.__name__:[str(c) for c in t.cache.values()] for t in (_dq,_dkv,_dkv_dq_tma)})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    binary_dir=args.output.with_suffix('')
    binary_dir.mkdir(exist_ok=True)
    report['resources']=[]
    for i,kernel in enumerate(_dkv_dq_tma.fn.device_caches[torch.cuda.current_device()][0].values()):
        unsupported = None
        try:
            kernel._init_handles()
        except triton.runtime.errors.OutOfResources as error:
            unsupported = str(error)
        path=binary_dir/f'tma_{i}.cubin'
        path.write_bytes(kernel.asm['cubin'])
        path.with_suffix('.ptx').write_text(kernel.asm['ptx'])
        sass=subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','-sass',str(path)],text=True)
        path.with_suffix('.sass').write_text(sass)
        report['resources'].append(dict(index=i,selected=any(kernel is k for k in selected),
            registers=getattr(kernel,'n_regs',None),
            spills=getattr(kernel,'n_spills',None),unsupported=unsupported,
            shared=kernel.metadata.shared,tma_reduce=sass.count('UTMAREDG'),
            calls=sum(' CALL' in line for line in sass.splitlines()),metadata=str(kernel.metadata)))
    args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
