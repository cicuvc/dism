"""Small graph replay and machine-code audit; suitable for Compute Sanitizer.

No datasets, checkpoints or external jobs. Artifacts belong under build/.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess
import torch
from flash_dism.kernels import topk_hard as k


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cuobjdump',default='/usr/local/cuda/bin/cuobjdump');a=p.parse_args()
    torch.set_num_threads(2);torch.manual_seed(91)
    b,n,h,d,r,dv,c=2,512,2,64,32,64,512
    q=torch.randn(b,n,h,d,device='cuda',dtype=torch.bfloat16)
    eq=torch.randn(h,c,d,device='cuda')
    iq=torch.randint(c,(b,h,n),device='cuda',dtype=torch.int32);ik=torch.randint_like(iq,c)
    sq=torch.randn(b,n,h,r,device='cuda',dtype=torch.bfloat16);sk=torch.randn_like(sq)
    v=torch.randn(b,n,h,dv,device='cuda',dtype=torch.bfloat16);do=torch.randn_like(v)
    tau=torch.tensor([0.,.4],device='cuda');hard=torch.ones(b,h,device='cuda',dtype=torch.bool)
    seg=torch.zeros(b,n,device='cuda',dtype=torch.int32)
    def step():
        ids,pi=k.select(q,eq,iq,hard,1.)
        out,state=k.forward(sq,sk,v,iq,ik,ids,pi,tau,hard,seg)
        grads=k.backward(sq,sk,v,do,iq,ik,ids,pi,hard,seg,state)
        router=k.router_backward(q,eq,ids,grads[-1],hard,1.)
        return out,grads,router
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):step()
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):out,grads,router=step()
    for mode in range(3):
        if mode==1:seg[:,256:]=1;hard[:,1]=False
        if mode==2:hard.zero_()
        graph.replay();torch.cuda.synchronize()
        assert all(torch.isfinite(x).all() for x in (out,*grads,*router))
        if mode==2:assert all(torch.count_nonzero(x)==0 for x in (out,*grads,*router))
    a.output.mkdir(parents=True,exist_ok=True);report=[]
    for name in ('_select','_router_dq','_router_deq','_boundaries','_lut','_normalizers','_forward','_backward'):
        fn=getattr(k,name)
        for index,compiled in enumerate(fn.device_caches[0][0].values()):
            path=a.output/f'{name}_{index}.cubin';path.write_bytes(compiled.asm['cubin'])
            sass=subprocess.check_output([a.cuobjdump,'--dump-sass',str(path)],text=True)
            (a.output/f'{name}_{index}.sass').write_text(sass)
            calls=len(re.findall(r'\bCALL(?:\.[A-Z]+)*\b',sass))
            report.append(dict(kernel=name,registers=compiled.n_regs,spills=compiled.n_spills,
                               shared_bytes=compiled.metadata.shared,sass_calls=calls))
    (a.output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2));print('graph mutable segments/masks and zero-hard check passed')


if __name__=='__main__':main()
