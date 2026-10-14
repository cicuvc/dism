"""Full standalone hard CUDA forward; excludes embedding, backward and planner."""
import argparse
import json

import torch

from bench_training import measure, profile_kernels, gpu_status
from flash_dism.hard_forward import forward_hard, hard_readout, hard_log_weights
from flash_dism.summary import summarize_hard, hard_chunk_scan
from flash_dism.varlen import VarlenLayout


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--profile',action='store_true')
    args=parser.parse_args()
    torch.manual_seed(417)
    b,n,h,r,dv=16,2048,6,32,64
    results=[]
    before=gpu_status()
    for packed in (False,True):
        for pattern in ('random','equal'):
            bn=(1,b*n) if packed else (b,n)
            sq,sk=[(torch.randn(*bn,h,r,device='cuda')*.3).bfloat16() for _ in range(2)]
            v=torch.randn(*bn,h,dv,device='cuda').bfloat16()
            iq,ik=[torch.randint(512,(bn[0],h,bn[1]),device='cuda',dtype=torch.int32) for _ in range(2)]
            if pattern=='equal':
                iq.zero_(); ik.zero_()
            hard=torch.ones(b,h,device='cuda',dtype=torch.bool)
            tau=torch.full((h,),.5,device='cuda')
            layout=None
            if packed:
                offsets=[0]
                for _ in range(b):
                    for length in (256,512,256,1024): offsets.append(offsets[-1]+length)
                layout=VarlenLayout.from_cu_seqlens(torch.tensor(offsets,dtype=torch.int32),b*n)
            _,_,state=forward_hard(sq,sk,v,iq,ik,hard,tau,layout=layout,save_state=True)
            max_length=max(layout.lengths) if layout else n
            functions={
                'summary':lambda: summarize_hard(iq,ik,hard,layout=layout),
                'chunk':lambda: hard_chunk_scan(state['summary'],layout=layout,heads=h),
                'lut':lambda: hard_log_weights(tau,max_length),
                'readout':lambda: hard_readout(sq,sk,v,iq,ik,hard,state['boundary'],state['log_weights'],layout=layout),
                'full':lambda: forward_hard(sq,sk,v,iq,ik,hard,tau,layout=layout),
            }
            row=dict(layout='ragged' if packed else 'fixed',pattern=pattern)
            row['timing']={name:measure(fn,warmup=10,repeats=20,samples=7) for name,fn in functions.items()}
            if args.profile:
                row['instrumented']=profile_kernels(functions['full'],iterations=20)
            results.append(row)
    print(json.dumps(dict(shape=dict(batch=b,row_tokens=n,heads=h,r=r,dv=dv),tau=.5,hard_fraction=1.,
                         gpu_before=before,gpu_after=gpu_status(),cases=results,
                         scope='summary + chunk + fresh LUT + readout, including initialization and launch gaps'),indent=2))


if __name__=='__main__': main()
