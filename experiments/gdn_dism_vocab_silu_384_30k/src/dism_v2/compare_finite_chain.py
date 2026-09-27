"""Save/compare guarded-tanh and finite-tanh long-chain states, no dense W."""
import argparse
import json
import math
import torch
from .core import forward
from .kernel_config import TILE_LSE
from .compare_tile_lse import metric


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--baseline',required=True)
    args=parser.parse_args()
    expected=None if TILE_LSE=='tanh' else torch.load(args.baseline,weights_only=True)
    saved={}
    for n in (1025,8193):
        for t in (-1.,0.,math.log(32)):
            torch.manual_seed(901)
            q=torch.zeros((1,1,n,32),device='cuda',dtype=torch.bfloat16)
            v=torch.randn((1,1,n,64),device='cuda',dtype=torch.bfloat16)
            labels=torch.zeros((1,1,n),device='cuda',dtype=torch.int64)
            lse=torch.zeros((1,1,n),device='cuda')
            tau=torch.tensor([t],device='cuda')
            o,l,s,b=forward(q,q,v,lse,tau,labels,labels,sm_scale=1.,
                direction='q_from_k',hard_prob=1.,return_debug=True)
            rows=torch.arange(n//32,device='cuda')*32+31
            cols=torch.arange(b.shape[-1],device='cuda')
            valid=cols[None,:]<=rows[:,None]
            values=[x.float().cpu() for x in (o,l,b[0,0,:n//32][valid])]
            key=f'{n}/{t}'
            if expected is None:
                saved[key]=values
            else:
                print(json.dumps(dict(case=key,metrics={name:metric(a,b) for name,a,b
                    in zip(('out','norm','reachable_boundary'),values,expected[key])})),flush=True)
    if expected is None:
        from pathlib import Path
        if Path(args.baseline).exists():
            raise FileExistsError(args.baseline)
        torch.save(saved,args.baseline)
        print(json.dumps(dict(cases=len(saved),baseline=args.baseline)))


if __name__=='__main__': main()
