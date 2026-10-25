"""Exercise history bounds and export compiled resources for sanitizer/SASS checks."""
import argparse
import json
from pathlib import Path

import torch
import triton
from flash_dism_v4.history import _history


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='dism_v4/build/history')
    args=parser.parse_args()
    destination=Path(args.output)
    destination.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(9)
    for block in (32,64,128):
        for n in (1,31,73,129,2048):
            q=torch.randint(3,(2,n),dtype=torch.int32).cuda()
            k=torch.randint(3,(2,n),dtype=torch.int32).cuda()
            seg=torch.zeros((2,n),dtype=torch.int32)
            seg[:,n//2:]=1
            reset=torch.zeros((2,n),dtype=torch.bool)
            reset[:,n//3]=True
            seg,reset=seg.cuda(),reset.cuda()
            tau=torch.tensor([0.,.03],device='cuda')
            prev=torch.zeros((2,n),dtype=torch.int32).cuda()
            for start in range(0,n,block):
                after=torch.zeros((2,n),dtype=torch.int32).cuda()
                loga=torch.full((2,block,n),-torch.inf).cuda()
                derivative=torch.zeros((2,block,n)).cuda()
                kernel=_history[(2,triton.cdiv(n,32))](
                    q,k,seg,reset,tau,prev,after,loga,derivative,
                    n,start,block,32,num_warps=8)
                assert not bool(torch.isnan(loga.cpu()).any())
                assert bool(torch.isfinite(derivative.cpu()).all())
                prev=after
            torch.cuda.synchronize()
            if n==2048:
                result=dict(block=block,registers=kernel.n_regs,spills=kernel.n_spills,
                            shared_bytes=kernel.metadata.shared)
                print(json.dumps(result),flush=True)
                (destination/f'history_b{block}.cubin').write_bytes(kernel.asm['cubin'])
                (destination/f'history_b{block}.ptx').write_text(kernel.asm['ptx'])
    print('history bounds cases complete',flush=True)


if __name__=='__main__':
    main()
