"""NCU target: skip 10 matching launches, collect the 11th summary kernel.

DISM_TILE_LSE=tanh python -m dism_v2.profile_forward_summary
Filter mangled name regex:.*summary_persistentILi64E.* (both LSE directions).
"""
import argparse
import json
import torch
from .embedding import forward as embedding
from .core import forward
from .kernel_config import TILE_LSE


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--direction',choices=('q_from_k','k_from_q'),default='q_from_k')
    parser.add_argument('--hard-prob',type=float,default=.5)
    parser.add_argument('--label-dtype',choices=('int32','int64'),default='int32')
    args=parser.parse_args()
    torch.manual_seed(0)
    q,k,v=[torch.randn(64,4,1024,64,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
    qv,kv=[torch.randn(4,512,64,device='cuda',dtype=torch.bfloat16) for _ in range(2)]
    tau=torch.full((4,),3.,device='cuda')
    emb=embedding(q,k,qv,kv,1.)
    if args.direction=='q_from_k': a,b,lse=q,emb[0],emb[3]
    else: a,b,lse=emb[1],k,emb[2]
    gen=torch.Generator(device='cuda').manual_seed(777)
    state=None
    for _ in range(11):
        dtype=torch.int32 if args.label_dtype=='int32' else torch.int64
        result=forward(a,b,v,lse,tau,emb[7].to(dtype),emb[6].to(dtype),sm_scale=1.,
            direction=args.direction,hard_prob=args.hard_prob,
            generator=gen if state is None else None,rng_state=state,
            return_rng_state=True,save_boundaries=True)
        state=result[-1]
    torch.cuda.synchronize()
    print(json.dumps(dict(tile_lse=TILE_LSE,batch=64,heads=4,n=1024,d=64,dv=64,vocab=512,
        direction=args.direction,hard_prob=args.hard_prob,tau=3.,scale=1.,
        label_dtype=args.label_dtype,finite_output=bool(torch.isfinite(result[0]).all()))),flush=True)


if __name__=='__main__':
    main()
