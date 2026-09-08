"""Fixed-input full-vs-tanh output/six-gradient diagnostic (not an oracle test).

DISM_TILE_LSE=full python -m dism_v2.compare_tile_lse --baseline /tmp/lse-full.pt
DISM_TILE_LSE=tanh python -m dism_v2.compare_tile_lse --baseline /tmp/lse-full.pt

Run in separate processes, without other GPU work. Full mode writes the baseline;
tanh mode reads it and prints JSON metrics. Existing precision tests stay intact.
"""
import argparse
import itertools
import json
import math
from pathlib import Path

import torch
from .autograd import voc_dism
from .kernel_config import TILE_LSE


def metric(actual, expected):
    a,b=actual.double().flatten(),expected.double().flatten()
    norm=float(b.norm())
    anorm=float(a.norm())
    return dict(max_abs=float((a-b).abs().max()),
        relative_l2=float((a-b).norm())/max(norm,1e-30),
        cosine=(float(torch.nn.functional.cosine_similarity(a,b,dim=0))
                if norm and anorm else float(norm==anorm)),
        finite=bool(torch.isfinite(a).all()))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--case',help='One case key, e.g. 128/32/139/q_from_k/0.0')
    p.add_argument('--tau-oracle',action='store_true',help='Same-embedding FP32 tau oracle for a selected pure-soft case')
    args=p.parse_args()
    if args.tau_oracle and (not args.case or not args.case.endswith('/0.0')):
        p.error('--tau-oracle requires one pure-soft --case')
    torch.backends.cuda.matmul.allow_tf32=False
    if TILE_LSE=='full' and args.baseline.exists():
        p.error('baseline already exists; choose a new path')
    expected=torch.load(args.baseline,weights_only=True) if TILE_LSE!='full' else None
    saved={}
    shapes=list(itertools.product((32,64,128),repeat=2))
    cases=[(d,dv,139) for d,dv in shapes]+[(64,64,n) for n in (513,1024)]
    for d,dv,n in cases:
        for direction,prob in itertools.product(('q_from_k','k_from_q'),(0.,.37,1.)):
            key=f'{d}/{dv}/{n}/{direction}/{prob}'
            if args.case and args.case!=key:
                continue
            torch.manual_seed(731+d+dv+n)
            def rand(shape):
                return torch.randn(shape,device='cuda',dtype=torch.bfloat16).requires_grad_()
            q,k=rand((1,2,n,d)),rand((1,2,n,d))
            v=rand((1,2,n,dv))
            qv,kv=rand((2,65,d)),rand((2,65,d))
            tau=torch.full((2,),math.log(d),device='cuda',requires_grad=True)
            inputs=(q,k,v,tau,qv,kv)
            do=torch.randn_like(v)
            gen=torch.Generator(device='cuda').manual_seed(932)
            out=voc_dism(*inputs,sm_scale=d**-.5,direction=direction,hard_prob=prob,
                generator=gen,embedding_backend='cuda',embedding_backward_backend='cuda')
            grads=torch.autograd.grad(out,inputs,do)
            tensors=[t.detach().float().cpu() for t in (out,*grads)]
            if expected is None:
                saved[key]=tensors
            else:
                metrics={name:metric(a,b) for name,a,b in
                    zip(('out','dq','dk','dv','dtau','dq_vocab','dk_vocab'),tensors,expected[key])}
                a,b=tensors[4],expected[key][4]
                relevant=b.abs()>1e-5
                metrics['dtau']['sign_flips']=int(((a*b<0)&relevant).sum())
                metrics['dtau']['nonzero_heads']=int(relevant.sum())
                if args.tau_oracle:
                    from .embedding import forward as emb_forward
                    from .dism_ref import InterpolationResult, voc_dism_ref
                    with torch.no_grad():
                        interp=InterpolationResult(*emb_forward(q,k,qv,kv,d**-.5))
                    tr=tau.detach().clone().requires_grad_()
                    ref=voc_dism_ref(q.detach().float(),k.detach().float(),v.detach().float(),
                        tr,qv.detach().float(),kv.detach().float(),sm_scale=d**-.5,
                        direction=direction,hard_prob=0.,interpolation=interp)
                    dt=torch.autograd.grad(ref,tr,do.float())[0]
                    metrics['dtau'].update(full_values=b.tolist(),tanh_values=a.tolist(),
                        same_embedding_oracle=dt.tolist())
                print(json.dumps(dict(case=key,metrics=metrics)),flush=True)
    if expected is None:
        torch.save(saved,args.baseline)
        print(json.dumps(dict(tile_lse=TILE_LSE,cases=len(saved),baseline=str(args.baseline))))


if __name__=='__main__':
    main()
