"""CPU-only tau tangent/covariance checks; no production kernel or model edits.

Differentiates the maintained thresholded-softplus recurrence using local
SQ/SK/V/dO fixtures. FP32 alternatives are arithmetic diagnostics, not CUDA
kernel accuracy or speed measurements. BF16-delta uses FP64 weights to isolate
only the saved-output rounding effect.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def recurrence(gate, tau):
    n=gate.shape[0]
    previous=torch.full_like(gate[0],-torch.inf)
    tangent=torch.zeros_like(previous)
    rows=[]; derivatives=[]
    for i in range(n):
        s=F.pad(previous[:-1],(1,0),value=-torch.inf)
        h=F.pad(tangent[:-1],(1,0))
        alpha=torch.where(s>20,torch.ones_like(s),torch.sigmoid(s))
        previous=gate[i]+tau+F.softplus(s)
        tangent=1+alpha*h
        rows.append(previous);derivatives.append(tangent)
    score=torch.stack(rows).masked_fill(~torch.ones_like(gate,dtype=torch.bool).tril(),-torch.inf)
    derivative=torch.stack(derivatives)
    maximum=score.amax(-1).clamp_min(0.).detach()
    weight=(score-maximum[:,None]).exp();fallback=(-maximum).exp()
    return weight,fallback,derivative


def centered_covariance(weight, fallback, x, h):
    # Include zero-value, zero-tangent fallback as a real weighted observation.
    w=torch.cat((weight,fallback[:,None]),1)
    x=F.pad(x,(0,1));h=F.pad(h,(0,1))
    p=w/w.sum(-1,keepdim=True)
    mx=(p*x).sum(-1,keepdim=True);mh=(p*h).sum(-1,keepdim=True)
    return (p*(x-mx)*(h-mh)).sum(-1)


def fixture(mode, n, tau, seed):
    rng=torch.Generator().manual_seed(seed)
    dtype=torch.float64
    sq,sk,v,do=[torch.randn(n,4,generator=rng,dtype=dtype)*.3 for _ in range(4)]
    if mode.startswith('soft'):
        scale=1 if mode=='soft' else 6
        q,k=[torch.randn(n,16,generator=rng,dtype=dtype)*scale for _ in range(2)]
        gate=q.log_softmax(-1)@k.softmax(-1).T
    else:
        q,k=[torch.randint(0,4,(n,),generator=rng) for _ in range(2)]
        if mode=='hard_repeated':q[:]=0;k[:]=0
        gate=torch.where(q[:,None]==k[None,:],0.,-torch.inf).to(dtype)
    t=torch.tensor(tau,dtype=dtype,requires_grad=True)
    weight,fallback,h=recurrence(gate,t)
    p=weight/(weight.sum(-1)+fallback)[:,None]
    readout=sq@sk.T
    out=(p*readout)@v
    loss=(out*do).sum()
    reference=torch.autograd.grad(loss,t)[0].item()
    weight,fallback,h=weight.detach(),fallback.detach(),h.detach()
    x=readout*(do@v.T)
    tangent64=centered_covariance(weight,fallback,x,h).sum().item()
    w32,f32,h32=recurrence(gate.float(),torch.tensor(tau,dtype=torch.float32))
    tangent32=centered_covariance(w32,f32,x.float(),h32).double().sum().item()
    rounded_delta=(out.detach().bfloat16().double()*do).sum(-1)
    rounded=((p.detach()*(x-rounded_delta[:,None]))*h).sum().item()
    eps=1e-4
    losses=[]
    for step in (-eps,eps):
        w,f,_=recurrence(gate,t.detach()+step)
        losses.append(((w/(w.sum(-1)+f)[:,None])*x).sum().item())
    fd=(losses[1]-losses[0])/(2*eps)
    assert abs(tangent64-reference)<1e-9*max(1.,abs(reference)),(mode,n,tau,reference,tangent64)
    return dict(mode=mode,n=n,tau=tau,reference=reference,jvp_fp64=tangent64,
        jvp_fp32=tangent32,bf16_delta_isolated=rounded,finite_difference=fd,
        abs_errors={name:abs(value-reference) for name,value in
            [('jvp_fp64',tangent64),('jvp_fp32',tangent32),('bf16_delta_isolated',rounded),('finite_difference',fd)]})


def row_examples():
    # CPU analogues of the shifted hard moments vs weighted centering. The
    # uniform case has an exact zero derivative up to an underflowed fallback.
    result=[]
    for name,n,lengths in [('uniform_long',4096,[2048]*64),('short_at_late_query',16384,[1]*1000+[2]*8)]:
        tau=.7;l=torch.tensor(lengths,dtype=torch.float64)
        logw=l*tau+torch.log(-torch.expm1(-l*tau))-torch.log(-torch.expm1(torch.tensor(-tau,dtype=torch.float64)))
        g=l+1/(torch.expm1(torch.tensor(tau,dtype=torch.float64)))*(-1)+l/torch.expm1(l*tau)
        g=torch.where(l*tau>700,l-1/torch.expm1(torch.tensor(tau,dtype=torch.float64)),g)
        maxlog=logw.max();w=(logw-maxlog).exp()[None,:];f=(-maxlog).exp()[None]
        x=torch.linspace(.11,1.03,len(lengths),dtype=torch.float64)[None,:]
        h=g[None,:]
        ref=centered_covariance(w,f,x,h).sum().item()
        p=(w/(w.sum()+f)).float();f32=(f/(w.sum()+f)).float()
        c=torch.tensor(n-1/np.expm1(tau),dtype=torch.float32)
        shifted=h.float()-c
        z=p.sum()+f32.sum();u=(p*x.float()).sum();v=(p*shifted).sum()-f32.sum()*c
        ww=(p*x.float()*shifted).sum()
        shifted_result=((ww-u*v/z)/z).item()
        centered=centered_covariance(w.float(),f.float(),x.float(),h.float()).double().sum().item()
        result.append(dict(name=name,reference=ref,shifted_fp32_analogue=shifted_result,centered_fp32=centered))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():p.error('output already exists')
    torch.set_num_threads(1)
    cases=[fixture(mode,n,tau,23) for mode in ('soft','soft_sharp','hard_random','hard_repeated')
           for n in (65,257) for tau in (.3,2.)]
    result=dict(note=__doc__,cases=cases,row_examples=row_examples())
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
    for mode in ('soft','soft_sharp','hard_random','hard_repeated'):
        rows=[r for r in cases if r['mode']==mode]
        print(mode,{k:max(r['abs_errors'][k] for r in rows) for k in rows[0]['abs_errors']})
    print('rows',result['row_examples'])


if __name__=='__main__':main()
