"""Independent FP64 checks for sampled local hard reads and corrected VJPs."""
import argparse,json
from pathlib import Path
import torch
from flash_dism.kernels import sampled_hard as kernels


def oracle(sq,sk,v,iq,ik,ids,z,tau,segments,draw):
    b,n,h,r=sq.shape
    s=sq.permute(0,2,1,3);k=sk.permute(0,2,1,3);value=v.permute(0,2,1,3)
    previous=sq.new_zeros((b,h,n));rows=[];residuals=[]
    pi=z.softmax(-1)
    for i in range(n):
        shift=torch.nn.functional.pad(previous[:,:,:-1],(1,0))
        if i:shift=shift*(segments[:,i,None,None]==segments[:,i-1,None,None])
        a=tau.exp()[None,:,None]*(1+shift)
        valid=(torch.arange(n,device=sq.device)[None,None,:]<=i)&(segments[:,i,None,None]==segments[:,None,:])
        a=a*valid;base=tau.exp()[None,:,None]*valid
        similarity=(s[:,:,i,None,:]*k).sum(-1)
        mix=torch.zeros_like(value[:,:,0]);residual=torch.zeros_like(mix)
        for slot in range(4):
            match=ik==ids[:,:,i,slot,None]
            w=a*match;u=base*match
            hc=((w*similarity)[...,None]*value).sum(-2)/(1+w.sum(-1,keepdim=True))
            bc=((u*similarity)[...,None]*value).sum(-2)/(1+u.sum(-1,keepdim=True))
            mix=mix+pi[:,:,i,slot,None]*bc
            residual=residual+(draw[:,:,i,None]==slot)*(hc-bc)
        rows.append(mix+residual);residuals.append(residual)
        previous=a*(iq[:,:,i,None]==ik)
    return torch.stack(rows,2).permute(0,2,1,3),torch.stack(residuals,2).permute(0,2,1,3)


def verify_expectation():
    """Enumerate all four local draws: output and every VJP recover top-four."""
    from check_topk_training import oracle as full_oracle
    gen=torch.Generator().manual_seed(611)
    b,n,h,r,d,c=1,11,2,3,5,7
    rand=lambda *shape:torch.randn(*shape,generator=gen,dtype=torch.float64)
    sq=rand(b,n,h,r).requires_grad_();sk=rand(b,n,h,r).requires_grad_()
    v=rand(b,n,h,d).requires_grad_();tau=torch.tensor([0.,.2],dtype=torch.float64,requires_grad=True)
    z=rand(b,h,n,4).requires_grad_();inputs=(sq,sk,v,tau,z)
    ids=(torch.randint(c,(b,h,n,1),generator=gen)+torch.arange(4))%c
    iq=ids[...,0];ik=torch.randint(c,(b,h,n),generator=gen)
    seg=torch.zeros(b,n,dtype=torch.int32);seg[:,5:]=1;g=rand(b,n,h,d)
    expected=full_oracle(sq,sk,v,iq,ik,ids,z,tau,seg)
    want=torch.autograd.grad((expected*g).sum(),inputs)
    found=[torch.zeros_like(t) for t in inputs];mean=torch.zeros_like(expected)
    for slot in range(4):
        draw=torch.full((b,h,n),slot,dtype=torch.int32)
        o,residual=oracle(sq,sk,v,iq,ik,ids,z,tau,seg,draw)
        weight=z.detach().softmax(-1)[...,slot].permute(0,2,1)[...,None]
        gs=g*weight;mean+=o.detach()*weight
        logpi=z.log_softmax(-1)[...,slot]
        reward=(residual*gs).sum(-1).permute(0,2,1).detach()
        loss=(o*gs).sum()+(reward*(logpi-logpi.detach())).sum()
        grads=torch.autograd.grad(loss,inputs)
        for dst,src in zip(found,grads):dst+=src
    for x,y in zip((mean,*found),(expected,*want)):
        torch.testing.assert_close(x,y,atol=1e-11,rtol=1e-10)
    print('FP64 four-draw expectation: output and SQ/SK/V/tau/router gradients passed',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--length',type=int,default=256)
    p.add_argument('--rank',type=int,default=32);p.add_argument('--value-dim',type=int,default=64)
    p.add_argument('--output',type=Path)
    p.add_argument('--expectation-only',action='store_true')
    a=p.parse_args();torch.set_num_threads(2);verify_expectation()
    if a.expectation_only:return
    if a.output is None:p.error('--output is required for GPU checks')
    torch.manual_seed(551)
    torch.cuda.set_per_process_memory_fraction(.25)
    torch.set_float32_matmul_precision('highest')
    b,n,h,r,d,c=1,a.length,2,a.rank,a.value_dim,8
    reports=[]
    for same in (False,True):
        sq=(torch.randn(b,n,h,r,device='cuda')*.2).bfloat16()
        sk=(torch.randn_like(sq.float())*.2).bfloat16();v=torch.randn(b,n,h,d,device='cuda').bfloat16()
        z=torch.randn(b,h,n,c,device='cuda');vals,ids=z.topk(4,-1)
        ids=ids.int().contiguous();pi=vals.softmax(-1).contiguous()
        iq=ids[:,:,:,0].contiguous();ik=torch.randint(c,(b,h,n),device='cuda',dtype=torch.int32)
        if same:iq.zero_();ik.zero_();ids[:]=torch.arange(4,device='cuda')
        tau=torch.tensor([0.,.1],device='cuda');hard=torch.ones(b,h,device='cuda',dtype=torch.bool)
        seg=torch.zeros(b,n,device='cuda',dtype=torch.int32)
        if n>256:seg[:,256:]=1
        counter=torch.tensor(55,device='cuda',dtype=torch.int64)
        draw=kernels.sample(pi,counter,1729)
        out,state=kernels.forward(sq,sk,v,iq,ik,ids,pi,tau,hard,seg,draw)
        do=torch.randn_like(v)
        grads=kernels.backward(sq,sk,v,do,iq,ik,ids,pi,hard,seg,state,draw)
        inputs=[t.double().requires_grad_() for t in (sq,sk,v,tau,vals)]
        expected,residual=oracle(*inputs[:3],iq,ik,ids,inputs[4],inputs[3],seg,draw)
        logpi=inputs[4].log_softmax(-1).gather(-1,draw.long()[...,None]).squeeze(-1)
        # Router-only residual score term. SQ/SK/V/tau use the full output VJP.
        reward=(residual*do).sum(-1).permute(0,2,1).detach()
        loss=(expected*do).sum()+(reward*(logpi-logpi.detach())).sum()
        eg=torch.autograd.grad(loss,inputs)
        row=dict(same=same,length=n,rank=r,value_dim=d,tensors={})
        for name,found,target in zip(('out','sq','sk','v','tau','logits'),(out,*grads),(expected,*eg)):
            x,y=found.detach().double().flatten(),target.detach().double().flatten()
            rel=float((x-y).norm()/y.norm().clamp_min(1e-12));cos=float(torch.nn.functional.cosine_similarity(x,y,dim=0))
            row['tensors'][name]=dict(relative_l2=rel,cosine=cos,finite=bool(torch.isfinite(x).all()))
            assert bool(torch.isfinite(x).all()) and rel<.025 and cos>.999,(name,row)
        reports.append(row);print(json.dumps(row),flush=True)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(reports,indent=2)+'\n')


if __name__=='__main__':main()
