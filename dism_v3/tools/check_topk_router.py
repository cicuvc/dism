"""Independent BF16-operand router gradient check, including soft-row masking."""
import json
import torch
from flash_dism.kernels import topk_hard as kernels


def main():
    torch.manual_seed(42);torch.set_num_threads(2)
    reports=[]
    for d,c in ((32,4),(32,8),(64,512)):
        b,n,h=2,256,3;beta=.7
        q=torch.randn(b,n,h,d,device='cuda',dtype=torch.bfloat16)
        eq=torch.randn(h,c,d,device='cuda',dtype=torch.float32)
        qref=q.double().requires_grad_();eref=eq.bfloat16().double().requires_grad_()
        z=torch.einsum('bnhd,hcd->bhnc',qref,eref)*beta
        iq=z.argmax(-1).int().contiguous()
        hard=torch.tensor([[1,0,1],[0,1,0]],device='cuda',dtype=torch.bool)
        ids,pi=kernels.select(q,eq,iq,hard,beta)
        expected_ids=z.argsort(dim=-1,descending=True)[...,:4]
        assert torch.equal(ids[hard],expected_ids[hard].int())
        p=z.gather(-1,ids.long()).softmax(-1)*hard[:,:,None,None]
        torch.testing.assert_close(pi,p.float(),atol=2e-6,rtol=2e-5)
        g=torch.randn_like(pi)
        dz=pi*(g-(g*pi).sum(-1,keepdim=True))
        actual=kernels.router_backward(q,eq,ids,dz,hard,beta)
        expected=torch.autograd.grad((p*g).sum(),(qref,eref))
        for name,got,want in zip(('q','q_vocab'),actual,expected):
            rel=((got.double()-want).norm()/want.norm()).item()
            assert rel<1e-5,(name,d,c,rel)
            reports.append(dict(d=d,c=c,name=name,relative_l2=rel))
    print(json.dumps(reports,indent=2))


def check_vocabulary(sampled=False):
    """Packed all-hard integration, with an independent FP64 output/gradient."""
    if sampled:
        from check_sampled_training import oracle
        from flash_dism.kernels import sampled_hard
    else:
        from check_topk_training import oracle
    from flash_dism.topk_training import vocabulary
    from flash_dism.kernels.interp_true import emb_fwd_true
    torch.manual_seed(63);torch.set_num_threads(2)
    n,h,d,r,dv,c=512,2,32,16,32,8
    q=(torch.randn(1,n,h,d,device='cuda')*.2).bfloat16().requires_grad_()
    key=torch.randn_like(q).requires_grad_()
    sq=(torch.randn(1,n,h,r,device='cuda')*.2).bfloat16().requires_grad_()
    sk=torch.randn_like(sq).requires_grad_()
    v=torch.randn(1,n,h,dv,device='cuda',dtype=torch.bfloat16).requires_grad_()
    eq=torch.randn(h,c,d,device='cuda',requires_grad=True);ek=torch.randn_like(eq,requires_grad=True)
    tau=torch.tensor([0.,.1],device='cuda',requires_grad=True)
    hard=torch.ones(2,h,device='cuda',dtype=torch.bool);direction=hard[:1].contiguous()
    seg=torch.zeros(1,n,device='cuda',dtype=torch.int32);seg[:,256:]=1
    bounds=torch.tensor([0,256,512],device='cuda',dtype=torch.int32)
    args=(q,key,sq,sk,v,eq,ek,tau)
    counter=torch.tensor(37,device='cuda',dtype=torch.int64)
    out=vocabulary(*args,direction=direction,hard=hard,segments=seg,boundaries=bounds,
                   sampled=sampled,counter=counter,seed=1729)
    g=torch.randn_like(out);actual=torch.autograd.grad(out,args,g)
    with torch.no_grad():
        *_,ik,iq=emb_fwd_true(q,key,eq.bfloat16(),ek.bfloat16(),1.,tau.detach())
        labels=lambda x:x.transpose(1,2).reshape(2,256,h).transpose(1,2).contiguous()
        iq,ik=labels(iq),labels(ik)
        ids,pi=kernels.select(q.reshape(2,256,h,d),eq,iq,hard,1.)
        if sampled:draw=sampled_hard.sample(pi,counter,1729)
    ref=[x.detach().double().requires_grad_() for x in (q,sq,sk,v,eq.bfloat16(),tau)]
    rq,rsq,rsk,rv,req,rt=ref
    z=torch.einsum('bnhd,hcd->bhnc',rq.reshape(2,256,h,d),req).gather(-1,ids.long())
    result=oracle(rsq.reshape(2,256,h,r),rsk.reshape(2,256,h,r),rv.reshape(2,256,h,dv),
                  iq,ik,ids,z,rt,seg.reshape(2,256),*([draw] if sampled else []))
    if sampled:
        want,residual=result;want=want.reshape_as(out)
        logpi=z.log_softmax(-1).gather(-1,draw.long()[...,None]).squeeze(-1)
        reward=(residual*g.reshape_as(residual)).sum(-1).permute(0,2,1).detach()
        loss=(want*g).sum()+(reward*(logpi-logpi.detach())).sum()
        grads=torch.autograd.grad(loss,ref)
    else:
        want=result.reshape_as(out);grads=torch.autograd.grad(want,ref,g.double())
    assert actual[1].count_nonzero()==0 and actual[6].count_nonzero()==0
    report={}
    for name,x,y in zip(('out','q','sq','sk','v','q_vocab','tau'),
                         (out,actual[0],actual[2],actual[3],actual[4],actual[5],actual[7]),(want,*grads)):
        rel=((x.detach().double()-y.detach()).norm()/y.detach().norm()).item()
        assert rel<.006,(name,rel)
        report[name]=rel
    print('packed vocabulary relative L2:',json.dumps(report))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--sampled',action='store_true');args=parser.parse_args()
    main()
    check_vocabulary(args.sampled)
