"""Independent FP64 oracle and checks for the RMSNorm readout."""
import torch
from flash_dism_v4.rmsnorm import rms_readout

def dense_oracle(q,k,qw,kw,v,tau,hard,segments,epsilon):
    """Small FP64 autograd oracle; explicit history and full vocabulary score."""
    ql=torch.einsum('bnhd,hcd->bnhc',q,qw).permute(0,2,1,3)
    kl=torch.einsum('bnhd,hcd->bnhc',k,kw).permute(0,2,1,3)
    logp=ql.log_softmax(-1)
    iq,ik=ql.argmax(-1),kl.argmax(-1)
    pk=torch.where(hard[:,:,None,None],torch.nn.functional.one_hot(ik,ql.shape[-1]).to(q),kl.softmax(-1))
    values=v.permute(0,2,1,3)
    prev=torch.zeros_like(iq,dtype=q.dtype)
    rows=[]
    for i in range(q.shape[1]):
        shifted=torch.nn.functional.pad(prev[...,:-1],(1,0))
        if i:
            shifted=shifted*(segments[:,i]==segments[:,i-1])[:,None,None]
        weight=tau.exp()[None,:,None]*(1+shifted)
        valid=(torch.arange(q.shape[1],device=q.device)<=i)[None,None,:]
        valid=valid & (segments[:,i,None]==segments)[:,None,:]
        weight=weight*valid
        m=weight*(pk*logp[:,:,i,None,:]).sum(-1).exp()
        u=(m[...,None]*values).sum(-2)
        rows.append(u*torch.rsqrt(u.square().mean(-1,keepdim=True)+epsilon))
        prev=weight*(iq[:,:,i,None]==ik)
    return torch.stack(rows,2).permute(0,2,1,3)


def validate():
    """Finite-epsilon, mixed heads, packed boundaries and long equal-symbol chains."""
    torch.manual_seed(2026)
    for sharp in (False,True):
        for epsilon in (1e-6,.1):
            b,n,h,d,c=2,73,2,4,7
            q=torch.randn(b,n,h,d,device='cuda')*.25
            k=torch.randn_like(q)*.25
            if sharp:
                q[:]=.2;k[:]=.3
            inputs=[q,k,torch.randn(h,c,d,device='cuda')*.3,
                    torch.randn(h,c,d,device='cuda')*.3,
                    torch.randn(b,n,h,5,device='cuda'),torch.tensor([.0,.15],device='cuda')]
            if sharp:
                inputs[2][:]=0;inputs[3][:]=0
                inputs[2][:,0]=1;inputs[3][:,0]=1
            inputs=[t.requires_grad_() for t in inputs]
            oracle=[t.detach().double().requires_grad_() for t in inputs]
            hard=torch.tensor([[True,False],[False,True]],device='cuda')
            seg=torch.zeros(b,n,device='cuda',dtype=torch.int32);seg[:,37:]=1
            a=rms_readout(*inputs,hard=hard,segment_ids=seg,block_size=32,epsilon=epsilon)
            e=dense_oracle(*oracle,hard,seg,epsilon)
            cot=torch.randn_like(a)
            ga=torch.autograd.grad((a*cot).sum(),inputs)
            ge=torch.autograd.grad((e*cot).sum(),oracle)
            torch.testing.assert_close(a.double(),e,atol=3e-6,rtol=3e-5)
            for x,y in zip(ga,ge):
                torch.testing.assert_close(x.double(),y,atol=5e-5,rtol=3e-4)
    print('FP64 oracle: 4 forward/backward cases passed',flush=True)


if __name__=='__main__':
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    validate()
