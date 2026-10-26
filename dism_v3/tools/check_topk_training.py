"""FP64 forward/backward oracle for the opt-in top-four training kernels."""
import argparse
import json
from pathlib import Path
import sys
import torch
from flash_dism.kernels import topk_hard as kernels


def oracle(sq,sk,v,iq,ik,ids,z,tau,segments):
    b,n,h,r=sq.shape
    s=sq.permute(0,2,1,3);k=sk.permute(0,2,1,3);value=v.permute(0,2,1,3)
    previous=sq.new_zeros((b,h,n));rows=[]
    pi=z.softmax(-1)
    for i in range(n):
        shift=torch.nn.functional.pad(previous[:,:,:-1],(1,0))
        if i:shift=shift*(segments[:,i,None,None]==segments[:,i-1,None,None])
        a=tau.exp()[None,:,None]*(1+shift)
        valid=(torch.arange(n,device=sq.device)[None,None,:]<=i)&(segments[:,i,None,None]==segments[:,None,:])
        a=a*valid
        similarity=(s[:,:,i,None,:]*k).sum(-1)
        o=torch.zeros_like(value[:,:,0])
        for slot in range(ids.shape[-1]):
            w=a*(ik==ids[:,:,i,slot,None])
            o=o+pi[:,:,i,slot,None]*((w*similarity)[...,None]*value).sum(-2)/(1+w.sum(-1,keepdim=True))
        rows.append(o)
        previous=a*(iq[:,:,i,None]==ik)
    return torch.stack(rows,2).permute(0,2,1,3)


def main():
    p=argparse.ArgumentParser();p.add_argument('--length',type=int,default=256)
    p.add_argument('--output',default='dism_v3/build/topk_training/check.json')
    a=p.parse_args();torch.set_num_threads(2);torch.manual_seed(551)
    torch.set_float32_matmul_precision('highest')
    b,n,h,r,d,c=1,a.length,2,32,64,8
    reports=[]
    for same in (False,True):
        sq=(torch.randn(b,n,h,r,device='cuda')*.2).bfloat16()
        sk=(torch.randn_like(sq.float())*.2).bfloat16();v=torch.randn(b,n,h,d,device='cuda').bfloat16()
        z=torch.randn(b,h,n,c,device='cuda')
        vals,ids=z.topk(4,-1);ids=ids.int().contiguous();pi=vals.softmax(-1).contiguous()
        iq=ids[:,:,:,0].contiguous();ik=torch.randint(c,(b,h,n),device='cuda',dtype=torch.int32)
        if same:iq.zero_();ik.zero_();ids[:]=torch.arange(4,device='cuda')
        tau=torch.tensor([0.,.1],device='cuda')
        hard=torch.ones(b,h,device='cuda',dtype=torch.bool)
        seg=torch.zeros(b,n,device='cuda',dtype=torch.int32)
        if n>256:seg[:,256:]=1
        out,state=kernels.forward(sq,sk,v,iq,ik,ids,pi,tau,hard,seg)
        do=torch.randn_like(v)
        grads=kernels.backward(sq,sk,v,do,iq,ik,ids,pi,hard,seg,state)
        inputs=[t.double().requires_grad_() for t in (sq,sk,v,tau,vals)]
        expected=oracle(*inputs[:3],iq,ik,ids,inputs[4],inputs[3],seg)
        eg=torch.autograd.grad((expected*do).sum(),inputs)
        row=dict(same=same,length=n,tensors={})
        for name,found,target in zip(('out','sq','sk','v','tau','logits'),(out,*grads),(expected,*eg)):
            x,y=found.detach().double().flatten(),target.detach().double().flatten()
            rel=float((x-y).norm()/y.norm().clamp_min(1e-12))
            cos=float(torch.nn.functional.cosine_similarity(x,y,dim=0))
            finite=bool(torch.isfinite(x).all())
            row['tensors'][name]=dict(relative_l2=rel,cosine=cos,finite=finite)
            assert finite and rel<.025 and cos>.999, (name,row)
        reports.append(row);print(json.dumps(row),flush=True)
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(reports,indent=2)+'\n')


if __name__=='__main__':main()
