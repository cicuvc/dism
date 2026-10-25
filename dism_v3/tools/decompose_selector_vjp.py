"""Decompose hard cotangents and norm geometry at fixed soft DISM operands."""
import argparse,json,gc
from pathlib import Path
import torch
from flash_dism.compiler import voc_forward


def norm(x):return x.double().norm().item()
def compare(a,b):
    aa=a.double().flatten();bb=b.double().flatten()
    return dict(cosine=float(aa.dot(bb)/(aa.norm()*bb.norm()).clamp_min(1e-30)),ratio=norm(b)/max(norm(a),1e-30),rms=norm(b)/(b.numel()**.5))


def main():
    p=argparse.ArgumentParser();p.add_argument('dump');p.add_argument('--head',type=int,required=True);p.add_argument('--output',required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.28)
    d=torch.load(a.dump,map_location='cpu',weights_only=False);h=a.head;n=d['context']
    soft=d['g_soft'][0,:,h].double();hard=d['g_hard'][0,:,h].double()
    us=d['u_soft'][0,:,h].double();uh=d['u_hard'][0,:,h].double()
    ln=d['norm_type']=='layernorm_sigmoid'
    def tangent(g,u):
        if ln:u=u-u.mean(-1,keepdim=True);g=g-g.mean(-1,keepdim=True)
        return g-u*((g*u).sum(-1,keepdim=True)/u.square().sum(-1,keepdim=True).clamp_min(1e-60))
    c=float((soft*hard).sum()/soft.square().sum().clamp_min(1e-60))
    signals={'soft':soft,'hard':hard,'parallel':c*soft,'residual':hard-c*soft,
             'hard_soft_tangent':tangent(hard,us),'soft_soft_tangent':tangent(soft,us),
             'hard_soft_radial':hard-tangent(hard,us)}
    result=dict(layer=d['layer'],head=h,document=d['document'],context=n,norm_type=d['norm_type'],parallel_coefficient=c,
        cotangent={k:compare(soft,v) for k,v in signals.items()},
        removed_fraction={name:norm(g-tangent(g,u))/max(norm(g),1e-30) for name,g,u in [('soft_at_soft',soft,us),('hard_at_hard',hard,uh),('hard_at_soft',hard,us)]},
        hybrid_rms_soft=norm(us)/(us.numel()**.5),hybrid_rms_hard=norm(uh)/(uh.numel()**.5))
    vals,kw=d['inputs'];vals=[v.cuda() if torch.is_tensor(v) and i!=12 else v for i,v in enumerate(vals)]
    for i in [0,1,5,6]:vals[i].requires_grad_(True)
    gradients={}
    for name,g in signals.items():
        output=voc_forward(*vals,**kw)[0]
        cot=torch.zeros_like(output);cot[0,:,h]=g.to(device='cuda',dtype=cot.dtype)
        grads=torch.autograd.grad(output,[vals[i] for i in [0,1,5,6]],cot)
        gradients[name]={k:(v[h] if 'vocab' in k else v[0,:,h]).detach().float().cpu()*n for k,v in zip(['q','k','q_vocab','k_vocab'],grads)}
        del output,grads,cot;gc.collect()
    result['vjp']={}
    for k in gradients['soft']:
        ref=gradients['soft'][k];gh=gradients['hard'][k]
        result['vjp'][k]={name:compare(ref,grad[k]) for name,grad in gradients.items()}
        result['vjp'][k]['linearity_error']=norm(gh-gradients['parallel'][k]-gradients['residual'][k])/max(norm(gh),1e-30)
        result['vjp'][k]['radial_fraction_of_hard_vjp']=norm(gradients['hard_soft_radial'][k])/max(norm(gh),1e-30)
    torch.save(gradients,out/'production_gradients.pt')
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
