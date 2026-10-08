import pytest
import torch
from flash_dism.voc import voc_dism
from flash_dism.reference.dism_v3_ref import dism_ref
from test_backward_summary import pytestmark


def torch_voc(q,k,sq,sk,v,eq,ek,tau,direction,hard):
    eq,ek=eq.bfloat16().float(),ek.bfloat16().float()
    qs=torch.einsum('bnhd,hvd->bhnv',q.float(),eq)
    ks=torch.einsum('bnhd,hvd->bhnv',k.float(),ek)
    q_from_k=torch.einsum('bhnv,hvd->bnhd',ks.softmax(-1),eq).bfloat16()
    k_from_q=torch.einsum('bhnv,hvd->bnhd',qs.softmax(-1),ek).bfloat16()
    select=direction[:,None,:,None]
    return dism_ref(torch.where(select,q,k_from_q),torch.where(select,q_from_k,k),
                    sq,sk,qs.logsumexp(-1).transpose(1,2),ks.logsumexp(-1).transpose(1,2),
                    qs.argmax(-1),ks.argmax(-1),direction,hard,v,tau)


@pytest.mark.parametrize('mode',['soft','mixed','hard'])
@pytest.mark.parametrize('direction_value',[False,True])
@pytest.mark.strict_gradient
def test_triton_embedding_end_to_end(mode,direction_value):
    torch.manual_seed(273)
    b,n,h=1,65,2
    q,k=[(torch.randn(b,n,h,64,device='cuda')*.2).bfloat16() for _ in range(2)]
    sq,sk=[torch.nn.functional.silu(torch.randn(b,n,h,32,device='cuda')*.5).bfloat16()
           for _ in range(2)]
    v=torch.randn(b,n,h,64,device='cuda').bfloat16()
    eq,ek=[torch.randn(h,512,64,device='cuda')*.2 for _ in range(2)]
    tau=torch.full((h,),2.,device='cuda')
    direction=torch.full((b,h),direction_value,device='cuda',dtype=torch.bool)
    hard=torch.rand(b,h,n,device='cuda')<{'soft':0.,'mixed':.5,'hard':1.}[mode]
    args=[x.requires_grad_() for x in (q,k,sq,sk,v,eq,ek,tau)]
    do=torch.randn_like(v)
    actual=voc_dism(*args,direction=direction,hard=hard,ctas=1)
    ga=torch.autograd.grad(actual,args,do)
    expected=torch_voc(*args,direction,hard)
    ge=torch.autograd.grad(expected,args,do.float())
    torch.testing.assert_close(actual.float(),expected,atol=.008,rtol=.025)
    for name,x,y in zip(('q','k','sq','sk','v','eq','ek','tau'),ga,ge):
        x,y=x.float().flatten(),y.float().flatten()
        if y.norm()<1e-8:
            assert x.norm()<1e-6,name
        else:
            rel=(x-y).norm()/y.norm()
            cos=torch.nn.functional.cosine_similarity(x,y,dim=0)
            assert rel<.03,(name,rel.item(),x.tolist() if name=='tau' else None,
                            y.tolist() if name=='tau' else None)
            assert cos>.999,(name,cos.item())
