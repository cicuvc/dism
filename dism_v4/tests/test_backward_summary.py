import pytest
import cu_flash_dism as _precision_backend
import torch
import torch.nn.functional as F
import cu_flash_dism
from flash_dism.forward import forward_core
from test_forward import inputs_for

pytestmark=pytest.mark.skipif(cu_flash_dism.summary_key_dim()!=64 or
                             cu_flash_dism.forward_head_dim()!=64 or
                             cu_flash_dism.forward_readout_dim()!=32,
                             reason='legacy fixtures require R32/D64/DV64; use test_backward_dims otherwise')


def prepare(n,mode='mixed',direction='mixed'):
    inputs=inputs_for(n,mode,direction)
    out,norm,state=forward_core(*inputs,ctas=1,save_state=True)
    do=torch.randn_like(out).bfloat16()
    delta=(out.float()*do.float()).sum(-1).transpose(1,2).contiguous()
    np=state['operands'][0].shape[1]
    dop=F.pad(do,(0,0,0,0,0,np-n)).contiguous()
    return inputs,state,do,dop,delta


def local_oracle(inputs,state,do,delta):
    q,k,sq,sk,v,*_=inputs
    n=q.shape[1]
    w=cu_flash_dism.backward_recompute_probe(state['operands'],state['vertical'],n)
    causal=torch.ones(n,n,device='cuda',dtype=torch.bool).tril()
    w=w.masked_fill(~causal,-1e6)
    # Match production FP32 BEFORE BF16 conversion; FP64 evaluation can fall
    # on the other side of a BF16 midpoint. Ideal-reference tests are separate.
    p=torch.exp2(w-state['lse2'][...,None])
    a=torch.einsum('bnhr,bmhr->bhnm',sq.float(),sk.float())
    b=torch.einsum('bnhd,bmhd->bhnm',do.float(),v.float())
    beta=(p*(a*b-delta[...,None])).double()
    alpha=torch.sigmoid(w.double()*0.6931471805599453)
    ca=(p*a).bfloat16().double()
    cb=(p*b).bfloat16().double()
    dv=torch.einsum('bhnm,bnhd->bmhd',ca,do.double())
    dsq=torch.einsum('bhnm,bmhr->bnhr',cb,sk.double())
    dsk=torch.einsum('bhnm,bnhr->bmhr',cb,sq.double())
    np=state['operands'][0].shape[1]
    alpha=F.pad(alpha,(0,np-n,0,np-n),value=1.)
    beta=F.pad(beta,(0,np-n,0,np-n),value=0.)
    summaries=[]
    step=cu_flash_dism.backward_summary_k()
    index=torch.arange(np,device='cuda')
    for chunk in range((n+step-1)//step):
        aa=torch.ones_like(alpha[...,0,0:np])
        bb=torch.zeros_like(aa)
        for j in range(step-1,-1,-1):
            key=chunk*step+j
            query=index+j
            valid=query<np
            av=torch.where(valid,alpha[...,query.clamp_max(np-1),key],1.)
            bv=torch.where(valid,beta[...,query.clamp_max(np-1),key],0.)
            aa=av*aa
            bb=bv+av*bb
        summaries.append((aa,bb))
    sa=torch.stack([x[0] for x in summaries],2)
    sb=torch.stack([x[1] for x in summaries],2)
    return dv,dsq,dsk,sa,sb


@pytest.mark.parametrize('n',[1,17,33,65,129,257])
@pytest.mark.parametrize('mode',['soft','mixed','hard'])
@pytest.mark.strict_gradient
@pytest.mark.skipif(not _precision_backend.fp32_enabled(),
                    reason='requires optional FP32 validation instances')
def test_summary_readout_gradients(n,mode):
    inputs,state,do,dop,delta=prepare(n,mode)
    actual=cu_flash_dism.backward_summary(state['operands'],state['vertical'],dop,
                                         state['lse2'],delta,n,1,True)
    torch.cuda.synchronize()
    expected=local_oracle(inputs,state,do,delta)
    for name,x,y in zip(('dv','dsq','dsk','a','b'),actual,expected):
        # Summary unused query padding is deliberately initialized to zero.
        if name in ('a','b'):
            end=(n+31)//32*32
            x=x[...,:end]; y=y[...,:end]
        torch.testing.assert_close(x.double(),y,atol=2e-4,rtol=3e-3,msg=name)
