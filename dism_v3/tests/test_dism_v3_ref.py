import torch
import pytest
from flash_dism.reference.dism_v3_ref import dism_wrapper, dism_ref, dism_ref_backward


def inputs(b=2,n=5,h=3,dtype=torch.float64):
    torch.manual_seed(19)
    q,k=[torch.randn(b,n,h,4,dtype=dtype)*.3 for _ in range(2)]
    sq,sk=[torch.nn.functional.silu(torch.randn(b,n,h,2,dtype=dtype)) for _ in range(2)]
    qw,kw=[torch.randn(7,4,dtype=dtype)*.4 for _ in range(2)]
    v=torch.randn(b,n,h,6,dtype=dtype)
    tau=torch.linspace(.1,.7,h,dtype=dtype)
    return q,k,sq,sk,qw,kw,v,tau


def oracle(args,direction,hard):
    q,k,sq,sk,qw,kw,v,tau=args
    b,n,h,_=q.shape
    outputs=[]
    for batch in range(b):
        heads=[]
        for head in range(h):
            qs=q[batch,:,head]@qw.T;ks=k[batch,:,head]@kw.T
            pq,pk=qs.softmax(-1),ks.softmax(-1)
            # Independently compute Jensen score as expectation of log probabilities.
            score=(qs.log_softmax(-1)@pk.T if direction[batch,head]
                   else pq@ks.log_softmax(-1).T)
            match=qs.argmax(-1)[:,None]==ks.argmax(-1)[None,:]
            m=torch.where(hard[batch,head,:,None],match.to(q.dtype),score.exp())*tau[head].exp()
            previous=torch.zeros(n,dtype=q.dtype);out=[]
            for i in range(n):
                current=torch.zeros_like(previous)
                for j in range(i+1):
                    current[j]=m[i,j]*(1+(previous[j-1] if j else 0))
                read=sq[batch,i,head]@sk[batch,:,head].T
                out.append((current*read)@v[batch,:,head]/(1+current.sum()))
                previous=current
            heads.append(torch.stack(out))
        outputs.append(torch.stack(heads,1))
    return torch.stack(outputs)


@pytest.mark.parametrize('mode',['soft','hard','mixed'])
@pytest.mark.parametrize('direction_value',[False,True])
@pytest.mark.parametrize('n',[1,5])
def test_oracle(mode,direction_value,n):
    a=inputs(n=n);direction=torch.full((2,3),direction_value)
    hard=torch.rand(2,3,n)<{'soft':0,'hard':1,'mixed':.5}[mode]
    got=dism_wrapper(*a,direction=direction,hard=hard)
    torch.testing.assert_close(got,oracle(a,direction,hard),atol=1e-12,rtol=1e-12)


@pytest.mark.parametrize('direction_value',[True,False])
def test_gradients(direction_value):
    a=tuple(x.requires_grad_() for x in inputs(b=1,n=3,h=1))
    hard=torch.tensor([[[False,True,False]]]);direction=torch.tensor([[direction_value]])
    f=lambda *x:dism_wrapper(*x,direction=direction,hard=hard)
    assert torch.autograd.gradcheck(f,a,fast_mode=True)
    upstream=torch.randn_like(f(*a))
    g1=torch.autograd.grad((f(*a)*upstream).sum(),a)
    g2=torch.autograd.grad((oracle(a,direction,hard)*upstream).sum(),a)
    for x,y in zip(g1,g2):torch.testing.assert_close(x,y,atol=1e-10,rtol=1e-10)


def test_bf16_and_replay():
    a=inputs(dtype=torch.bfloat16)
    out=dism_wrapper(*a,generator=torch.Generator().manual_seed(12))
    assert out.shape==(2,5,3,6) and out.dtype==torch.float32 and torch.isfinite(out).all()
    torch.testing.assert_close(out,dism_wrapper(*a,generator=torch.Generator().manual_seed(12)),atol=0,rtol=0)


def test_hard_unreachable():
    b,n,h,c=1,3,1,2
    q=torch.ones(b,n,h,c,dtype=torch.float64)
    lse=torch.zeros(b,n,h,dtype=torch.float64)
    ids=torch.zeros(b,h,n,dtype=torch.long)
    got=dism_ref(q,q,q,q,lse,lse,ids,ids+1,torch.ones(b,h,dtype=torch.bool),
                 torch.ones(b,h,n,dtype=torch.bool),q,torch.ones(h))
    assert torch.equal(got,torch.zeros_like(got))


@pytest.mark.parametrize('mode',['soft','hard','mixed','unmatched'])
@pytest.mark.parametrize('gate',['finite','large_score'])
@pytest.mark.parametrize('n',[1,5])
def test_explicit_backward(mode,gate,n):
    a=inputs(n=n)
    q,k,sq,sk,_,_,v,tau=a
    if gate=='large_score':tau.fill_(25.)
    lq=torch.randn(2,n,3,dtype=torch.float64)
    lk=torch.randn_like(lq)
    ids=torch.randint(0,2,(2,3,n));idk=torch.randint(0,2,(2,3,n))
    if mode=='unmatched':ids.zero_();idk.fill_(1)
    direction=torch.tensor([[True,False,True],[False,True,False]])
    hard=torch.rand(2,3,n)<{'soft':0,'hard':1,'mixed':.5,'unmatched':1}[mode]
    names=['q_vec','k_vec','sq_vec','sk_vec','q_lse','k_lse','v','rtau']
    variables=[x.detach().requires_grad_() for x in (q,k,sq,sk,lq,lk,v,tau)]
    q,k,sq,sk,lq,lk,v,tau=variables
    args=(q,k,sq,sk,lq,lk,ids,idk,direction,hard,v,tau)
    out=dism_ref(*args);do=torch.randn_like(out)
    expected=torch.autograd.grad(out,variables,do)
    actual=dism_ref_backward(*args,do)
    for name,ref in zip(names,expected):
        assert torch.isfinite(actual[name]).all(),name
        assert not actual[name].requires_grad
        torch.testing.assert_close(actual[name],ref,atol=2e-11,rtol=2e-10,msg=name)


def test_explicit_backward_bf16():
    q,k,sq,sk,_,_,v,tau=inputs(dtype=torch.bfloat16)
    lse=torch.randn(2,5,3,dtype=torch.bfloat16)
    xs=[x.detach().requires_grad_() for x in (q,k,sq,sk,lse,lse.clone(),v,tau)]
    q,k,sq,sk,lq,lk,v,tau=xs
    ids=torch.randint(0,2,(2,3,5));hard=torch.rand(2,3,5)<.5
    args=(q,k,sq,sk,lq,lk,ids,ids,torch.ones(2,3,dtype=torch.bool),hard,v,tau)
    out=dism_ref(*args);do=torch.randn_like(out)
    expected=torch.autograd.grad(out,xs,do)
    got=dism_ref_backward(*args,do)
    for value,ref in zip(got.values(),expected):
        assert value.dtype==torch.float32
        torch.testing.assert_close(value.bfloat16(),ref,atol=.002,rtol=.016)
