"""Fixed-direction CUDA core tests with explicit BF16 interpolation oracle."""
import itertools
import math
from dataclasses import replace
import pytest
import torch
from dism_v2.core import forward, RowRNGState
from dism_v2.dism_ref import interpolation_ref, voc_dism_ref

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")

def philox_word(seed, offset, row):
    """Independent integer CPU oracle, Random123 Philox4x32-10 convention."""
    mask=2**32-1
    c=[(offset//4)&mask,(offset//4)>>32,row&mask,row>>32]
    k=[seed&mask,seed>>32]
    for _ in range(10):
        p0=0xD2511F53*c[0]; p1=0xCD9E8D57*c[2]
        c=[(p1>>32)^c[1]^k[0],p1&mask,(p0>>32)^c[3]^k[1],p0&mask]
        k=[(k[0]+0x9E3779B9)&mask,(k[1]+0xBB67AE85)&mask]
    return c[0]

def test_philox_known_answer():
    assert philox_word(0,0,0)==0x6627E8D5

@pytest.mark.parametrize("d,dv",itertools.product((32,64,128),repeat=2))
@pytest.mark.parametrize("direction",("q_from_k","k_from_q"))
@torch.no_grad()
def test_mixed_rng(d,dv,direction):
    check_case(d,dv,139,direction,0.37)

@pytest.mark.parametrize("n",(1,33,65,129,257,513))
@torch.no_grad()
def test_mixed_rng_tails(n):
    check_case(64,64,n,"q_from_k",0.63)

@pytest.mark.parametrize("probability",(0.,1.,0.01,0.5,0.99))
@torch.no_grad()
def test_rng_default_generator_and_row_identity(probability):
    # Unmatched hard rows have exactly zero normalizer; soft rows do not.
    # Thus the actual row decisions can be observed without any mask buffer.
    shape=(2,2,257)
    x=torch.zeros((*shape,32),device="cuda",dtype=torch.bfloat16)
    lse=torch.zeros(shape,device="cuda")
    labels=torch.zeros(shape,device="cuda",dtype=torch.long)
    inputs=(x,x,x,lse,torch.zeros(2,device="cuda"),labels,labels+1)
    kwargs=dict(sm_scale=1.,direction="q_from_k",hard_prob=probability)
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.cuda.manual_seed(321)
        g=torch.cuda.default_generators[torch.cuda.current_device()]
        before=g.get_offset()
        o,l,state=forward(*inputs,**kwargs,return_rng_state=True)
        assert g.get_offset()==before+(4 if 0<probability<1 else 0)
        p32=torch.tensor(probability,dtype=torch.float32).item()
        expected=torch.tensor([(philox_word(state.seed,state.offset,r)>>8)*2**-24<p32
            for r in range(math.prod(shape))],device="cuda").reshape(shape)
        assert torch.equal(l==0,expected)
        assert torch.count_nonzero(o)==0
        saved=g.get_state()
        again=forward(*inputs,**kwargs,rng_state=state)
        assert torch.equal(saved,g.get_state()) and torch.equal(l,again[1])
        # Re-seeding reproduces both metadata and results.
        torch.cuda.manual_seed(321)
        result=forward(*inputs,**kwargs,return_rng_state=True)
        assert state==result[-1] and torch.equal(l,result[1])
        with pytest.raises(ValueError,match="match"):
            forward(*inputs,**kwargs,rng_state=replace(state,shape=(1,4,257)))
        with pytest.raises(ValueError,match="offset"):
            forward(*inputs,**kwargs,rng_state=replace(state,offset=1))
        with pytest.raises(ValueError,match="exclusive"):
            forward(*inputs,**kwargs,rng_state=state,generator=g)
        with pytest.raises(RuntimeError,match="generator"):
            forward(*inputs,**kwargs,generator=torch.Generator())

@pytest.mark.parametrize("d,dv",itertools.product((32,64,128),repeat=2))
@pytest.mark.parametrize("direction",("q_from_k","k_from_q"))
@pytest.mark.parametrize("hard",(False,True))
@torch.no_grad()
def test_dimensions(d,dv,direction,hard):
    check_case(d,dv,139,direction,hard)

@pytest.mark.parametrize("n",(1,17,31,32,33,63,64,65,127,128,129,257,513))
@torch.no_grad()
def test_tails(n):
    check_case(64,64,n,"q_from_k",False)

def check_case(d,dv,n,direction,hard):
    generator=torch.Generator(device="cuda").manual_seed(41+n+d+dv)
    def rand(shape): return torch.randn(shape,device="cuda",dtype=torch.bfloat16,generator=generator)
    batch,heads=2,2
    q,k,v=rand((batch,heads,n,d)),rand((batch,heads,n,d)),rand((batch,heads,n,dv))
    qvoc,kvoc=rand((heads,11,d)),rand((heads,11,d))
    tau=torch.tensor([-0.5,0.5],device="cuda")
    scale=d**-0.5
    interp=interpolation_ref(q,k,qvoc,kvoc,scale)
    interp=replace(interp,q_from_k=interp.q_from_k.bfloat16(),k_from_q=interp.k_from_q.bfloat16())
    a,b,lse=(q,interp.q_from_k,interp.q_lse) if direction=="q_from_k" else (interp.k_from_q,k,interp.k_lse)
    inputs=(a.contiguous(),b.contiguous(),v,lse.contiguous(),tau,
        interp.q_index.contiguous(),interp.k_index.contiguous())
    kwargs=dict(sm_scale=scale,direction=direction,hard_prob=float(hard),return_debug=True)
    rng=torch.Generator(device="cuda").manual_seed(2**63+12345)
    rng.set_offset(2**34+12)
    before=rng.get_offset()
    actual,l2,summary,boundary,state=forward(*inputs,**kwargs,generator=rng,return_rng_state=True)
    mixed=0<float(hard)<1
    assert rng.get_offset()==before+(4 if mixed else 0)
    mask=torch.tensor(hard,device="cuda",dtype=torch.bool)
    if mixed:
        assert state.seed==2**63+12345 and state.offset==before
        p32=torch.tensor(float(hard),dtype=torch.float32).item()
        mask=torch.tensor([(philox_word(state.seed,state.offset,row)>>8)*2**-24<p32
            for row in range(batch*heads*n)],device="cuda").reshape(batch,heads,n,1)
        default_before=torch.cuda.get_rng_state()
        replay=forward(*inputs,**kwargs,rng_state=state)
        assert torch.equal(default_before,torch.cuda.get_rng_state())
        for x,y in zip((actual,l2,summary,boundary),replay):
            torch.testing.assert_close(x,y,atol=0,rtol=0)
        # Fresh calls use the next per-row Philox block, not the same masks.
        *_,next_state=forward(*inputs,**kwargs,generator=rng,return_rng_state=True)
        assert next_state.offset==before+4 and rng.get_offset()==before+8
    expected,aux=voc_dism_ref(q,k,v,tau,qvoc,kvoc,hard_prob=float(hard),direction=direction,
        sm_scale=scale,interpolation=interp,hard_mask=mask,return_aux=True)
    torch.cuda.synchronize()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.float(),expected.float(),atol=0.008,rtol=0.012)
    ln2=math.log(2.0)
    expected_l=torch.logaddexp(torch.logsumexp(aux["scores"],dim=-1),torch.zeros_like(l2))/ln2
    torch.testing.assert_close(l2,expected_l,atol=2e-5,rtol=2e-5)
    # Independent double recurrence checks both local summary components and
    # all resolved checkpoint values, including identity padding after N.
    np=boundary.shape[-1]; rows=summary.shape[2]*32
    logs=aux["log_m"].double()/ln2
    previous=torch.full((batch,heads,np),-torch.inf,device="cuda",dtype=torch.float64)
    local_a=torch.zeros_like(previous); local_b=torch.full_like(previous,-torch.inf)
    for i in range(rows):
        shifted=torch.nn.functional.pad(previous[...,:-1],(1,0),value=-torch.inf)
        if i%32==0:
            local_a.zero_(); local_b.fill_(-torch.inf)
        sa=torch.nn.functional.pad(local_a[...,:-1],(1,0),value=0)
        sb=torch.nn.functional.pad(local_b[...,:-1],(1,0),value=-torch.inf)
        m=torch.zeros_like(previous); valid=torch.zeros_like(previous,dtype=torch.bool)
        if i<n:
            m[...,:n]=logs[...,i,:]
            m[...,:n].masked_fill_(torch.arange(n,device="cuda")>i,-torch.inf)
            valid[...,:n]=True
        offset=torch.where(valid,m,torch.full_like(m,-torch.inf))
        previous=torch.logaddexp((shifted+m)*ln2,offset*ln2)/ln2
        local_a=sa+m
        local_b=torch.logaddexp((sb+m)*ln2,offset*ln2)/ln2
        if i%32==31:
            torch.testing.assert_close(boundary[...,i//32,:].double(),previous,atol=3e-5,rtol=3e-5)
            torch.testing.assert_close(summary[...,i//32,:,0].double(),local_a,atol=3e-5,rtol=3e-5)
            torch.testing.assert_close(summary[...,i//32,:,1].double(),local_b,atol=3e-5,rtol=3e-5)
