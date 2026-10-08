import pytest
import torch

from flash_dism.reference.dism_v3_ref import dism_ref,dism_wrapper
from flash_dism.reference.dism_decode_ref import dism_decode_ref,dism_wrapper_decode


def inputs(dtype=torch.float64,n=17):
    torch.manual_seed(17)
    def vec(c):
        return (.4*torch.randn(2,n,3,c)).to(dtype)
    q,k,sq,sk,v=vec(5),vec(5),vec(4),vec(4),vec(7)
    lq,lk=[(2+torch.rand(2,n,3)).to(dtype) for _ in range(2)]
    iq,ik=[torch.randint(3,(2,3,n)) for _ in range(2)]
    direction=torch.tensor([[True,False,True],[False,True,False]])
    hard=torch.rand(2,3,n)<.5
    tau=torch.tensor([0.,.5,1.],dtype=dtype)
    return q,k,sq,sk,lq,lk,iq,ik,direction,hard,v,tau


def section(x,start,end):
    return tuple(value if i in (8,11) else
                 value[:,:,start:end] if i in (6,7,9) else value[:,start:end]
                 for i,value in enumerate(x))


@pytest.mark.parametrize('dtype',[torch.float64,torch.float32,torch.bfloat16])
@pytest.mark.parametrize('mode',['soft','mixed','hard'])
@pytest.mark.parametrize('chunks',[[17],[1]*17,[3,1,8,5]])
def test_decode_matches_dense(dtype,mode,chunks):
    x=list(inputs(dtype))
    if mode!='mixed':
        x[9].fill_(mode=='hard')
    expected=dism_ref(*x)
    cache=None
    result=[]
    start=0
    for n in chunks:
        old=cache
        snapshot=None if old is None else old.last_w.clone()
        out,cache=dism_decode_ref(*section(x,start,start+n),cache=cache)
        assert cache.length==start+n
        assert cache.last_w.shape==(2,3,start+n)
        assert not cache.last_w.requires_grad
        if old is not None:
            torch.testing.assert_close(old.last_w,snapshot,atol=0,rtol=0)
            assert old.length==start
        result.append(out)
        start+=n
    tolerance=2e-12 if dtype==torch.float64 else 2e-6
    torch.testing.assert_close(torch.cat(result,1),expected,atol=tolerance,rtol=tolerance)
    assert cache.k_vec.dtype==(torch.float64 if dtype==torch.float64 else torch.float32)
    assert cache.last_w.numel()==2*3*17  # only one row, not a dense W matrix


def test_hard_fallback_and_self_diagonal():
    x=list(inputs(n=31))
    x[9].fill_(True)
    x[6].fill_(0)
    x[7].fill_(1)
    out,cache=dism_decode_ref(*x)
    assert torch.count_nonzero(out)==0
    assert torch.isneginf(cache.last_w).all()
    x[7].fill_(0)
    out,cache=dism_decode_ref(*x)
    torch.testing.assert_close(out,dism_ref(*x),atol=2e-12,rtol=2e-12)
    state=torch.zeros(3,dtype=torch.float64)
    for _ in range(31):
        state=x[-1]+torch.nn.functional.softplus(state) if _ else x[-1].clone()
    torch.testing.assert_close(cache.last_w[:,:,-1],state.expand(2,3),atol=2e-12,rtol=2e-12)


def test_snapshot_reset_and_invalid_reuse():
    x=inputs()
    out,cache=dism_decode_ref(*section(x,0,3))
    saved=cache.k_vec.clone()
    x[1][:,:3].zero_()
    torch.testing.assert_close(cache.k_vec,saved,atol=0,rtol=0)
    fresh,_=dism_decode_ref(*section(x,3,5))
    torch.testing.assert_close(fresh,dism_ref(*section(x,3,5)),atol=2e-12,rtol=2e-12)
    changed=list(section(x,3,5))
    changed[8]=~changed[8]
    with pytest.raises(ValueError,match='remain fixed'):
        dism_decode_ref(*changed,cache=cache)
    changed=list(section(x,3,5))
    changed[-1]=changed[-1]+.1
    with pytest.raises(ValueError,match='remain fixed'):
        dism_decode_ref(*changed,cache=cache)
    with pytest.raises(ValueError,match='empty'):
        dism_decode_ref(*section(x,3,3),cache=cache)


@pytest.mark.parametrize('hard_prob',[0.,.5,1.])
def test_vocabulary_wrapper(hard_prob):
    q,k,sq,sk,lq,lk,iq,ik,direction,hard,v,tau=inputs()
    eq,ek=[.3*torch.randn(9,5,dtype=q.dtype) for _ in range(2)]
    hard=torch.rand_like(hard,dtype=torch.float32)<hard_prob
    expected=dism_wrapper(q,k,sq,sk,eq,ek,v,tau,direction=direction,hard=hard)
    cache=None
    results=[]
    for start,end in [(0,4),(4,5),(5,17)]:
        out,cache=dism_wrapper_decode(q[:,start:end],k[:,start:end],sq[:,start:end],sk[:,start:end],
            eq,ek,v[:,start:end],tau,cache=cache,direction=direction if start==0 else None,
            hard=hard[:,:,start:end])
        results.append(out)
    torch.testing.assert_close(torch.cat(results,1),expected,atol=2e-12,rtol=2e-12)


def test_large_scores_are_normalized_stably():
    x=list(inputs(n=65))
    x[9].fill_(True)
    x[6].zero_()
    x[7].zero_()
    x[-1].fill_(100.)  # Deliberately far outside training range, >exp overflow.
    prefix,cache=dism_decode_ref(*section(x,0,64))
    out,updated=dism_decode_ref(*section(x,64,65),cache=cache)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(torch.cat((prefix,out),1),dism_ref(*x),atol=2e-12,rtol=2e-12)
    replay,_=dism_decode_ref(*section(x,64,65),cache=cache)
    torch.testing.assert_close(replay,out,atol=0,rtol=0)


def test_wrapper_samples_direction_once():
    q,k,sq,sk,_,_,_,_,_,_,v,tau=inputs(n=3)
    eq,ek=[torch.randn(7,5,dtype=q.dtype) for _ in range(2)]
    generator=torch.Generator().manual_seed(82)
    first,cache=dism_wrapper_decode(q[:,:1],k[:,:1],sq[:,:1],sk[:,:1],eq,ek,v[:,:1],tau,
                                     hard_prob=1.,generator=generator)
    rng=generator.get_state().clone()
    rest,cache=dism_wrapper_decode(q[:,1:],k[:,1:],sq[:,1:],sk[:,1:],eq,ek,v[:,1:],tau,
                                    cache=cache,hard_prob=1.,generator=generator)
    assert torch.equal(rng,generator.get_state())
    expected=dism_wrapper(q,k,sq,sk,eq,ek,v,tau,direction=cache.direction,hard_prob=1.)
    torch.testing.assert_close(torch.cat((first,rest),1),expected,atol=2e-12,rtol=2e-12)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA device unavailable')
@pytest.mark.parametrize('dtype',[torch.bfloat16,torch.float64])
def test_cuda_torch_decode(dtype):
    x=tuple(value.cuda() for value in inputs(dtype))
    first,cache=dism_decode_ref(*section(x,0,16))
    last,cache=dism_decode_ref(*section(x,16,17),cache=cache)
    tolerance=2e-12 if dtype==torch.float64 else 2e-6
    torch.testing.assert_close(torch.cat((first,last),1),dism_ref(*x),atol=tolerance,rtol=tolerance)
