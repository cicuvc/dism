"""Key-owned CUDA dV vs voc_dism_ref autograd with identical BF16 interpolation."""
from dataclasses import replace
import itertools
import json
import math
import pytest
import torch
from dism_v2.core import forward_interpolated
from dism_v2.backward import value_gradient,delta
from dism_v2.dism_ref import interpolation_ref,voc_dism_ref

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")


@pytest.fixture(autouse=True)
def exact_oracle_matmul():
    previous=torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32=False
    try: yield
    finally: torch.backends.cuda.matmul.allow_tf32=previous


def word(seed,offset,row):
    mask=2**32-1
    a,b,c,d=(offset//4)&mask,(offset//4)>>32,row&mask,row>>32
    k0,k1=seed&mask,seed>>32
    for _ in range(10):
        p0,p1=0xD2511F53*a,0xCD9E8D57*c
        a,b,c,d=(p1>>32)^b^k0,p1&mask,(p0>>32)^d^k1,p0&mask
        k0,k1=(k0+0x9E3779B9)&mask,(k1+0xBB67AE85)&mask
    return a


@pytest.mark.parametrize("d,dv",itertools.product((32,64,128),repeat=2))
@pytest.mark.parametrize("direction",("q_from_k","k_from_q"))
@pytest.mark.parametrize("probability",(0.,.37,1.))
def test_dv_dimensions(d,dv,direction,probability,record_property):
    run(d,dv,139,direction,probability,record_property)


@pytest.mark.parametrize("n",(1,17,31,32,63,64,65,128,129,257,513))
def test_dv_tails(n,record_property):
    run(64,128,n,"random",.63,record_property)


@pytest.mark.parametrize("mode",("chain","break","bounded_soft"))
@pytest.mark.parametrize("direction",("q_from_k","k_from_q"))
@pytest.mark.parametrize("n",(1025,8193))
def test_dv_long(mode,direction,n,record_property):
    run(64,128,n,direction,0. if mode=="bounded_soft" else 1.,record_property,mode)


@pytest.mark.parametrize("n",(2049,8193))
@pytest.mark.parametrize("direction",("q_from_k","k_from_q"))
@pytest.mark.parametrize("probability",(0.,.37))
def test_dv_long_soft(n,direction,probability,record_property):
    run(64,128,n,direction,probability,record_property,"bounded_soft")


@pytest.mark.parametrize("d,dv",itertools.product((32,64,128),repeat=2))
@pytest.mark.parametrize("direction",("q_from_k","k_from_q"))
@pytest.mark.parametrize("probability",(0.,.37,1.))
def test_backward_summary(d,dv,direction,probability,record_property):
    run(d,dv,139,direction,probability,record_property,check_summary=True)


@pytest.mark.parametrize("n",(1,17,64,65,129,513,1025,2049))
@pytest.mark.parametrize("mode",("chain","break","random"))
def test_backward_summary_tails(n,mode,record_property):
    run(64,128,n,"random",.37 if mode=="random" else 1.,record_property,mode,check_summary=True)


def run(d,dv,n,direction,probability,record_property,mode="random",check_summary=False):
    if torch.cuda.get_device_capability()!=(12,0): pytest.skip("sm120a only")
    batch,heads=(1,1) if n>513 else (2,2)
    gen=torch.Generator(device="cuda").manual_seed(741+n+d+dv)
    def rand(shape): return torch.randn(shape,device="cuda",dtype=torch.bfloat16,generator=gen)
    q,k,v=rand((batch,heads,n,d)),rand((batch,heads,n,d)),rand((batch,heads,n,dv))
    qvoc,kvoc=rand((heads,11,d)),rand((heads,11,d))
    scale=d**-.5
    with torch.no_grad(): interp=interpolation_ref(q,k,qvoc,kvoc,scale)
    interp=replace(interp,q_from_k=interp.q_from_k.bfloat16().contiguous(),
        k_from_q=interp.k_from_q.bfloat16().contiguous(),q_lse=interp.q_lse.contiguous(),
        k_lse=interp.k_lse.contiguous(),q_index=interp.q_index.contiguous(),k_index=interp.k_index.contiguous())
    if mode in ("chain","break"):
        interp=replace(interp,q_index=torch.zeros_like(interp.q_index),
            k_index=torch.full_like(interp.k_index,mode=="break"))
    tau=torch.full((heads,),math.log(d) if mode!="random" else .2,device="cuda")
    out,norm,edges,state=forward_interpolated(q,k,v,tau,interp,sm_scale=scale,direction=direction,
        hard_prob=probability,generator=gen,save_boundaries=True,return_rng_state=True)
    a,b,lse=(q,interp.q_from_k,interp.q_lse) if state.direction=="q_from_k" else (interp.k_from_q,k,interp.k_lse)
    dout=rand(v.shape)
    kwargs=dict(sm_scale=scale,rng_state=state)
    before=torch.cuda.get_rng_state(); explicit_before=gen.get_state()
    stream=torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream): actual=value_gradient(a,b,dout,lse,tau,interp.q_index,interp.k_index,norm,edges,**kwargs)
    torch.cuda.current_stream().wait_stream(stream)
    assert torch.equal(before,torch.cuda.get_rng_state()) and torch.equal(explicit_before,gen.get_state())
    replay=value_gradient(a,b,dout,lse,tau,interp.q_index,interp.k_index,norm,edges,**kwargs)
    torch.testing.assert_close(actual,replay,atol=0,rtol=0)
    assert actual.dtype==torch.float32 and actual.shape==v.shape and torch.isfinite(actual).all()
    p32=torch.tensor(probability,dtype=torch.float32).item()
    mask=torch.tensor([(word(state.seed,state.offset,r)>>8)*2**-24<p32 for r in range(batch*heads*n)],
        device="cuda").reshape(batch,heads,n,1)
    # FP32 v avoids output BF16 rounding in the mathematical dV oracle.
    vf=v.float().requires_grad_()
    reference,aux=voc_dism_ref(q,k,vf,tau,qvoc,kvoc,sm_scale=scale,direction=state.direction,
        hard_prob=probability,interpolation=interp,hard_mask=mask,return_aux=True)
    expected,=torch.autograd.grad(reference,vf,dout.float())
    if check_summary:
        dd=delta(dout,out)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            fused,summary,boundary=value_gradient(a,b,dout,lse,tau,interp.q_index,interp.k_index,norm,edges,**kwargs,v=v,delta=dd)
        torch.cuda.current_stream().wait_stream(stream)
        assert torch.equal(before,torch.cuda.get_rng_state()) and torch.equal(explicit_before,gen.get_state())
        torch.testing.assert_close(fused,actual,atol=2e-6,rtol=2e-6)
        np=summary.shape[-2]
        with torch.no_grad():
            # Isolate the reverse operator from already rounded forward W/L2.
            # This independent diagnostic rescan reads only saved sparse edges.
            from test_dism_v2_recompute import probe
            reconstructed=probe().recompute(a,b,lse,tau,interp.q_index,interp.k_index,
                edges.vertical,edges.horizontal,scale,state.direction=="k_from_q",probability,state.seed,state.offset)
            alpha=torch.ones((batch,heads,np,np),device="cuda",dtype=torch.float64)
            emission=torch.zeros_like(alpha)
            w=reconstructed[...,:n,:n].double()*math.log(2)
            finite=torch.isfinite(w)&torch.isfinite(aux["scores"])
            record_property("summary_w_max_abs",(w[finite]-aux["scores"].double()[finite]).abs().max().item() if finite.any() else 0.)
            alpha[...,:n,:n]=torch.sigmoid(w)
            emission[...,:n,:n]=torch.exp(w-norm.double()[...,None]*math.log(2))*(dout.double()@v.double().transpose(-1,-2)-dd.double()[...,None])
            following=torch.zeros((batch,heads,np),device="cuda",dtype=torch.float64)
            local_a=torch.ones_like(following); local_b=torch.zeros_like(following)
            errors=[0.,0.,0.]
            for k in range(np-1,-1,-1):
                if k%32==31: local_a.fill_(1); local_b.zero_()
                sf=torch.nn.functional.pad(following[...,1:],(0,1))
                sa=torch.nn.functional.pad(local_a[...,1:],(0,1),value=1)
                sb=torch.nn.functional.pad(local_b[...,1:],(0,1))
                following=alpha[...,k]*sf+emission[...,k]
                local_a=alpha[...,k]*sa
                local_b=alpha[...,k]*sb+emission[...,k]
                if k%32==0:
                    for index,(x,y) in enumerate(((summary[...,k//32,:,0],local_a),(summary[...,k//32,:,1],local_b),(boundary[...,k//32,:],following))):
                        errors[index]=max(errors[index],(x.double()-y).abs().max().item())
                    torch.testing.assert_close(summary[...,k//32,:,0].double(),local_a,atol=2e-5,rtol=2e-4)
                    torch.testing.assert_close(summary[...,k//32,:,1].double(),local_b,atol=2e-4,rtol=5e-4)
                    torch.testing.assert_close(boundary[...,k//32,:].double(),following,atol=5e-4,rtol=1e-3)
            record_property("summary",json.dumps(dict(d=d,dv=dv,n=n,mode=mode,probability=probability,
                max_abs_first=errors[0],max_abs_second=errors[1],max_abs_boundary=errors[2])))
    difference=(actual-expected).double()
    denom=expected.double().norm()
    relative=(difference.norm()/denom).item() if denom>0 else difference.norm().item()
    cosine=torch.nn.functional.cosine_similarity(actual.double().flatten(),expected.double().flatten(),dim=0).item() if denom>0 else 1.
    with torch.no_grad():
        scores=aux["scores"]
        weights=torch.exp(scores-torch.logaddexp(torch.logsumexp(scores,-1),torch.zeros_like(norm))[...,None])
        quantized=weights.bfloat16().float().transpose(-1,-2)@dout.float()
        quantization_rmse=(quantized-expected).double().square().mean().sqrt().item()
        kernel_quantized_rmse=(quantized-actual).double().square().mean().sqrt().item()
    record_property("dv",json.dumps(dict(d=d,dv=dv,n=n,direction=state.direction,probability=probability,mode=mode,
        max_abs=difference.abs().max().item(),rmse=difference.square().mean().sqrt().item(),relative_l2=relative,cosine=cosine,
        quantization_rmse=quantization_rmse,kernel_quantized_rmse=kernel_quantized_rmse)))
    torch.testing.assert_close(actual,expected,atol=.008,rtol=.012)
    assert relative<.004 and cosine>.99998
    if mode=="break": assert torch.count_nonzero(actual)==0
    if n==1:
        args=(a,b,dout,lse,tau,interp.q_index,interp.k_index,norm)
        with pytest.raises(ValueError,match="offset"):
            value_gradient(*args,edges,sm_scale=scale,rng_state=replace(state,offset=1))
        with pytest.raises(ValueError,match="shape"):
            value_gradient(*args,edges,sm_scale=scale,rng_state=replace(state,shape=(1,1,1)))
        with pytest.raises(RuntimeError,match="granularity"):
            value_gradient(*args,replace(edges,horizontal=edges.horizontal[:,:,:0,:]),**kwargs)
