import itertools
import re
import subprocess
import pytest
import torch
from dism_v2.embedding import forward,backward,_backward_extension

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")


@pytest.mark.parametrize("d,direction,n,v",itertools.product(
    [32,64,128],["q_from_k","k_from_q"],[1,65],[1,31,65,129]))
def test_same_state(d,direction,n,v,warp_specialized=False,vocab_symmetric=False,**config):
    torch.manual_seed(d+n+v)
    q,k=[torch.randn(2,2,n,d,device="cuda",dtype=torch.bfloat16) for _ in range(2)]
    eq,ek=[torch.randn(2,v,d,device="cuda",dtype=torch.bfloat16) for _ in range(2)]
    raw=forward(q,k,eq,ek,d**-.5)
    u=torch.randn(q.shape,device="cuda",dtype=torch.float32)
    lam=torch.randn(q.shape[:3],device="cuda",dtype=torch.float32)
    full=direction=="q_from_k"
    out=raw[0 if full else 1]
    actual=backward(q,k,eq,ek,out,raw[2],raw[3],u,lam,direction=direction,
                    sm_scale=d**-.5,return_preprocess=True,
                    warp_specialized=warp_specialized,vocab_symmetric=vocab_symmetric,**config)
    x,y,key,value,lx,ly=(k,q,ek,eq,raw[2],raw[3]) if full else (q,k,eq,ek,raw[3],raw[2])
    delta=(out.float()*u).sum(-1)
    torch.testing.assert_close(actual[4],delta,atol=5e-6,rtol=2e-6)
    torch.testing.assert_close(actual[5],u.bfloat16(),atol=0,rtol=0)
    u=u.bfloat16().float()
    px=torch.exp(torch.einsum("bhnd,hvd->bhnv",x.float(),key.float())*d**-.5-lx[...,None])
    py=torch.exp(torch.einsum("bhnd,hvd->bhnv",y.float(),value.float())*d**-.5-ly[...,None])
    gx=(px*(torch.einsum("bhnd,hvd->bhnv",u,value.float())-delta[...,None])).bfloat16().float()
    gy=(py*lam[...,None]).bfloat16().float()
    dx=torch.einsum("bhnv,hvd->bhnd",gx,key.float())*d**-.5
    dy=torch.einsum("bhnv,hvd->bhnd",gy,value.float())*d**-.5
    dk=torch.einsum("bhnv,bhnd->hvd",gx,x.float())*d**-.5
    dv=torch.einsum("bhnv,bhnd->hvd",gy,y.float())*d**-.5+torch.einsum("bhnv,bhnd->hvd",px.bfloat16().float(),u)
    ref=(dy,dx,dv,dk) if full else (dx,dy,dk,dv)
    for a,b in zip(actual,ref):
        torch.testing.assert_close(a,b,atol=.003,rtol=.008)


@pytest.mark.parametrize("d,direction,n,v",itertools.product(
    [32,64],["q_from_k","k_from_q"],[1,17,65],[1,65,129]))
def test_symmetric_narrow(d,direction,n,v):
    test_same_state(d,direction,n,v,warp_specialized=True,vocab_symmetric=True)


def test_symmetric_shared64_codegen():
    sass=subprocess.check_output(["/usr/local/cuda/bin/cuobjdump","--dump-sass",
                                  _backward_extension().__file__],text=True)
    kernels=[p for p in re.split(r"Function\s*:\s*",sass)[1:]
             if "vocabulary_symmetricILi64ELi16" in p.splitlines()[0]]
    assert len(kernels)==2  # shared and register-resident variants
    for body in kernels:
        assert not re.search(r"\bCALL(?:\.|\s)",body)
        assert "UTMALDG" in body and "USETMAXREG" in body


def test_codegen():
    so=_backward_extension().__file__
    sass=subprocess.check_output(["/usr/local/cuda/bin/cuobjdump","--dump-sass",so],text=True)
    assert not re.search(r"\bCALL(?:\.|\s)",sass)
    # Spills are accepted for the configuration sweep; CALLs are not.
    for body in re.split(r"Function\s*:\s*",sass)[1:]:
        name=body.splitlines()[0]
        if "_ws" in name or "vocabulary_symmetric" in name:
            assert "UTMALDG" in body and "USETMAXREG" in body,name


def test_lse_fma_codegen():
    """Guard against reintroducing a per-score LOG2E multiply in the WS loops.

    These are whole-kernel static instruction ceilings, not execution counts.
    Numerical tests independently check that the removed multiplies were safe.
    """
    sass=subprocess.check_output(["/usr/local/cuda/bin/cuobjdump","--dump-sass",
                                  _backward_extension().__file__],text=True)
    assert not re.search(r"\bCALL(?:\.|\s)",sass)
    count=0
    for body in re.split(r"Function\s*:\s*",sass)[1:]:
        name=body.splitlines()[0]
        if "_ws" not in name: continue
        count+=1
        assert "UTMALDG" in body and "USETMAXREG" in body
        tile=int(re.search(r"ILi\d+ELi(\d+)E",name).group(1))
        ceiling=3*tile
        assert len(re.findall(r"\bFMUL(?:\.|\s)",body))<=ceiling, name
    assert count==11  # three token + eight feasible paired-vocabulary kernels


from dism_v2.benchmark_embedding_backward import configurations


@pytest.mark.parametrize("d,name,config",[(d,name,cfg) for d in (32,64,128)
    for name,cfg in configurations(d) if name!="single"])
@pytest.mark.parametrize("direction",["q_from_k","k_from_q"])
@pytest.mark.parametrize("n,v",[(1,1),(17,65),(65,129),(257,257),(1025,65)])
def test_configurations(d,name,config,direction,n,v):
    test_same_state(d,direction,n,v,**config)


@pytest.mark.parametrize("d,direction,v",itertools.product(
    [32,64,128],["q_from_k","k_from_q"],[1,65,129]))
def test_reference(d,direction,v,record_property,**config):
    from dism_v2.dism_ref import interpolation_ref
    from dism_v2.emb_kernel import emb_bwd_wrapper
    torch.manual_seed(331+d+v)
    q,k=[torch.randn(2,2,65,d,device="cuda",dtype=torch.bfloat16) for _ in range(2)]
    eq,ek=[torch.randn(2,v,d,device="cuda",dtype=torch.bfloat16) for _ in range(2)]
    raw=forward(q,k,eq,ek,d**-.5)
    u=torch.randn(q.shape,device="cuda",dtype=torch.float32)
    lam=torch.randn(q.shape[:3],device="cuda",dtype=torch.float32)
    full=direction=="q_from_k"
    actual=backward(q,k,eq,ek,raw[0 if full else 1],raw[2],raw[3],u,lam,
                    direction=direction,sm_scale=d**-.5,**config)
    zero=torch.zeros_like(u)
    upstream=(u,zero,None,lam) if full else (zero,u,lam,None)
    triton=emb_bwd_wrapper(q,k,eq,ek,*raw[:4],*upstream,d**-.5)
    for a,b in zip(actual,triton):
        torch.testing.assert_close(a,b,atol=.003,rtol=.008)
    leaves=[x.float().requires_grad_() for x in (q,k,eq,ek)]
    ref=interpolation_ref(*leaves,d**-.5)
    outputs=(ref.q_from_k,ref.q_lse) if full else (ref.k_from_q,ref.k_lse)
    expected=torch.autograd.grad(outputs,leaves,(u,lam))
    failures=[]
    for name,a,b in zip(("q","k","q_voc","k_voc"),actual,expected):
        error=a.double()-b.double()
        record_property(name+"_relative_l2",(error.norm()/b.double().norm().clamp_min(1e-30)).item())
        record_property(name+"_max_abs",error.abs().max().item())
        record_property(name+"_cosine",torch.nn.functional.cosine_similarity(
            a.double().flatten(),b.double().flatten(),dim=0).item())
        try: torch.testing.assert_close(a,b,atol=.02,rtol=.02)
        except AssertionError: failures.append(name)
    assert not failures, f"FP32 oracle precision failures: {failures}"


@pytest.mark.parametrize("d,name,config",[(d,name,cfg) for d in (32,64,128)
    for name,cfg in configurations(d) if name!="single"])
@pytest.mark.parametrize("direction",["q_from_k","k_from_q"])
@pytest.mark.parametrize("v",[65,129])
def test_config_reference(d,name,config,direction,v,record_property):
    test_reference(d,direction,v,record_property,**config)


@pytest.mark.parametrize("d,direction",itertools.product([32,64,128],["q_from_k","k_from_q"]))
def test_selected_long(d,direction):
    test_same_state(d,direction,4096,1024,warp_specialized=True,
                    vocab_token_step=64 if d==32 else 32)


@pytest.mark.parametrize("d,symmetric,step,shared",[(32,True,32,False),
    (64,True,32,False),(64,True,16,True),(128,False,32,False)])
@pytest.mark.parametrize("direction",["q_from_k","k_from_q"])
def test_selected_wide(d,symmetric,step,shared,direction):
    test_same_state(d,direction,1024,8192,warp_specialized=True,
                    vocab_symmetric=symmetric,vocab_token_step=step,vocab_shared=shared)


def test_invalid_configurations():
    q=torch.zeros(1,1,1,128,device='cuda',dtype=torch.bfloat16)
    eq=q[0]
    raw=forward(q,q,eq,eq)
    for cfg,reason in [(dict(vocab_token_step=64),'shared memory capacity'),
                       (dict(vocab_token_step=8),'16/32/64'),
                       (dict(vocab_symmetric=True,vocab_shared=True),'symmetric D64')]:
        with pytest.raises(RuntimeError,match=reason):
            backward(q,q,eq,eq,raw[0],raw[2],raw[3],q.float(),raw[2],
                     direction='q_from_k',warp_specialized=True,**cfg)
