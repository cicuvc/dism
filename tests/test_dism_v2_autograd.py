"""Six-input autograd through selectable embedding backends and CUDA core."""
import itertools
import math
import json
import re
import subprocess
from dataclasses import replace
import pytest
import torch
from dism_v2.autograd import voc_dism
from dism_v2.core import forward_interpolated
from dism_v2.backward import delta,value_gradient,operand_gradient
from dism_v2.emb_kernel import emb_fwd_wrapper,emb_bwd_wrapper
from dism_v2.dism_ref import InterpolationResult,interpolation_ref,voc_dism_ref
from test_dism_v2_precision import row_mask

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")
NAMES=("q","k","v","rtau","q_voc","k_voc")


@pytest.mark.parametrize("backend",("cuda","cuda_symmetric"))
@pytest.mark.parametrize("d",(32,64,128))
def test_all_cuda_no_embedding_fallback(backend,d,monkeypatch):
    import dism_v2.autograd as module
    def forbidden(*args,**kwargs):
        raise AssertionError("all-CUDA path must not call Triton embedding")
    monkeypatch.setattr(module,"emb_fwd_wrapper",forbidden)
    monkeypatch.setattr(module,"emb_bwd_wrapper",forbidden)
    ins,dout=inputs(d,128,65)
    leaves=[x.detach().requires_grad_() for x in ins]
    out=voc_dism(*leaves,sm_scale=d**-.5,hard_prob=.37,
        embedding_backend="cuda",embedding_backward_backend=backend)
    grads=torch.autograd.grad(out,leaves,dout)
    assert torch.isfinite(out).all() and all(torch.isfinite(g).all() for g in grads)

def test_embedding_row_store_ownership():
    import ast
    from dism_v2.emb_kernel import _interp_bwd
    tree=ast.parse(_interp_bwd.src)
    found=[]
    def visit(node,guarded=False):
        if isinstance(node,ast.If) and ast.unparse(node.test)=="pid_v == 0":
            for child in node.body: visit(child,True)
            for child in node.orelse: visit(child,guarded)
            return
        if isinstance(node,ast.Call) and ast.unparse(node.func)=="tl.store":
            names={x.id for x in ast.walk(node.args[0]) if isinstance(x,ast.Name)}
            if names & {"dq","dk"}: found.append(guarded)
        for child in ast.iter_child_nodes(node): visit(child,guarded)
    visit(tree)
    assert found==[True,True],"dq/dk stores must have one pid_v owner, not ceil(V/64) writers"

@pytest.fixture(autouse=True)
def exact_matmul():
    old=torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32=False
    try: yield
    finally: torch.backends.cuda.matmul.allow_tf32=old

def inputs(d,dv,n,vocab=65,tau=.2):
    if torch.cuda.get_device_capability()!=(12,0): pytest.skip("sm120a only")
    gen=torch.Generator(device="cuda").manual_seed(714+d+dv+n+vocab)
    def rand(shape): return torch.randn(shape,device="cuda",dtype=torch.bfloat16,generator=gen)
    q=rand((2,2,n,d));k=rand(q.shape);v=rand((2,2,n,dv))
    ts=torch.full((2,),tau,device="cuda")
    qv=rand((2,vocab,d));kv=rand(qv.shape)
    return (q,k,v,ts,qv,kv),rand(v.shape)

def manual(ins,dout,scale,state,embedding_backend="triton"):
    q,k,v,tau,qv,kv=ins
    if embedding_backend=="cuda":
        from dism_v2.embedding import forward as embedding_forward
        raw=embedding_forward(q,k,qv,kv,scale,warp_specialized=True)
    else:
        raw=emb_fwd_wrapper(q,k,qv,kv,scale)
    emb=InterpolationResult(*raw)
    emb=replace(emb,q_index=emb.q_index.long(),k_index=emb.k_index.long())
    out,norm,edges=forward_interpolated(q,k,v,tau,emb,sm_scale=scale,direction=state.direction,
        hard_prob=state.hard_prob,rng_state=state,save_boundaries=True)
    if state.direction=="q_from_k": a,b,lse=q,emb.q_from_k,emb.q_lse
    else: a,b,lse=emb.k_from_q,k,emb.k_lse
    dd=delta(dout,out);kw=dict(sm_scale=scale,rng_state=state)
    dv,_,g=value_gradient(a,b,dout,lse,tau,emb.q_index,emb.k_index,norm,edges,
        **kw,v=v,delta=dd,warp_specialized=True)
    da,db,dl,dt=operand_gradient(a,b,v,dout,lse,tau,emb.q_index,emb.k_index,norm,dd,edges,g,**kw)
    z=torch.zeros_like(a,dtype=torch.float32)
    if state.direction=="q_from_k":
        dq,dk,dqv,dkv=emb_bwd_wrapper(q,k,qv,kv,*raw[:4],db,z,None,dl,scale)
        dq+=da
    else:
        dq,dk,dqv,dkv=emb_bwd_wrapper(q,k,qv,kv,*raw[:4],z,da,dl,None,scale)
        dk+=db
    return out,(dq,dk,dv,dt,dqv,dkv)

@pytest.mark.parametrize("d,dv",itertools.product((32,64,128),repeat=2))
@pytest.mark.parametrize("direction",("q_from_k","k_from_q","random"))
@pytest.mark.parametrize("probability",(0.,.37,1.))
@pytest.mark.parametrize("embedding_backend",("triton","cuda"))
@pytest.mark.parametrize("embedding_backward_backend",("triton","cuda","cuda_symmetric"))
def test_autograd_wiring(d,dv,direction,probability,embedding_backend,embedding_backward_backend):
    ins,dout=inputs(d,dv,65)
    leaves=[x.detach().requires_grad_() for x in ins]
    gen=torch.Generator(device="cuda").manual_seed(932)
    out,state=voc_dism(*leaves,sm_scale=d**-.5,direction=direction,hard_prob=probability,
        generator=gen,return_rng_state=True,embedding_backend=embedding_backend,
        embedding_backward_backend=embedding_backward_backend)
    before=gen.get_state();default_before=torch.cuda.get_rng_state()
    grads=torch.autograd.grad(out,leaves,dout)
    assert torch.equal(before,gen.get_state()) and torch.equal(default_before,torch.cuda.get_rng_state())
    with torch.no_grad(): expected,reference=manual(ins,dout,d**-.5,state,embedding_backend)
    torch.testing.assert_close(out,expected,atol=0,rtol=0)
    for x,y,leaf in zip(grads,reference,ins):
        assert x.shape==leaf.shape and x.dtype==leaf.dtype and torch.isfinite(x).all()
        torch.testing.assert_close(x.float(),y.to(leaf.dtype).float(),atol=.002,rtol=.008)
    if probability==1.:
        assert all(torch.count_nonzero(grads[i])==0 for i in (0,1,4,5))

@pytest.mark.parametrize("d",(32,64,128))
@pytest.mark.parametrize("vocab",(1,31,64,65,129))
def test_embedding_backward(d,vocab,record_property):
    (q,k,_,_,qv,kv),_=inputs(d,32,65,vocab)
    raw=emb_fwd_wrapper(q,k,qv,kv,d**-.5)
    gen=torch.Generator(device="cuda").manual_seed(113)
    gout=[torch.randn(x.shape,device="cuda",dtype=torch.float32,generator=gen) for x in raw[:4]]
    actual=emb_bwd_wrapper(q,k,qv,kv,*raw[:4],*gout,d**-.5)
    leaves=[x.float().requires_grad_() for x in (q,k,qv,kv)]
    oracle=interpolation_ref(*leaves,d**-.5)
    expected=torch.autograd.grad((oracle.q_from_k,oracle.k_from_q,oracle.k_lse,oracle.q_lse),
        leaves,gout)
    failures=[]
    for name,x,y in zip(("q","k","q_voc","k_voc"),actual,expected):
        record_property(name+"_relative_l2",((x-y).double().norm()/y.double().norm()).item())
        record_property(name+"_max_abs",(x-y).abs().max().item())
        record_property(name+"_cosine",torch.nn.functional.cosine_similarity(
            x.double().flatten(),y.double().flatten(),dim=0).item())
        try: torch.testing.assert_close(x,y,atol=.02,rtol=.02)
        except AssertionError: failures.append(name)
    assert not failures,f"embedding precision failures: {failures}"

@pytest.mark.parametrize("oracle_kind",("torch","same_embedding"))
@pytest.mark.parametrize("direction",("q_from_k","k_from_q"))
@pytest.mark.parametrize("probability",(0.,.37,1.))
@pytest.mark.parametrize("d,dv",itertools.product((32,64,128),repeat=2))
@pytest.mark.parametrize("backend",("triton","cuda","cuda_symmetric"))
def test_autograd_reference(d,dv,direction,probability,oracle_kind,record_property,backend):
    check_reference(d,dv,direction,probability,oracle_kind,record_property,
        embedding_backend="triton" if backend=="triton" else "cuda",embedding_backward_backend=backend)

def check_reference(d,dv,direction,probability,oracle_kind,record_property,n=65,tau=.2,embedding_backend="triton",embedding_backward_backend="triton"):
    ins,dout=inputs(d,dv,n,tau=tau)
    leaves=[x.detach().requires_grad_() for x in ins]
    out,state=voc_dism(*leaves,sm_scale=d**-.5,direction=direction,hard_prob=probability,
        generator=torch.Generator(device="cuda").manual_seed(721),return_rng_state=True,
        embedding_backend=embedding_backend,embedding_backward_backend=embedding_backward_backend)
    actual=torch.autograd.grad(out,leaves,dout)
    ref=[x.float().detach().requires_grad_() for x in ins]
    interp=None
    if oracle_kind=="same_embedding":
        # Same embedding VALUES and labels; FP32 torch embedding Jacobian.
        fp=interpolation_ref(ref[0],ref[1],ref[4],ref[5],d**-.5)
        with torch.no_grad():
            if embedding_backend=="cuda":
                from dism_v2.embedding import forward as embedding_forward
                raw=InterpolationResult(*embedding_forward(ins[0],ins[1],ins[4],ins[5],d**-.5,warp_specialized=True))
            else:
                raw=InterpolationResult(*emb_fwd_wrapper(ins[0],ins[1],ins[4],ins[5],d**-.5))
        values={name:getattr(fp,name)+(getattr(raw,name).float()-getattr(fp,name)).detach()
            for name in ("q_from_k","k_from_q","q_lse","k_lse")}
        interp=replace(fp,**values,q_index=raw.q_index.long(),k_index=raw.k_index.long())
    expected=voc_dism_ref(*ref,sm_scale=d**-.5,direction=state.direction,hard_prob=probability,
        hard_mask=row_mask(state),interpolation=interp)
    rg=torch.autograd.grad(expected,ref,dout.float())
    failures=[]
    for name,x,y in zip(("out",)+NAMES,(out,)+actual,(expected,)+rg):
        diff=x.double()-y.double();den=y.double().norm()
        record_property(name+"_max_abs",diff.abs().max().item())
        record_property(name+"_relative_l2",(diff.norm()/den).item() if den>0 else diff.norm().item())
        record_property(name+"_cosine",torch.nn.functional.cosine_similarity(
            x.double().flatten(),y.double().flatten(),dim=0).item() if den>0 else 1.)
        if name=="rtau":
            flips=int(((x*y<0)&(y.abs()>1e-5)).sum())
            record_property("rtau_sign_flips",flips)
            if flips: failures.append("rtau_sign")
            record_property("rtau_near_zero",int((y.abs()<=1e-5).sum()))
            record_property("rtau_actual",str(x.tolist()))
            record_property("rtau_oracle",str(y.tolist()))
        try: torch.testing.assert_close(x.float(),y,atol=.008 if name=="out" else .02,
            rtol=.012 if name=="out" else .02)
        except AssertionError: failures.append(name)
    assert not failures,f"reference precision failures: {failures}"

@pytest.mark.parametrize("n,probability",((17,0.),(139,0.),(139,.37),(513,0.)))
@pytest.mark.parametrize("direction",("q_from_k","k_from_q"))
@pytest.mark.parametrize("oracle_kind",("torch","same_embedding"))
@pytest.mark.parametrize("backend",("triton","cuda","cuda_symmetric"))
def test_autograd_reference_tau_bound(n,probability,direction,oracle_kind,record_property,backend):
    check_reference(64,128,direction,probability,oracle_kind,record_property,n,math.log(64),
        embedding_backend="triton" if backend=="triton" else "cuda",embedding_backward_backend=backend)

@pytest.mark.parametrize("backend",("triton","cuda","cuda_symmetric"))
def test_training_backward(backend):
    ins,_=inputs(32,64,65)
    leaves=[x.detach().requires_grad_() for x in ins]
    for step in range(3):
        out=voc_dism(*leaves,sm_scale=32**-.5,hard_prob=.37,
            embedding_backend="cuda",embedding_backward_backend=backend)
        out.float().square().mean().backward()
        for x in leaves:
            assert x.grad is not None and x.grad.dtype==x.dtype and torch.isfinite(x.grad).all()
        with torch.no_grad():
            for x in leaves:
                x.add_(x.grad,alpha=-1e-3)
                x.grad=None

@pytest.mark.parametrize("n,vocab",((1,1),(17,31),(63,64),(129,129),(257,257),(1025,65)))
@pytest.mark.parametrize("backend",("triton","cuda","cuda_symmetric"))
def test_replay_tails_and_partial_grad(n,vocab,backend):
    ins,dout=inputs(64,32,n,vocab)
    leaves=[x.detach().requires_grad_() for x in ins]
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    kw=dict(embedding_backend="triton" if backend=="triton" else "cuda",embedding_backward_backend=backend)
    with torch.cuda.stream(stream):
        out,state=voc_dism(*leaves,sm_scale=64**-.5,hard_prob=.37,return_rng_state=True,**kw)
        g=torch.autograd.grad(out,leaves,dout)
    torch.cuda.current_stream().wait_stream(stream)
    before=torch.cuda.get_rng_state()
    replay=voc_dism(*leaves,sm_scale=64**-.5,hard_prob=.37,rng_state=state,**kw)
    again=torch.autograd.grad(replay,leaves,dout)
    assert torch.equal(before,torch.cuda.get_rng_state())
    torch.testing.assert_close(out,replay,atol=0,rtol=0)
    for x,y in zip(g,again): torch.testing.assert_close(x,y,atol=.002,rtol=.008)
    only_tau=list(ins);only_tau[3]=ins[3].detach().requires_grad_()
    out_tau=voc_dism(*only_tau,sm_scale=64**-.5,hard_prob=.37,rng_state=state,**kw)
    gt,=torch.autograd.grad(out_tau,only_tau[3],dout)
    torch.testing.assert_close(gt,g[3],atol=3e-5,rtol=3e-5)

def test_autograd_contract():
    ins,dout=inputs(32,64,17)
    with pytest.raises(ValueError,match="embedding_backward_backend"):
        voc_dism(*ins,embedding_backward_backend="invalid")
    with pytest.raises(TypeError,match="BF16"): voc_dism(ins[0].float(),*ins[1:])
    with pytest.raises(ValueError,match="contiguous"): voc_dism(ins[0].transpose(0,1),*ins[1:])
    with pytest.raises(ValueError,match="hard_prob"): voc_dism(*ins,hard_prob=[.2])
    leaves=[x.detach().requires_grad_() for x in ins]
    out=voc_dism(*leaves)
    with pytest.raises(NotImplementedError,match="higher-order"):
        torch.autograd.grad(out,leaves,dout,create_graph=True)
    out=voc_dism(*leaves)
    old=torch.are_deterministic_algorithms_enabled()
    warn=torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        with pytest.raises(RuntimeError,match="nondeterministic"):
            torch.autograd.grad(out,leaves,dout)
    finally: torch.use_deterministic_algorithms(old,warn_only=warn)

@pytest.mark.parametrize("d",(32,64,128))
def test_embedding_codegen(d,monkeypatch,record_property):
    from dism_v2 import emb_kernel as emb
    seen={}
    for fn in (emb.emb_fwd,emb._interp_bwd_preprocess,emb._interp_bwd):
        original=fn.run
        def capture(*args,_original=original,**kwargs):
            kernel=_original(*args,**kwargs)
            seen[kernel.name]=kernel
            return kernel
        monkeypatch.setattr(fn,"run",capture)
    (q,k,_,_,qv,kv),_=inputs(d,32,65,129)
    raw=emb.emb_fwd_wrapper(q,k,qv,kv,d**-.5)
    zero=torch.zeros_like(q,dtype=torch.float32)
    emb.emb_bwd_wrapper(q,k,qv,kv,*raw[:4],zero,zero,None,None,d**-.5)
    assert len(seen)==3
    resources={}
    for name,kernel in seen.items():
        cubin=next(path for key,path in kernel.metadata_group.items() if key.endswith(".cubin"))
        sass=subprocess.check_output(["/usr/local/cuda/bin/cuobjdump","-sass",cubin],text=True)
        assert not re.search(r'\bCALL\b',sass)
        resources[name]=dict(registers=kernel.n_regs,spills=kernel.n_spills,
            shared=kernel.metadata.shared,num_stages=kernel.metadata.num_stages)
    record_property("embedding_resources",json.dumps(resources))
