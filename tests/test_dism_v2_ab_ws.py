"""12-warp B3: independent G/GEMM oracle, tails, raw reference and codegen."""
import itertools
import os
import re
import subprocess
from dataclasses import replace
from pathlib import Path
import pytest
import torch
from torch.utils.cpp_extension import CUDA_HOME
from dism_v2.backward import _extension
from test_dism_v2_dv import run,exact_oracle_matmul,pytestmark

def test_ab_ws_contract():
    from dism_v2.core import forward
    from dism_v2.backward import delta,value_gradient,operand_gradient
    if torch.cuda.get_device_capability()!=(12,0): pytest.skip("sm120a only")
    a=torch.zeros((1,1,17,32),device="cuda",dtype=torch.bfloat16)
    b=a.clone();v=a.clone();dout=a.clone()
    lse=torch.ones((1,1,17),device="cuda");tau=torch.zeros(1,device="cuda")
    labels=torch.zeros((1,1,17),device="cuda",dtype=torch.int64)
    out,norm,edges,state=forward(a,b,v,lse,tau,labels,labels,sm_scale=32**-.5,direction="q_from_k",
        hard_prob=0.,save_boundaries=True,return_rng_state=True)
    dd=delta(dout,out)
    kw=dict(sm_scale=32**-.5,rng_state=state)
    _,_,boundary=value_gradient(a,b,dout,lse,tau,labels,labels,norm,edges,**kw,v=v,delta=dd,warp_specialized=True)
    import inspect
    assert inspect.signature(operand_gradient).parameters["warp_specialized"].default is True
    args=[a,b,v,dout,lse,tau,labels,labels,norm,dd,edges,boundary]
    result=operand_gradient(*args,**kw)
    assert all(torch.count_nonzero(x)==0 for x in result)
    with pytest.raises(ValueError,match="offset"):
        operand_gradient(*args,**dict(kw,rng_state=replace(state,offset=1)))
    with pytest.raises(ValueError,match="shape"):
        operand_gradient(*args,**dict(kw,rng_state=replace(state,shape=(1,1,1))))
    with pytest.raises(TypeError,match="ScanBoundaries"):
        operand_gradient(*args[:10],None,boundary,**kw)
    for index,value,pattern in ((0,a.float(),"BF16"),(2,v.float(),"V must"),
            (9,dd.bfloat16(),"delta"),(11,boundary[:,:,:0,:],"G32")):
        bad=args.copy();bad[index]=value
        with pytest.raises(RuntimeError,match=pattern): operand_gradient(*bad,**kw)
    with pytest.raises(NotImplementedError,match="higher-order"):
        operand_gradient(a.detach().requires_grad_(),*args[1:],**kw)
    previous=torch.are_deterministic_algorithms_enabled()
    warn=torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        with pytest.raises(RuntimeError,match="nondeterministic"): operand_gradient(*args,**kw)
    finally: torch.use_deterministic_algorithms(previous,warn_only=warn)

@pytest.mark.parametrize("d,dv",itertools.product((32,64,128),repeat=2))
@pytest.mark.parametrize("direction",("q_from_k","k_from_q"))
@pytest.mark.parametrize("probability",(0.,.37,1.))
def test_ab_dimensions(d,dv,direction,probability,record_property):
    run(d,dv,139,direction,probability,record_property,
        check_summary=True,warp_specialized=True,check_g=True,check_ab=True,ab_warp_specialized=True)

@pytest.mark.parametrize("n",(1,17,31,32,63,64,65,129,513,1025,2049))
@pytest.mark.parametrize("mode",("chain","break","bounded_soft"))
def test_ab_tails(n,mode,record_property):
    run(64,128,n,"random",0. if mode=="bounded_soft" else 1.,record_property,mode,
        check_summary=True,warp_specialized=True,check_g=True,check_ab=True,ab_warp_specialized=True)

@pytest.mark.parametrize("d,dv",itertools.product((32,64,128),repeat=2))
@pytest.mark.parametrize("direction",("q_from_k","k_from_q"))
@pytest.mark.parametrize("probability",(0.,.37,1.))
def test_ab_reference_dimensions(d,dv,direction,probability,record_property):
    run(d,dv,139,direction,probability,record_property,
        check_summary=True,warp_specialized=True,check_g=True,check_ab=True,ab_warp_specialized=True,check_reference=True)

@pytest.mark.parametrize("n",(17,65,139))
@pytest.mark.parametrize("direction",("q_from_k","k_from_q"))
def test_ab_reference_bounded_soft(n,direction,record_property):
    run(64,128,n,direction,0.,record_property,"bounded_soft",
        check_summary=True,warp_specialized=True,check_g=True,check_ab=True,ab_warp_specialized=True,check_reference=True)

def test_codegen():
    optimization=int(os.environ.get('DISM_BWD_OPT','0'))
    sass=subprocess.check_output([str(Path(CUDA_HOME)/"bin/cuobjdump"),"-sass",_extension().__file__],text=True)
    count=0
    for block in sass.split('Function : ')[1:]:
        dims=re.search(r'_ZN7dism_v25ab_ws3runILi(\d+)ELi(\d+)',block.splitlines()[0])
        if dims is None: continue
        count+=1
        assert not re.search(r'\bCALL\b',block)
        hard_only=False
        if optimization>=11:
            spec=int(re.search(r'ILi\d+ELi\d+ELi(\d+)E',block.splitlines()[0]).group(1))
            hard_only=spec%5 in (1,2)
        assert 'UTMALDG.5D' in block
        assert ('UTMAREDG.3D.ADD' in block)==(not hard_only)
        assert re.search(r'USETMAXREG\.DEALLOC\.CTAPOOL\s+0x28',block)
        assert re.search(r'USETMAXREG\.TRY_ALLOC\.CTAPOOL\s+UP\d+, 0xe8',block)
        assert 'MUFU.TANH' in block
        assert ('LDSM.16.' if hard_only else 'LDSM.16.MT88') in block
        if optimization==0 and int(dims[1])<128 and (int(dims[1]),int(dims[2]))!=(64,128):
            assert not re.search(r'\b(?:LDL|STL)\b',block)
        # Legacy resource gate only. Optimization experiments explicitly allow
        # spill; test_ws_codegen_with_spill_report records every instance.
    assert count==(90 if optimization>=10 else 9)

def test_tau_codegen():
    sass=subprocess.check_output([str(Path(CUDA_HOME)/"bin/cuobjdump"),"-sass",_extension().__file__],text=True)
    blocks=[b for b in sass.split("Function : ")[1:]
        if "_ZN7dism_v210scalar_bwd10reduce_tau" in b.splitlines()[0]]
    assert len(blocks)==1
    assert not re.search(r'\b(?:CALL|LDL|STL)\b',blocks[0])
