"""Real transposed MMA + query-column RNG + sparse-boundary rescan (diagnostic)."""
from functools import lru_cache
from pathlib import Path
import os
import re
import subprocess
import math

import pytest
import torch
from torch.utils.cpp_extension import load, CUDA_HOME
from dism_v2.core import forward, RowRNGState

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")


@lru_cache(None)
def probe():
    root=Path(__file__).resolve().parents[1]
    source=root/"experiments/glx_recompute"
    glx=Path(os.environ.get("GLX_ROOT","/home/cicuvc/cs/projects/glx"))
    return load(name="dism_v2_recompute_probe",sources=[str(source/f) for f in ("bindings.cpp","recompute.cu")],
        extra_include_paths=[str(root/"include"),str(glx/"include")],
        extra_cflags=["-O2","-std=c++20"],
        extra_cuda_cflags=["-O3","-std=c++20","-lineinfo","--extended-lambda",
            "--expt-relaxed-constexpr","-gencode=arch=compute_120a,code=sm_120a","--ptxas-options=-v"],
        extra_ldflags=["-lcuda"])


def word(seed,offset,row):
    mask=2**32-1
    a,b,c,d=(offset//4)&mask,(offset//4)>>32,row&mask,row>>32
    k0,k1=seed&mask,seed>>32
    for _ in range(10):
        p0=0xD2511F53*a; p1=0xCD9E8D57*c
        a,b,c,d=(p1>>32)^b^k0,p1&mask,(p0>>32)^d^k1,p0&mask
        k0,k1=(k0+0x9E3779B9)&mask,(k1+0xBB67AE85)&mask
    return a


@pytest.mark.parametrize("d",(32,64,128))
@pytest.mark.parametrize("direction",("q_from_k","k_from_q"))
@pytest.mark.parametrize("mode",("soft","mixed","break","chain"))
@pytest.mark.parametrize("n",(17,65,139,257))
@torch.no_grad()
def test_real_transposed_recompute(d,direction,mode,n):
    if torch.cuda.get_device_capability()!=(12,0):
        pytest.skip("sm120a only")
    batch,heads=2,2
    g=torch.Generator(device="cuda").manual_seed(912+d+n)
    def rand(shape):
        return torch.randn(shape,device="cuda",dtype=torch.bfloat16,generator=g)
    a,b=rand((batch,heads,n,d)),rand((batch,heads,n,d))
    v=rand((batch,heads,n,32))
    lse=torch.full((batch,heads,n),3.,device="cuda")
    tau=torch.tensor([-.4,.2],device="cuda")
    ql=torch.arange(n,device="cuda").remainder(7).expand(batch,heads,n).contiguous()
    kl=ql.clone()
    p={"soft":0.,"mixed":.37,"break":1.,"chain":1.}[mode]
    if mode=="break": kl+=8
    if mode=="chain": ql.zero_(); kl.zero_(); tau.fill_(math.log(d))
    state=RowRNGState(2**63+1245,2**34+16,(batch,heads,n),direction,p)
    out,norm,edges=forward(a,b,v,lse,tau,ql,kl,sm_scale=d**-.5,direction=direction,
        hard_prob=p,rng_state=state,save_boundaries=True)
    rng_before=torch.cuda.get_rng_state()
    w=probe().recompute(a,b,lse,tau,ql,kl,edges.vertical,edges.horizontal,d**-.5,
        direction=="k_from_q",p,state.seed,state.offset)
    assert torch.equal(rng_before,torch.cuda.get_rng_state())
    # FP64 dot and independent diagonal recurrence, including identity padding.
    logs=a.double()@b.double().transpose(-1,-2)*d**-.5
    logs-=lse.double().unsqueeze(-1 if direction=="q_from_k" else -2)
    logs+=tau.double()[None,:,None,None]
    prob32=torch.tensor(p,dtype=torch.float32).item()
    hard=torch.tensor([(word(state.seed,state.offset,r)>>8)*2**-24<prob32
        for r in range(batch*heads*n)],device="cuda").reshape(batch,heads,n,1)
    hard_logs=torch.where(ql[...,None]==kl[...,None,:],tau.double()[None,:,None,None],-torch.inf)
    logs=torch.where(hard,hard_logs,logs)
    logs.masked_fill_(torch.arange(n,device="cuda")[None,:]>torch.arange(n,device="cuda")[:,None],-torch.inf)
    np=w.shape[-1]
    previous=torch.full((batch,heads,np),-torch.inf,device="cuda",dtype=torch.float64)
    for i in range(np):
        shifted=torch.nn.functional.pad(previous[...,:-1],(1,0),value=-torch.inf)
        current=shifted.clone()
        if i<n: current[...,:n]=logs[...,i,:]+torch.logaddexp(shifted[...,:n],torch.zeros_like(shifted[...,:n]))
        previous=current
        torch.testing.assert_close(w[...,i,:].double(),current/math.log(2),atol=5e-5,rtol=3e-5)
    torch.testing.assert_close(w[...,15::16].transpose(-1,-2),edges.vertical,atol=5e-5,rtol=3e-5)
    torch.testing.assert_close(w[...,63::64,:],edges.horizontal,atol=5e-5,rtol=3e-5)


def test_recompute_codegen():
    if torch.cuda.get_device_capability()!=(12,0): pytest.skip("sm120a only")
    tool=str(Path(CUDA_HOME)/"bin/cuobjdump")
    sass=subprocess.check_output([tool,"--dump-sass",probe().__file__],text=True)
    assert "UTMALDG.5D" in sass
    assert not re.search(r"\b(?:CALL|LDL|STL)(?:\.|\s)",sass)
    resources=subprocess.check_output([tool,"--dump-resource-usage",probe().__file__],text=True)
    sizes=re.findall(r"(?:STACK|LOCAL):(\d+)",resources)
    assert sizes and all(int(x)==0 for x in sizes)
