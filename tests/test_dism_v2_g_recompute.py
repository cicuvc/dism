"""Diagnostic real W/dP/E -> reverse scan from production G32 boundaries."""
from functools import lru_cache
from pathlib import Path
import itertools
import os
import re
import subprocess
import torch
import pytest
from torch.utils.cpp_extension import load,CUDA_HOME
from test_dism_v2_dv import run,exact_oracle_matmul,pytestmark

@lru_cache(None)
def probe():
    root=Path(__file__).resolve().parents[1];source=root/"experiments/glx_g_recompute"
    return load(name="dism_g_recompute_probe",sources=[str(source/f) for f in ("bindings.cpp","probe.cu")],
        extra_include_paths=[str(root/"include"),str(Path(os.environ.get("GLX_ROOT","/home/cicuvc/cs/projects/glx"))/"include")],
        extra_cflags=["-O2","-std=c++20"],extra_cuda_cflags=["-O3","-std=c++20","-lineinfo","--extended-lambda",
            "--expt-relaxed-constexpr","-gencode=arch=compute_120a,code=sm_120a","--ptxas-options=-v"],
        extra_ldflags=["-lcuda"],verbose=os.environ.get("DISM_VERBOSE_BUILD")=="1")

@pytest.mark.parametrize("d,dv",itertools.product((32,64,128),repeat=2))
@pytest.mark.parametrize("direction",("q_from_k","k_from_q"))
@pytest.mark.parametrize("probability",(0.,.37,1.))
def test_g_dimensions(d,dv,direction,probability,record_property):
    run(d,dv,139,direction,probability,record_property,check_summary=True,warp_specialized=True,check_g=True)

@pytest.mark.parametrize("n",(1,17,31,32,63,64,65,129,513,1025))
@pytest.mark.parametrize("mode",("chain","break","bounded_soft"))
def test_g_tails(n,mode,record_property):
    run(64,128,n,"random",0. if mode=="bounded_soft" else 1.,record_property,mode,
        check_summary=True,warp_specialized=True,check_g=True)

def test_codegen():
    sass=subprocess.check_output([str(Path(CUDA_HOME)/"bin/cuobjdump"),"-sass",probe().__file__],text=True)
    assert not re.search(r"\b(?:CALL|LDL|STL)\b",sass)
    assert "UTMALDG.5D" in sass and "MUFU.TANH" in sass

def check_gemm(g,a,b,mask,scale,record_property,fused_ab=None):
    from test_dism_v2_da_tma import probe as gemm_probe
    n,d=a.shape[-2:];np=g.shape[-1]
    errors=[0.,0.]
    quant_rel=[0.,0.];quant_cos=[1.,1.]
    fused_errors=[0.,0.]
    for bh in range(a.shape[0]*a.shape[1]):
        aa=a.flatten(0,1)[bh];bb=b.flatten(0,1)[bh]
        soft=g.flatten(0,1)[bh,:n,:]*(~mask.flatten(0,1)[bh])
        grouped=soft.T.contiguous().reshape(np//16,16,n)
        keys=torch.nn.functional.pad(bb,(0,0,0,np-n)).view(np//16,16,d)
        output=torch.zeros((n,d),device=a.device,dtype=torch.float32)
        gemm_probe().run(grouped,keys,output,8,scale)
        db=gemm_probe().run_db(grouped,aa,scale)
        quant=grouped.bfloat16().double()
        expected_a=(quant.transpose(-1,-2)@keys.double()).sum(0)*scale
        expected_b=(quant@aa.double())*scale
        if fused_ab is not None:
            for i,y in enumerate((expected_a,expected_b.reshape(np,d)[:n])):
                x=fused_ab[i].flatten(0,1)[bh]
                fused_errors[i]=max(fused_errors[i],(x.double()-y).abs().max().item())
                torch.testing.assert_close(x.double(),y,atol=5e-4,rtol=2e-3)
        for i,(x,y) in enumerate(((output,expected_a),(db,expected_b))):
            errors[i]=max(errors[i],(x.double()-y).abs().max().item())
            torch.testing.assert_close(x.double(),y,atol=3e-5,rtol=3e-5)
        fp32_a=(grouped.double().transpose(-1,-2)@keys.double()).sum(0)*scale
        fp32_b=(grouped.double()@aa.double())*scale
        for i,(x,y) in enumerate(((output,fp32_a),(db,fp32_b))):
            if y.norm()>0:
                quant_rel[i]=max(quant_rel[i],((x.double()-y).norm()/y.norm()).item())
                quant_cos[i]=min(quant_cos[i],torch.nn.functional.cosine_similarity(x.double().flatten(),y.flatten(),dim=0).item())
    record_property("da_same_g_max_abs",errors[0]);record_property("db_same_g_max_abs",errors[1])
    if fused_ab is not None:
        record_property("fused_da_max_abs",fused_errors[0]);record_property("fused_db_max_abs",fused_errors[1])
    for i,name in enumerate(("da","db")):
        record_property(name+"_fp32_g_relative_l2",quant_rel[i])
        record_property(name+"_fp32_g_cosine",quant_cos[i])
