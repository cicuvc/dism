"""Isolated Gsoft shared-transpose -> dA MMA -> asynchronous TMA FP32 add."""
from functools import lru_cache
from pathlib import Path
import os
import re
import subprocess
import pytest
import torch
from torch.utils.cpp_extension import load, CUDA_HOME

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")

@lru_cache(None)
def probe():
    root=Path(__file__).resolve().parents[1]
    source=root/"experiments/glx_da_tma"
    return load(name="dism_da_tma_probe",sources=[str(source/f) for f in ("bindings.cpp","probe.cu")],
        extra_include_paths=[str(root/"include")],extra_cflags=["-O2","-std=c++20"],
        extra_cuda_cflags=["-O3","-std=c++20","-lineinfo","--extended-lambda","--expt-relaxed-constexpr",
            "-gencode=arch=compute_120a,code=sm_120a","--ptxas-options=-v"],extra_ldflags=["-lcuda"],
        verbose=os.environ.get("DISM_VERBOSE_BUILD")=="1")

@pytest.mark.parametrize("d",(32,64,128))
@pytest.mark.parametrize("n",(1,17,63,64,65,139,257))
@pytest.mark.parametrize("warps,groups",((1,1),(1,9),(8,19)))
def test_da_tma(d,n,warps,groups,record_property):
    if torch.cuda.get_device_capability()!=(12,0): pytest.skip("sm120a only")
    gen=torch.Generator(device="cuda").manual_seed(717+d+n+groups)
    g=torch.randn((groups,16,n),device="cuda",generator=gen)
    g[...,::7]=0 # Simulated hard query columns; input here is already Gsoft.
    b=torch.randn((groups,16,d),device="cuda",dtype=torch.bfloat16,generator=gen)
    backing=torch.full((n*d+256,),12345.,device="cuda")
    out=backing[:n*d].view(n,d);out.fill_(.125)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream): probe().run(g,b,out,warps,d**-.5)
    torch.cuda.current_stream().wait_stream(stream)
    expected=.125+(g.bfloat16().double().transpose(-1,-2)@b.double()).sum(0)*d**-.5
    error=(out.double()-expected).abs().max().item()
    record_property("max_abs",error)
    ideal=.125+(g.double().transpose(-1,-2)@b.double()).sum(0)*d**-.5
    record_property("fp32_g_max_abs",(out.double()-ideal).abs().max().item())
    record_property("fp32_g_relative_l2",((out.double()-ideal).norm()/ideal.norm()).item())
    record_property("fp32_g_cosine",torch.nn.functional.cosine_similarity(out.double().flatten(),ideal.flatten(),dim=0).item())
    torch.testing.assert_close(out.double(),expected,atol=3e-5,rtol=3e-5)
    assert torch.all(backing[n*d:]==12345)

def test_codegen():
    path=probe().__file__
    sass=subprocess.check_output([str(Path(CUDA_HOME)/"bin/cuobjdump"),"-sass",path],text=True)
    assert not re.search(r"\b(?:CALL|LDL|STL)\b",sass)
    assert "LDSM.16.MT88" in sass and "HMMA" in sass
    assert "UTMAREDG.2D.ADD" in sass and "UTMACMDFLUSH" in sass
    assert not re.search(r"\b(?:ATOM|RED)\.",sass) # No scalar global atomic fallback.
