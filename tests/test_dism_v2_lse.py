"""Independent guard for rl/lse.cu::approx and full cross-chunk LSE."""
import os
import math
from pathlib import Path
import subprocess

import pytest
import torch
from torch.utils.cpp_extension import CUDA_HOME
from dism_v2.kernel_config import TILE_LSE


@pytest.fixture(autouse=True)
def exact_matmul():
    previous=torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32=False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32=previous


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_tanh_lse_formula_and_identity(tmp_path):
    root=Path(__file__).resolve().parents[1]
    binary=tmp_path/"lse"
    glx=Path(os.environ.get("GLX_ROOT","/home/cicuvc/cs/projects/glx"))
    subprocess.run([str(Path(CUDA_HOME)/"bin/nvcc"),"-O3","-std=c++20","-arch=sm_120a",
        "-DDISM_TILE_LSE_TANH=1","-I"+str(root/"dism_v2/csrc"),"-I"+str(glx/"include"),
        str(root/"tests/csrc/dism_v2_lse.cu"),"-o",str(binary)],check=True)
    subprocess.run([str(binary)],check=True)


@pytest.mark.skipif(not torch.cuda.is_available() or TILE_LSE not in ('tanh','tanh_finite'),
                   reason='experimental tanh precision regression; run with DISM_TILE_LSE=tanh')
def test_tanh_tau_sign_at_dimension_bound():
    """Known failure remains an ordinary failure in the experimental suite."""
    from dism_v2.autograd import voc_dism
    from dism_v2.embedding import forward as emb_forward
    from dism_v2.dism_ref import InterpolationResult, voc_dism_ref
    d,dv,n=128,32,139
    torch.manual_seed(731+d+dv+n)
    def rand(shape):
        return torch.randn(shape,device='cuda',dtype=torch.bfloat16)
    q,k=rand((1,2,n,d)),rand((1,2,n,d))
    v=rand((1,2,n,dv))
    qv,kv=rand((2,65,d)),rand((2,65,d))
    tau=torch.full((2,),math.log(d),device='cuda',requires_grad=True)
    do=torch.randn_like(v)
    out=voc_dism(q,k,v,tau,qv,kv,sm_scale=d**-.5,direction='q_from_k',hard_prob=0.,
        embedding_backend='cuda',embedding_backward_backend='cuda')
    actual=torch.autograd.grad(out,tau,do)[0]
    with torch.no_grad():
        interp=InterpolationResult(*emb_forward(q,k,qv,kv,d**-.5))
    tr=tau.detach().clone().requires_grad_()
    reference=voc_dism_ref(q.float(),k.float(),v.float(),tr,qv.float(),kv.float(),
        sm_scale=d**-.5,direction='q_from_k',hard_prob=0.,interpolation=interp)
    expected=torch.autograd.grad(reference,tr,do.float())[0]
    flips=(actual*expected<0)&(expected.abs()>1e-5)
    assert not flips.any(),f'tau sign: tanh={actual.tolist()}, oracle={expected.tolist()}'
