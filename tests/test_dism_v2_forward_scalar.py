"""Isolated codegen/numeric checks for forward EX2 FTZ and mixed-row RNG."""
from functools import lru_cache
from pathlib import Path
import os
import re
import subprocess

import pytest
import torch
from torch.utils.cpp_extension import CUDA_HOME, load_inline

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@lru_cache(None)
def extension():
    root=Path(__file__).resolve().parents[1]
    return load_inline(name='dism_v2_forward_scalar_probe',
        cpp_sources='torch::Tensor scalar_ex2(torch::Tensor);\n'
                    'torch::Tensor scalar_rng(torch::Tensor, bool);',
        cuda_sources=r'''
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include "log_affine.cuh"
#include "row_rng.cuh"
__global__ void ex2_probe(const float* x,float* y,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) y[i]=dism_v2::exp2_ftz(x[i]);
}
template<bool MIXED> __global__ void rng_probe(const float* p,int* y,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) y[i]=dism_v2::row_hard<MIXED>(0x123456789abcdefULL,28,i,p[i]);
}
torch::Tensor scalar_ex2(torch::Tensor x) {
    c10::cuda::CUDAGuard guard(x.device()); auto y=torch::empty_like(x);
    ex2_probe<<<(x.numel()+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(x.data_ptr<float>(),y.data_ptr<float>(),x.numel());
    return y;
}
torch::Tensor scalar_rng(torch::Tensor p,bool mixed) {
    c10::cuda::CUDAGuard guard(p.device()); auto y=torch::empty_like(p,p.options().dtype(torch::kInt32));
    if(mixed) rng_probe<true><<<(p.numel()+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(p.data_ptr<float>(),y.data_ptr<int>(),p.numel());
    else rng_probe<false><<<(p.numel()+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(p.data_ptr<float>(),y.data_ptr<int>(),p.numel());
    return y;
}
''', functions=['scalar_ex2','scalar_rng'],
        extra_include_paths=[str(root/'dism_v2/csrc'),
            str(Path(os.environ.get('GLX_ROOT','/home/cicuvc/cs/projects/glx'))/'include')],
        extra_cflags=['-O2','-std=c++20'],extra_cuda_cflags=['-O3','-std=c++20',
            '-gencode=arch=compute_120a,code=sm_120a'])


def test_ex2_ftz_numeric():
    x=torch.tensor([-float('inf'),-150.,-149.,-127.,-126.00001,-126.,-125.99999,
                    -100.,-1.,0.,1.,20.,127.,128.,float('inf'),float('nan')],device='cuda')
    y=extension().scalar_ex2(x)
    expected=torch.exp2(x)
    expected=torch.where(expected<torch.finfo(torch.float32).tiny,0.,expected)
    torch.testing.assert_close(y,expected,atol=0.,rtol=2e-6,equal_nan=True)
    assert (y[:5]==0).all()
    assert y[5]==torch.finfo(torch.float32).tiny


def test_mixed_rng_same_decisions():
    p=torch.tensor([0.,1.,.37,.5,1.-2**-24,2**-24],device='cuda').repeat(4096)
    x=extension().scalar_rng(p,True)
    torch.testing.assert_close(x,extension().scalar_rng(p,False),atol=0,rtol=0)
    assert not x[::6].any()
    assert x[1::6].all()


def test_forward_scalar_codegen():
    s=subprocess.check_output([str(Path(CUDA_HOME)/'bin/cuobjdump'),'--dump-sass',extension().__file__],text=True)
    parts=re.split(r'Function : (\S+)',s)
    bodies={parts[i]:parts[i+1] for i in range(1,len(parts),2)}
    ex=next(b for n,b in bodies.items() if 'ex2_probe' in n)
    rng=next(b for n,b in bodies.items() if 'rng_probeILb1E' in n)
    assert ex.count('MUFU.EX2')==1
    assert not re.search(r'\b(?:FMUL|FADD|FFMA|CALL)(?:\.|\s)',ex)
    assert rng.count('FSETP.')==1 # Only the uniform<probability comparison.
    assert not re.search(r'\bCALL(?:\.|\s)',rng)
