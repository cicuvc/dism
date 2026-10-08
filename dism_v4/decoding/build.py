"""Standalone inference extension build (no dependency on the training binary).

Example on this workstation:
  CUDA_HOME=/usr/local/cuda-13.4 python dism_v4/decoding/build.py
Use the blkw environment. MAX_JOBS defaults to 2; TORCH_CUDA_ARCH_LIST can
override the active GPU architecture. Device translation units are Torch-free.
"""
import os
from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


@lru_cache(None)
def load_cuda():
    root=Path(__file__).parent
    major,minor=torch.cuda.get_device_capability()
    os.environ.setdefault('TORCH_CUDA_ARCH_LIST',f'{major}.{minor}')
    os.environ.setdefault('MAX_JOBS','2')
    fp64=os.environ.get('DISM_DECODE_FP64_PREFIX','0')=='1'
    define=f'-DDISM_DECODE_FP64_PREFIX={int(fp64)}'
    timing=os.environ.get('DISM_DECODE_TIMING','0')=='1'
    name='dism_decode_cuda_fp64' if fp64 else 'dism_decode_cuda'
    return load(name=name+('_timing' if timing else ''),
                sources=[str(root/name) for name in ('cuda_bind.cpp','cuda_kernels.cu','rebuild_parallel.cu','linear.cu',
                                                   'gpu_planner_bind.cpp','gpu_planner.cu')],
                extra_cflags=['-O3',define]+(['-DDISM_DECODE_TIMING=1'] if timing else []),
                extra_cuda_cflags=['-O3','-lineinfo',define],verbose=False)


if __name__=='__main__':
    module=load_cuda()
    print(module.__file__)
