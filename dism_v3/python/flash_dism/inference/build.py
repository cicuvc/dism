"""Independent SAM extensions; installed binaries first, user-cache JIT fallback.

Device translation units are Torch-free. MAX_JOBS defaults to two. Set
CUDA_HOME and TORCH_CUDA_ARCH_LIST before building, including cross builds.
"""
from functools import lru_cache
import importlib
import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


def _load(name, sources, *, cuda=False, flags=(), force=False):
    if not force:
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError as exc:
            if exc.name != name:
                raise
    os.environ.setdefault('MAX_JOBS', '2')
    if cuda and 'TORCH_CUDA_ARCH_LIST' not in os.environ:
        if not torch.cuda.is_available():
            raise RuntimeError('Set TORCH_CUDA_ARCH_LIST when building without a visible GPU')
        major, minor = torch.cuda.get_device_capability()
        os.environ['TORCH_CUDA_ARCH_LIST'] = f'{major}.{minor}'
    root = Path(__file__).parent / 'csrc'
    return load(name=name, sources=[str(root / s) for s in sources],
                extra_cflags=['-O3', *flags],
                extra_cuda_cflags=['-O3', '-lineinfo', *flags],
                with_cuda=cuda, verbose=False)


@lru_cache(None)
def load_prefill():
    return _load('_dism_prefill', ['prefill_bind.cpp'])


@lru_cache(None)
def load_cuda():
    fp64 = os.environ.get('DISM_DECODE_FP64_PREFIX', '0') == '1'
    timing = os.environ.get('DISM_DECODE_TIMING', '0') == '1'
    name = 'dism_decode_cuda' + ('_fp64' if fp64 else '') + ('_timing' if timing else '')
    return build_decoder(name, fp64=fp64, timing=timing)


def build_decoder(name='dism_decode_cuda', *, fp64=False, timing=False, force=False):
    flags = [f'-DDISM_DECODE_FP64_PREFIX={int(fp64)}']
    if timing:
        flags.append('-DDISM_DECODE_TIMING=1')
    return _load(name, ['cuda_bind.cpp', 'cuda_kernels.cu', 'rebuild_parallel.cu',
                       'linear.cu', 'gpu_planner_bind.cpp', 'gpu_planner.cu'],
                 cuda=True, flags=flags, force=force)


if __name__ == '__main__':
    import argparse
    import shutil
    parser = argparse.ArgumentParser()
    parser.add_argument('--component', choices=('prefill', 'decode'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    module = (_load('_dism_prefill', ['prefill_bind.cpp'], force=True)
              if args.component == 'prefill' else build_decoder(force=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(module.__file__, args.output)
