"""Run existing numerical gates against a private algorithm diagnostic module.

Example: --variant tau_vectors -- -q tests/test_hard_prefill_backward_fused.py
No installed kernel file is rewritten; the substitution lasts one process.
"""
import argparse
from pathlib import Path
import sys

import torch
import pytest

import bench_planner_event as bench
from planner_algorithm_variants import ALGORITHM_VARIANTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', required=True, choices=ALGORITHM_VARIANTS)
    parser.add_argument('--baseline', default='8d47ebf326abe736a1b17bf41eae1cf9e3e13141')
    parser.add_argument('pytest_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    for module in (bench.flash_dism, bench.cu_flash_dism):
        if not Path(module.__file__).resolve().is_relative_to(bench.ROOT/'dism_v3/python'):
            raise RuntimeError('foreign module')
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(.09)
    bench.BASE = args.baseline
    directory = bench.ROOT/'build/planner-algorithm/check-variants'
    directory.mkdir(parents=True, exist_ok=True)
    module = bench.load_variant(args.variant, directory)
    sys.modules['flash_dism.kernels.hard_backward_fused'] = module
    import flash_dism.kernels
    flash_dism.kernels.hard_backward_fused = module
    forwarded = args.pytest_args
    if forwarded[:1] == ['--']:
        forwarded = forwarded[1:]
    status = pytest.main(forwarded)
    print(f'Peak PyTorch allocated: {torch.cuda.max_memory_allocated()/2**20:.1f} MiB')
    raise SystemExit(status)


if __name__ == '__main__':
    main()
