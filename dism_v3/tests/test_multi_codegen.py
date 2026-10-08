"""Audit the actual fatbinary inputs, not stale single-shape build artifacts."""
from pathlib import Path
import re
import subprocess

import pytest

from test_multi_config import CONFIGS

ROOT = Path(__file__).resolve().parents[1]
PRIMARY = {
    'wmma_tma_preprocess': 'wmma_tma_wasp_persistent_summarization_kernel',
    'chunk_scan': 'chunk_scan_async_kernel',
    'forward': 'dism_forward_kernel',
    'backward_summary': 'dism_backward_summary_kernel',
    'backward_chunk': 'reverse_chunk_async',
    'backward_qk': 'dism_backward_qk_kernel',
    'varlen_summary': 'varlen_summary_kernel',
    'varlen_chunk': 'varlen_chunk_kernel',
    'varlen_forward': 'varlen_forward_kernel',
    'varlen_backward_summary': 'varlen_backward_summary_kernel',
    'varlen_backward_qk': 'varlen_backward_qk_kernel',
}


@pytest.mark.parametrize('config', CONFIGS)
def test_twelve_primary_instances(config):
    r,d,dv = config
    tag = f'r{r}_d{d}_v{dv}'
    count = 0
    for source, kernel in PRIMARY.items():
        obj = ROOT / f'build/object/{tag}/{source}.cu.dev.sm120a.o'
        sass = subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','-sass',str(obj)],text=True)
        functions = re.findall(r'Function\s*:\s*(\S+)',sass)
        selected = [f for f in functions if kernel in f]
        assert len(selected) == (2 if source == 'varlen_chunk' else 1), selected
        assert all(f'dism_{tag}' in f and 'KernelConfig' in f for f in selected), selected
        assert 'CALL' not in sass, (config,source)
        dependency = (ROOT / f'build/deps/{tag}/{source}.cu.d').read_text()
        assert '/torch/include/' not in dependency, source
        count += len(selected)
    assert count == 12
