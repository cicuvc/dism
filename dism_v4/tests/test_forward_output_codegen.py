"""Both output precisions must coexist without an output-precision device flag."""
from pathlib import Path
import re
import subprocess

import pytest
import cu_flash_dism as cu


@pytest.mark.parametrize('name', ['forward', 'varlen_forward', 'output_store_probe'])
def test_forward_precision_instances(name):
    root = Path(__file__).resolve().parents[1]
    binary = root / f'build/object/r32_d64_v64/{name}.cu.dev.sm120a.o'
    sass = subprocess.check_output(
        ['/usr/local/cuda/bin/cuobjdump', '-sass', str(binary)], text=True)
    functions = re.findall(r'Function\s*:\s*(\S+)', sass)
    assert len(functions) == (2 if cu.fp32_enabled() else 1), functions
    assert any('ILb0' in f for f in functions), functions
    assert any('ILb1' in f for f in functions) == cu.fp32_enabled(), functions
    assert 'CALL' not in sass
    types = (root / 'include/forward/types.cuh').read_text()
    assert 'DISM_OUTPUT_BF16' not in types
    assert 'bool fp32_output' not in types
