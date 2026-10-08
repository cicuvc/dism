"""Check the built cubin, not merely source-level inlining/register intent."""
from pathlib import Path
import subprocess
import re
import pytest
import cu_flash_dism


def linked_cubin(root, name):
    return root / f'build/object/r32_d64_v64/{name}.cu.dev.sm120a.o'


def test_summary_codegen():
    root = Path(__file__).resolve().parents[1]
    tool = '/usr/local/cuda/bin/cuobjdump'
    summary_name = 'wmma_tma_preprocess'
    for name in (summary_name, 'summary_probes'):
        binary = linked_cubin(root, name)
        sass = subprocess.check_output([tool, '--dump-sass', str(binary)], text=True)
        assert 'CALL' not in sass
        assert 'MUFU.TANH' in sass
        if name == summary_name:
            assert sass.count('USETMAXREG.DEALLOC') == 1
            assert sass.count('USETMAXREG.TRY_ALLOC') == 1
            assert sass.count('UTMALDG') >= 2


def test_mask_predication_codegen():
    """With lineinfo, check the hard-overwrite PTX's actual SASS."""
    root = Path(__file__).resolve().parents[1]
    binary = linked_cubin(root, 'wmma_tma_preprocess')
    source = (root / 'include/summary/kernel_common.cuh').read_text().splitlines()
    start = next(i + 1 for i, line in enumerate(source) if 'void finish_score_pair(' in line)
    end = next(i + 1 for i, line in enumerate(source)
               if i + 1 > start and line.startswith('template <class Score'))
    sass = subprocess.check_output(['/usr/local/cuda/bin/nvdisasm', '-gi', str(binary)], text=True)
    selected = source_instructions(sass, 'kernel_common.cuh', start, end)
    if not selected:
        pytest.skip('build with DISM_LINEINFO=1 for source-mapped predication check')
    assert any('SEL' in line for line in selected)
    assert not any(re.search(r'\b(?:BRA|BRX|CALL)(?:\.|\s)', line) for line in selected), selected


def source_instructions(sass, filename, start=1, end=100000):
    in_comments = False
    pending = active = False
    selected = []
    for line in sass.splitlines():
        if '//## File' in line:
            if not in_comments:
                pending = False
            in_comments = True
            match = re.search(re.escape(filename) + r'", line (\d+)', line)
            pending |= bool(match and start <= int(match[1]) < end)
        elif re.search(r'/\*[0-9a-f]+\*/', line):
            if in_comments:
                active, in_comments = pending, False
            if active:
                selected.append(line)
    return selected
