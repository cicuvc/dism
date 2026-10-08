"""The default extension links production entrypoints only, not stale probe objects."""
import json
from pathlib import Path
import subprocess

import pytest
import cu_flash_dism as cu


@pytest.mark.skipif(hasattr(cu,'scan_probe'),reason='production-only build assertion')
def test_production_api_and_link_inputs():
    required = {
        'summarization','chunk_scan','forward_output',
        'backward_delta','backward_summary','backward_chunk','backward_qk',
        'varlen_summary','varlen_chunk','varlen_forward','varlen_delta',
        'varlen_backward_summary','varlen_backward_chunk','varlen_backward_qk',
        'summary_key_dim','forward_head_dim','forward_readout_dim',
    }
    for backend in cu.configs.values():
        public={name for name in dir(backend) if not name.startswith('_')}
        # Explicit debug builds may additionally expose dense gradient diagnostics.
        assert public-required <= {'backward_debug_enabled','backward_summary_debug','backward_qk_debug'}
        assert required <= public
    root=Path(__file__).resolve().parents[1]
    commands=json.loads((root/'build/compile_commands.json').read_text())
    assert not any('/probe/' in str(command['file']) for command in commands)
    symbols=subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','-symbols',cu.__file__],text=True)
    assert 'probe_kernel' not in symbols
    assert 'pack_kernel' not in symbols

