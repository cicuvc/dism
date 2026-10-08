"""Keep the README's executable CUDA examples in sync with public APIs."""
from pathlib import Path
import re

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason='README examples require CUDA')
def test_readme_python_examples():
    text = (Path(__file__).parents[1] / 'README.md').read_text()
    blocks = re.findall(r'```python\n(.*?)\n```', text, flags=re.DOTALL)
    assert len(blocks) == 4
    namespace = {'__name__': '__readme__'}
    for block in blocks:
        exec(compile(block, 'README.md', 'exec'), namespace)
    assert namespace['y'].shape == (1, 768, 256)
    assert torch.isfinite(namespace['y']).all()
    assert namespace['token_out'].shape == (1, 1, 4, 64)
    assert torch.isfinite(namespace['token_out']).all()
    assert namespace['prompt_out'].shape == (1, 65, 4, 64)
    assert torch.isfinite(namespace['prompt_out']).all()
    assert namespace['decoder'].position == 66
