"""Aligned and partial output tiles, especially the vector dq_lse TMA path."""
import pytest
import cu_flash_dism as cu

from test_backward_summary import pytestmark
from test_backward_acceptance import (
    test_summary_component_acceptance as _summary,
    test_qk_component_acceptance as _qk,
)


@pytest.mark.parametrize('n', [4,32,36,128,260])
@pytest.mark.parametrize('mode', ['soft','mixed','hard'])
@pytest.mark.skipif(not cu.fp32_enabled(), reason='requires optional FP32 validation instances')
def test_reduction_layout_and_tail(n,mode):
    _summary(n,mode)
    _qk(n,mode)
