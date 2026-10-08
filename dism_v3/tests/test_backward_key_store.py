"""BF16 final stores must equal RN conversion of private FP32 accumulators."""
import pytest
import cu_flash_dism as _precision_backend
import torch
import cu_flash_dism as cu

from flash_dism.backward import backward_core
from test_backward_summary import prepare, pytestmark


@pytest.mark.parametrize('n', [1,17,32,36,128,129,257,260,513])
@pytest.mark.parametrize('mode', ['soft','mixed','hard'])
@pytest.mark.parametrize('ctas', [0,1])
@pytest.mark.skipif(not _precision_backend.fp32_enabled(),
                    reason='requires optional FP32 validation instances')
def test_private_gradient_bf16_store(n, mode, ctas):
    _, state, do, _, _ = prepare(n, mode)
    fp32 = backward_core(state, do, ctas=ctas, fp32_output=True)
    bf16 = backward_core(state, do, ctas=ctas, fp32_output=False)
    for name in ('v','sk_vec','k_vec','k_lse'):
        assert bf16[name].dtype == torch.bfloat16
        torch.testing.assert_close(bf16[name], fp32[name].bfloat16(), rtol=0, atol=0)


def test_disabled_dense_diagnostics_fail_explicitly():
    if cu.backward_debug_enabled():
        pytest.skip('diagnostic build')
    _, state, _, dop, delta = prepare(17)
    common = (state['operands'], state['vertical'], dop, state['lse2'], delta)
    with pytest.raises(RuntimeError, match='DISM_BACKWARD_DEBUG=1'):
        cu.backward_summary_debug(*common,17)
    a,b = cu.backward_summary(*common,17,1,False)[-2:]
    with pytest.raises(RuntimeError, match='DISM_BACKWARD_DEBUG=1'):
        cu.backward_qk_debug(*common,cu.backward_chunk(a,b),17)
