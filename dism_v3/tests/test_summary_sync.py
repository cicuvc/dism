"""Repeated persistent handoffs; run under racecheck for synchronization checks.

Numerical equality alone cannot establish race freedom. The first result is
checked against the mathematical oracle, then repeated launches must agree
exactly on the defined causal outputs. Run separately for each built KStages.
"""
import pytest
import torch
import cu_flash_dism

from test_summary import (
    assert_causal_summary,
    make_inputs,
    summarize_inputs,
    summary_oracle,
)


@pytest.mark.parametrize('local_wait', [False, True])
@pytest.mark.parametrize('unit_arrivals', [False, True])
@pytest.mark.parametrize('tasks', [1, 101])
def test_metadata_pipe_isolated(local_wait, unit_arrivals, tasks):
    source = torch.arange(tasks * 256, dtype=torch.int32, device='cuda').reshape(tasks, 256)
    for _ in range(4):
        actual = cu_flash_dism.metadata_pipe_probe(source, local_wait, unit_arrivals)
        torch.testing.assert_close(actual, source, rtol=0, atol=0)


@pytest.mark.parametrize('n', [513, 1281, 2305])
@pytest.mark.parametrize('ctas', [1, 2])
def test_repeated_workload_handoffs(n, ctas):
    inputs = make_inputs(n, 'mixed', 'mixed')
    first = summarize_inputs(inputs, ctas=ctas)
    assert_causal_summary(first, summary_oracle(inputs, first.shape[-2]))
    rows = torch.arange(first.shape[-3], device='cuda') * 32 + 31
    columns = torch.arange(first.shape[-2], device='cuda')
    valid = (columns[None, :] <= rows[:, None])[None, None, :, :, None]
    expected = first.masked_select(valid)
    for _ in range(8):
        actual = summarize_inputs(inputs, ctas=ctas).masked_select(valid)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
