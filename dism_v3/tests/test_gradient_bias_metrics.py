"""Verify scale/noise distinctions used by the gradient audit."""
import importlib.util
from pathlib import Path

import pytest
import torch


def load_script(name):
    path = Path(__file__).resolve().parents[1] / 'tools' / f'{name}.py'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scale_and_orthogonal_noise_are_distinct():
    metrics = load_script('analyze_gradient_bias').metrics
    y = torch.tensor([1., 0.])
    scaled = metrics(2*y, y)
    assert scaled['cosine'] == pytest.approx(1.)
    assert scaled['norm_ratio'] == pytest.approx(2.)
    assert scaled['gain_bias'] == pytest.approx(1.)
    noisy = metrics(torch.tensor([1., 1.]), y)
    assert noisy['norm_ratio'] == pytest.approx(2**.5)
    assert noisy['cosine'] == pytest.approx(2**-.5)
    assert noisy['gain_bias'] == pytest.approx(0.)


def test_zero_reference_does_not_fabricate_cosine():
    metrics = load_script('analyze_gradient_bias').metrics
    result = metrics(torch.tensor([1.]), torch.tensor([0.]))
    assert result['cosine'] is None
    assert result['norm_ratio'] is None
    assert result['actual_norm'] == 1.


def test_summary_pools_energy_and_keeps_seeds_separate():
    metrics = load_script('analyze_gradient_bias').metrics
    summarize = load_script('summarize_gradient_bias').summarize
    rows = [dict(seed=0, **metrics(torch.tensor([2.]), torch.tensor([1.]))),
            dict(seed=1, **metrics(torch.tensor([0.]), torch.tensor([1.])))]
    result = summarize(rows)
    assert result['pooled']['gain_bias'] == pytest.approx(0.)
    assert result['pooled']['norm_ratio'] == pytest.approx(2**.5)
    assert result['norm_larger_fraction'] == .5
    assert result['by_seed'][0]['gain_bias'] == 1.
    assert result['by_seed'][1]['gain_bias'] == -1.


def test_temporal_error_vector_distinguishes_cancellation():
    metrics = load_script('analyze_gradient_bias').metrics
    drift = load_script('summarize_gradient_bias').error_drift
    y = torch.tensor([1.])
    coherent = drift([metrics(2*y, y), metrics(2*y, y)], metrics(4*y, 2*y))
    assert coherent['coherence'] == pytest.approx(2**.5)
    assert coherent['mean_error_over_rms_error'] == pytest.approx(1.)
    cancelled = drift([metrics(2*y, y), metrics(0*y, y)], metrics(2*y, 2*y))
    assert cancelled['rms_step_error_norm'] == pytest.approx(1.)
    assert cancelled['mean_error_vector_norm'] == pytest.approx(0.)
