"""The relaxed gate must still reject broken scales/directions/zero paths."""
import pytest
import torch

from gradient_acceptance import assert_gradient


@pytest.mark.parametrize('name', ['q_vec','q_lse','rtau'])
@pytest.mark.parametrize('factor', [-1., 0., 2.])
def test_rejects_broken_gradients(name,factor):
    reference = torch.tensor([1.,2.,3.])
    with pytest.raises(AssertionError):
        assert_gradient(reference*factor,reference,name)


def test_keeps_zero_gradient_gate():
    with pytest.raises(AssertionError):
        assert_gradient(torch.tensor([.001]),torch.zeros(1),'rtau')


def test_allows_small_tau_cancellation_error():
    assert_gradient(torch.tensor([-.0052]),torch.tensor([.0071]),'rtau')


def test_shape_and_nonfinite_fail():
    with pytest.raises(AssertionError):
        assert_gradient(torch.ones(2),torch.ones(1),'q_vec')
    with pytest.raises(AssertionError):
        assert_gradient(torch.tensor([float('nan')]),torch.ones(1),'q_vec')
