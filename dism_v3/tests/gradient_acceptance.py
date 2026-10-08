"""Explicit initial-release tolerances; original strict tests never use these."""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GradientTolerance:
    cosine: float
    relative_l2: float
    absolute_l2: float


VECTOR = GradientTolerance(cosine=.999, relative_l2=.03, absolute_l2=1e-4)
LSE = GradientTolerance(cosine=.995, relative_l2=.10, absolute_l2=.01)
TAU = GradientTolerance(cosine=.95, relative_l2=.50, absolute_l2=.02)


def assert_gradient(actual, reference, name, tolerance=None):
    """Check direction, error and norm bias; do not fabricate near-zero cosine.

    The absolute allowance handles small reference norms; cosine is enforced
    above that noise floor. Exact-zero structural gradients keep their own
    tight gate, including in hard mode.
    """
    assert actual.shape == reference.shape, name
    x, y = actual.double().flatten(), reference.double().flatten()
    assert torch.isfinite(x).all() and torch.isfinite(y).all(), name
    yn, xn = y.norm().item(), x.norm().item()
    if yn < 1e-10:
        assert xn < 1e-6, (name, 'zero-reference gradient', xn)
        return
    tolerance = tolerance or (TAU if name in ('tau', 'rtau') else
                              LSE if name in ('q_lse', 'k_lse') else VECTOR)
    error = (x-y).norm().item()
    bound = tolerance.absolute_l2 + tolerance.relative_l2*yn
    assert error <= bound, (name, 'L2 error', error, 'bound', bound)
    assert abs(xn-yn) <= bound, (name, 'norm ratio', xn/yn)
    if yn > tolerance.absolute_l2:
        cosine = torch.nn.functional.cosine_similarity(x, y, dim=0).item()
        assert cosine >= tolerance.cosine, (name, 'cosine', cosine, tolerance.cosine)
