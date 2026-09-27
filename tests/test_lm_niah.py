from collections import Counter
import pytest
from dism_v2.eval_lm_niah import build_prompt, case_plan


def test_balanced_cases():
    plan = case_plan(256, 9321)
    assert len(plan) == len({p[0] for p in plan}) == 128
    assert plan == case_plan(256, 9321)
    assert all(a != b for _, a, b, _ in plan)
    counts = Counter((depths[0], target) for _, target, _, depths in plan)
    assert len(counts) == 32 and set(counts.values()) == {4}
    with pytest.raises(ValueError):
        case_plan(255, 9321)


@pytest.mark.parametrize('length', [2048, 8192])
@pytest.mark.parametrize('depth', [.1, .35, .65, .9])
def test_prompt_layout(length, depth):
    ids, at = build_prompt(list(range(8192)), length, [-1, -2], [-3], depth)
    assert len(ids) == length
    assert ids[at:at+2] == [-1, -2]
    assert ids[-1] == -3
