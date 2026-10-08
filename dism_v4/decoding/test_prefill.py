"""Standalone CPU checks: python dism_v4/decoding/test_prefill.py."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

from prefill import HardPrefillPlan, load_prefill

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from prefill_reference import PrefillPlan
from test_prefill_reference import dense_exact, torch_oracle


def test():
    rng = np.random.default_rng(8137)
    load_prefill()
    errors = dict(fp64=0., fp32=0., torch=0.)
    cases = 0
    for seed in range(100):
        n = 1 + seed * 2
        keys, queries = rng.integers(0, 2 + seed % 11, (2, n))
        if seed % 5 == 0:
            keys[:] = queries[:] = 0
        elif seed % 5 == 1:
            queries = keys.copy()
        elif seed % 5 == 2:
            keys = np.arange(n) % 7
            queries = (np.arange(n) + 3) % 7
        elif seed % 5 == 3:
            queries += 100
        reset = rng.random(n) < .1
        if seed % 3 == 0:
            reset[:] = False
        elif seed % 3 == 1:
            reset[:] = True
        tau = (0., 1e-12, .7, np.log(64.), -.7)[(seed // 5) % 5]
        r, dv = ((3, 5), (16, 32), (32, 64))[seed % 3]
        sq, sk = rng.normal(size=(2, n, r))
        value = rng.normal(size=(n, dv))
        plan = HardPrefillPlan(queries, keys, tau, reset=reset)
        out = plan.execute(sq, sk, value, dtype=np.float64)
        fp32 = plan.execute(sq, sk, value)
        expected, den = dense_exact(queries, keys, reset, tau, sq, sk, value)
        pyout, pyden = PrefillPlan(queries, keys, reset, tau).evaluate(sq, sk, value)
        oracle = torch_oracle(queries, keys, reset, tau, sq, sk, value)
        np.testing.assert_allclose(out, expected, rtol=3e-11, atol=3e-11)
        np.testing.assert_allclose(out, pyout, rtol=3e-11, atol=3e-11)
        np.testing.assert_allclose(plan.arrays()["logden"], den, rtol=3e-12, atol=3e-11)
        np.testing.assert_allclose(out, oracle, rtol=2e-8, atol=2e-8)
        np.testing.assert_allclose(fp32, expected, rtol=3e-5, atol=3e-5)
        for name, result in (("fp64", out), ("fp32", fp32), ("torch", oracle)):
            errors[name] = max(errors[name], float(np.max(np.abs(result-expected))))
        # Basis values expose every normalized causal pair; no signed cancellation.
        if n < 65:
            ones = np.ones((n, 1))
            basis = plan.execute(ones, ones, np.eye(n), dtype=np.float64)
            correct, _ = dense_exact(queries, keys, reset, tau, ones, ones, np.eye(n))
            np.testing.assert_allclose(basis, correct, rtol=3e-11, atol=3e-11)
        cut = max(1, n//2)
        prefix = HardPrefillPlan(queries[:cut], keys[:cut], tau, reset=reset[:cut])
        np.testing.assert_allclose(out[:cut], prefix.execute(sq[:cut], sk[:cut], value[:cut], dtype=np.float64), rtol=3e-11, atol=3e-11)
        arrays = plan.arrays()
        arrays["weight"][:] = 0  # Export ownership must not mutate the native plan.
        np.testing.assert_array_equal(out, plan.execute(sq, sk, value, dtype=np.float64))
        cases += 1

    timings = []
    for pattern in ("repeat", "random", "distinct"):
        for n in (256, 1024, 4096, 16384):
            keys, queries = rng.integers(0, 16, (2, n))
            if pattern == "repeat": keys[:] = queries[:] = 0
            if pattern == "distinct": keys = queries = np.arange(n)
            reset = np.zeros(n, dtype=bool)
            start = time.perf_counter()
            plan = HardPrefillPlan(queries, keys, np.log(64.), reset=reset)
            elapsed = time.perf_counter()-start
            # Realistic vector dimensions at 4K; avoid a long CPU mock at 16K.
            if n <= 4096:
                sq, sk = rng.normal(size=(2, n, 16))
                value = rng.normal(size=(n, 32))
                start = time.perf_counter()
                out = plan.execute(sq, sk, value, dtype=np.float64)
                execute_ms = (time.perf_counter()-start)*1000
                # Bound the dense test cost while also checking the 4K long chain.
                if pattern == "repeat" or n <= 1024:
                    expected, _ = dense_exact(queries, keys, reset, np.log(64.), sq, sk, value)
                    np.testing.assert_allclose(out, expected, rtol=2e-9, atol=2e-9)
            else:
                execute_ms = None
            timings.append(dict(pattern=pattern, **plan.statistics(), plan_ms=elapsed*1000, mock_fp64_ms=execute_ms))

    invalid = [([], [], .7, None), ([1], [1, 2], .7, None),
               ([1.5], [1], .7, None), ([2**40], [1], .7, None),
               ([1], [1], float("nan"), None), ([1], [1], .7, [0])]
    for q, k, tau, reset in invalid:
        try:
            HardPrefillPlan(q, k, tau, reset=reset)
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError("invalid input accepted")
    return dict(status="PASS", cases=cases, max_abs=errors, timings=timings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    report = test()
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
