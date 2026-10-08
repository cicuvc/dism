"""Standalone CPU numerical checks; does not build/import the CUDA extension.

python dism_v4/tools/test_prefill_reference.py --output /tmp/prefill_results.json
"""
import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import torch

from prefill_reference import PrefillPlan


def dense_exact(queries, keys, resets, tau, sq, sk, value):
    n = len(keys)
    previous = np.full(n, -np.inf)
    result = []
    logden = []
    for i in range(n):
        shifted = np.concatenate(([-np.inf], previous[:-1]))
        if resets[i]:
            shifted[:] = -np.inf
        previous = np.where(queries[i] == keys, tau + np.logaddexp(0., shifted), -np.inf)
        scores = previous.copy()
        scores[i+1:] = -np.inf
        denominator = np.logaddexp.reduce(np.concatenate(([0.], scores)))
        result.append((np.exp(scores - denominator) * (sk @ sq[i])) @ value)
        logden.append(denominator)
    return np.array(result), np.array(logden)


def torch_oracle(queries, keys, resets, tau, sq, sk, value):
    path = Path(__file__).resolve().parents[1] / "python/flash_dism/reference/dism_v4_ref.py"
    spec = importlib.util.spec_from_file_location("prefill_torch_oracle", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    n = len(keys)
    z = torch.zeros(1, n, 1, 1, dtype=torch.float64)
    lse = z[..., 0]
    bnhr = lambda x: torch.from_numpy(x)[None, :, None, :]
    delta = torch.where(torch.from_numpy(resets)[None, None, :], torch.inf, 0.).double()
    return module.dism_ref(
        z, z, bnhr(sq), bnhr(sk), lse, lse,
        torch.from_numpy(queries)[None, None, :], torch.from_numpy(keys)[None, None, :],
        torch.ones(1, 1, dtype=torch.bool), torch.ones(1, 1, n, dtype=torch.bool),
        delta, bnhr(value), torch.tensor([tau], dtype=torch.float64),
    )[0, :, 0].numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    rng = np.random.default_rng(4821)
    records = []
    started = time.perf_counter()
    for seed in range(60):
        n = 1 + seed * 2
        alphabet = (2, 7, 40)[seed % 3]
        keys = rng.integers(0, alphabet, n)
        queries = rng.integers(0, alphabet, n)
        pattern = seed % 6
        if pattern == 0:
            keys[:] = queries[:] = 0
        elif pattern == 1:
            queries = keys.copy()
        elif pattern == 2:
            keys = np.arange(n) % 5
            queries = (np.arange(n) + 2) % 5
        elif pattern == 3:
            queries += alphabet  # Total mismatch, fallback only.
        resets = rng.random(n) < .12
        if seed % 5 == 0:
            resets[:] = False
        elif seed % 5 == 1:
            resets[:] = True
        tau = (0., 1e-12, .7, np.log(64.), -.7)[seed % 5]
        r, dv = ((3, 5), (16, 8), (32, 64))[seed % 3]
        sq, sk = rng.normal(size=(2, n, r))
        value = rng.normal(size=(n, dv))
        plan = PrefillPlan(queries, keys, resets, tau)
        plan.audit(queries, keys, resets)
        actual, logden = plan.evaluate(sq, sk, value)
        expected, exact_logden = dense_exact(queries, keys, resets, tau, sq, sk, value)
        oracle = torch_oracle(queries, keys, resets, tau, sq, sk, value)
        np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-11)
        np.testing.assert_allclose(logden, exact_logden, rtol=2e-12, atol=2e-11)
        # Existing torch softplus switches to x at x>20; exact logaddexp does not.
        np.testing.assert_allclose(actual, oracle, rtol=2e-8, atol=2e-8)
        fp32, _ = plan.evaluate(sq, sk, value, dtype=np.float32)
        np.testing.assert_allclose(fp32, expected, rtol=2e-5, atol=2e-5)
        cut = max(1, n // 2)
        prefix = PrefillPlan(queries[:cut], keys[:cut], resets[:cut], tau)
        prefix_out, prefix_den = prefix.evaluate(sq[:cut], sk[:cut], value[:cut])
        np.testing.assert_allclose(actual[:cut], prefix_out, rtol=2e-11, atol=2e-11)
        np.testing.assert_allclose(logden[:cut], prefix_den, rtol=2e-12, atol=2e-11)
        records.append(dict(seed=seed, **plan.statistics(),
                            exact_max_abs=float(np.max(np.abs(actual - expected))),
                            oracle_max_abs=float(np.max(np.abs(actual - oracle))),
                            fp32_max_abs=float(np.max(np.abs(fp32 - expected)))))

    # Long repeat makes exp(tau*L) overflow even FP64; no dense pair audit here.
    growth = []
    for n in (256, 512, 1024, 2048):
        keys = queries = np.zeros(n, dtype=np.int64)
        resets = np.zeros(n, dtype=bool)
        sq, sk = rng.normal(size=(2, n, 3))
        value = rng.normal(size=(n, 5))
        start = time.perf_counter()
        plan = PrefillPlan(queries, keys, resets, np.log(64.))
        planning_seconds = time.perf_counter() - start
        actual, logden = plan.evaluate(sq, sk, value)
        expected, exact_logden = dense_exact(queries, keys, resets, np.log(64.), sq, sk, value)
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(logden, exact_logden, rtol=1e-12, atol=1e-9)
        assert np.isfinite(actual).all()
        growth.append(dict(**plan.statistics(), planning_seconds=planning_seconds,
                           exact_max_abs=float(np.max(np.abs(actual - expected))),
                           max_logden=float(logden.max())))
    report = dict(status="PASS", cases=len(records), seconds=time.perf_counter()-started,
                  max_exact_abs=max(x["exact_max_abs"] for x in records),
                  max_oracle_abs=max(x["oracle_max_abs"] for x in records),
                  max_fp32_abs=max(x["fp32_max_abs"] for x in records),
                  long_chain=growth, cases_detail=records)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "cases_detail"}, indent=2))


if __name__ == "__main__":
    main()
