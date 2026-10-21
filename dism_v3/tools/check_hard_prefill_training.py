"""Numerical event-backward study; no production dispatch or kernel changes."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from flash_dism.inference import HardPrefillPlan
from hard_prefill_grad_reference import SymbolicPlanner, dense_oracle, dense_exact_oracle

sys.path.append(str(Path(__file__).resolve().parents[1] / 'tests'))
from gradient_acceptance import assert_gradient


def fixture(pattern, n, r=16, dv=32, seed=318):
    rng = np.random.default_rng(seed)
    q, k = rng.integers(0, 7, size=(2, n), dtype=np.int32)
    if pattern == 'repeat':
        q.fill(0)
        k.fill(0)
    elif pattern == 'mismatch':
        q.fill(0)
        k.fill(1)
    elif pattern == 'periodic':
        q = (np.arange(n) % 3).astype(np.int32)
        k = ((np.arange(n) + 1) % 3).astype(np.int32)
    elif pattern != 'random':
        raise ValueError(pattern)
    vectors = [rng.normal(size=(n, c))*.2 for c in (r, r, dv, dv)]
    return q, k, *vectors


def metrics(actual, expected):
    a, b = np.asarray(actual, np.float64).ravel(), np.asarray(expected, np.float64).ravel()
    an, bn = np.linalg.norm(a), np.linalg.norm(b)
    return dict(max_abs=float(np.max(np.abs(a-b), initial=0.)),
                relative_l2=float(np.linalg.norm(a-b)/bn) if bn else None,
                cosine=float(a@b/(an*bn)) if an and bn else None,
                norm_ratio=float(an/bn) if bn else None,
                signed_bias_over_rms=float(np.mean(a-b)/np.sqrt(np.mean(b*b))) if bn else None)


def check_case(pattern, n, tau, r=16, dv=32, seed=318):
    q, k, sq, sk, value, dout = fixture(pattern, n, r, dv, seed)
    symbolic = SymbolicPlanner(q, k)
    program = symbolic.compile(tau)
    native = HardPrefillPlan(q, k, tau)
    arrays = native.arrays()
    for name in ('offsets', 'rows'):
        np.testing.assert_array_equal(getattr(program, name), arrays[name])
    for name in ('decay', 'weight', 'logden'):
        np.testing.assert_allclose(getattr(program, name), arrays[name], rtol=2e-11, atol=2e-11)
    out, grads, tangent = program.evaluate(sq, sk, value, dout)
    reference, rg = dense_oracle(q, k, sq, sk, value, tau, dout, direction=bool(seed % 2))
    exact, eg = dense_exact_oracle(q, k, sq, sk, value, tau, dout)
    original = native.execute(sq, sk, value, dtype=np.float64)
    np.testing.assert_allclose(out, original, rtol=2e-10, atol=2e-11)
    np.testing.assert_allclose(out, exact, rtol=2e-8, atol=2e-10)
    for a, b in zip(grads, eg):
        np.testing.assert_allclose(a, b, rtol=2e-8, atol=2e-10)
    # Keep and report failures of the ORIGINAL strict maintained-oracle gate.
    # Neither the production oracle nor its acceptance policy is changed.
    maintained_strict_failures = {}
    for name, a, b in zip(('out', 'sq', 'sk', 'v', 'tau'), (out, *grads), (reference, *rg)):
        try:
            np.testing.assert_allclose(a, b, rtol=2e-8, atol=2e-10)
        except AssertionError as error:
            maintained_strict_failures[name] = str(error)
    for name, a, b in zip(('sq_vec', 'sk_vec', 'v', 'rtau'), grads, rg):
        assert_gradient(torch.as_tensor(a), torch.as_tensor(b), name)
    np.testing.assert_allclose(grads[3], np.sum(tangent*dout), rtol=2e-10, atol=2e-11)
    # dSQ really is the existing prefill executor with swapped vector roles.
    swapped = native.execute(dout, value, sk, dtype=np.float64)
    np.testing.assert_allclose(grads[0], swapped, rtol=2e-10, atol=2e-11)
    step = 1e-5 / np.sqrt(n)
    plus, minus = [HardPrefillPlan(q, k, tau+s).execute(sq, sk, value, dtype=np.float64)
                   for s in (step, -step)]
    fd = float(np.sum((plus-minus)*dout)/(2*step))
    finite_difference_failures = {}
    for h, derivative in [(step, fd)]:
        try:
            np.testing.assert_allclose(grads[3], derivative, rtol=2e-6, atol=2e-8)
        except AssertionError as error:
            finite_difference_failures[str(h)] = str(error)
    # A larger, independent step diagnoses cancellation when true dtau is tiny.
    plus, minus = [HardPrefillPlan(q, k, tau+s).execute(sq, sk, value, dtype=np.float64)
                   for s in (1e-3, -1e-3)]
    fd_large = float(np.sum((plus-minus)*dout)/(2e-3))
    # Pure FP32 event arithmetic, NOT a claim about a BF16 GPU backward.
    out32, grads32, tangent32 = program.evaluate(sq, sk, value, dout, dtype=np.float32)
    np.testing.assert_allclose(out32, reference, rtol=2e-4, atol=2e-6)
    fp32_strict_failures, fp32_production_failures = {}, {}
    for name, a, b in zip(('sq_vec', 'sk_vec', 'v', 'rtau'), grads32, rg):
        try:
            np.testing.assert_allclose(a, b, rtol=2e-4, atol=2e-6)
        except AssertionError as error:
            fp32_strict_failures[name] = str(error)
        try:
            assert_gradient(torch.as_tensor(a), torch.as_tensor(b), name)
        except AssertionError as error:
            fp32_production_failures[name] = str(error)
    return dict(pattern=pattern, n=n, tau=tau, r=r, dv=dv, seed=seed,
                native=native.statistics(),
                maintained_oracle_strict_failures=maintained_strict_failures,
                finite_difference_strict_failures=finite_difference_failures,
                fp32_strict_failures=fp32_strict_failures,
                fp32_production_failures=fp32_production_failures,
                exact_fp64={name: metrics(a, b) for name, a, b in zip(
                    ('out', 'sq', 'sk', 'v', 'tau'), (out, *grads), (exact, *eg))},
                fp64={name: metrics(a, b) for name, a, b in zip(
                    ('out', 'sq', 'sk', 'v', 'tau'), (out, *grads), (reference, *rg))},
                fp32={name: metrics(a, b) for name, a, b in zip(
                    ('out', 'sq', 'sk', 'v', 'tau'), (out32, *grads32), (reference, *rg))},
                tau_finite_difference=fd, tau_finite_difference_large_step=fd_large,
                tau_event=float(grads[3]),
                tau_tangent=float(np.sum(tangent*dout)),
                tau_tangent_fp32=float(np.sum(tangent32*dout)))


def check_gpu_case(pattern, tau, n=256):
    """Reuse existing GPU prefill for dSQ; compare unchanged native core too."""
    from flash_dism.inference.prefill_triton import TritonPrefillPlan
    from flash_dism.forward import forward_core
    from flash_dism.backward import backward_core
    q, k, *vectors = fixture(pattern, n, r=32, dv=64)
    sq, sk, v, do = [torch.tensor(x, device='cuda').bfloat16() for x in vectors]
    cpu = [x.float().cpu().double().numpy() for x in (sq, sk, v, do)]
    plan = HardPrefillPlan(q, k, tau)
    out, grads, _ = SymbolicPlanner(q, k).compile(tau).evaluate(*cpu)
    reference, rg = dense_oracle(q, k, *cpu[:3], tau, cpu[3])
    for a, b in zip(grads, rg):
        np.testing.assert_allclose(a, b, rtol=2e-8, atol=2e-10)
    gpu = {}
    for precision in ('bf16', 'tf32x3'):
        executor = TritonPrefillPlan(plan, mma_precision=precision)
        actual = executor.execute(do, v, sk)
        assert_gradient(actual.cpu(), torch.from_numpy(grads[0]), 'sq_vec')
        gpu[precision] = metrics(actual.cpu().numpy(), grads[0])
    dummy = torch.zeros((1, n, 1, 64), device='cuda', dtype=torch.bfloat16)
    lse = torch.zeros((1, n, 1), device='cuda')
    iq, ik = [torch.tensor(x, device='cuda').reshape(1, 1, n) for x in (q, k)]
    outputs = []
    native_metrics, native_failures = {}, {}
    for direction in (False, True):
        inputs = (dummy, dummy, sq[None, :, None], sk[None, :, None], v[None, :, None],
                  lse, lse, iq, ik, torch.tensor([[direction]], device='cuda'),
                  torch.ones((1, 1), device='cuda', dtype=torch.bool),
                  torch.tensor([tau], device='cuda', dtype=torch.float32))
        y, _, state = forward_core(*inputs, save_state=True)
        g = backward_core(state, do[None, :, None].contiguous())
        outputs.append(y)
        for name in ('q_vec', 'k_vec', 'q_lse', 'k_lse'):
            assert torch.count_nonzero(g[name]) == 0
        measurements = {'out': metrics(y.float().cpu().numpy().reshape(n, -1), reference)}
        for name, expected in zip(('sq_vec', 'sk_vec', 'v', 'rtau'), rg):
            actual = g[name].float().cpu().reshape(np.shape(expected))
            measurements[name] = metrics(actual.numpy(), expected)
            try:
                assert_gradient(actual, torch.from_numpy(expected), name)
            except AssertionError as error:
                native_failures[f'{direction}:{name}'] = str(error)
        native_metrics[str(direction)] = measurements
    torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)
    return dict(pattern=pattern, n=n, tau=tau, gpu_dsq=gpu,
                native_core=native_metrics, native_core_acceptance_failures=native_failures)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokens', nargs='+', type=int, default=[1, 17, 65, 256])
    parser.add_argument('--taus', nargs='+', type=float, default=[-2., -1e-8, 0., 1e-8, .5, 2., np.log(64.), 16.])
    parser.add_argument('--seeds', nargs='+', type=int, default=[318, 319])
    parser.add_argument('--gpu-check', action='store_true')
    parser.add_argument('--readout-dim', type=int, default=16)
    parser.add_argument('--value-dim', type=int, default=32)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('choose a fresh output path')
    torch.set_num_threads(1)
    report = dict(git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  torch=torch.__version__, numpy=np.__version__,
                  method='CPU diagnostic only; unchanged native event program, maintained FP64 v3 oracle, '
                         'analytic event backward, analytic tau tangent and native-plan tau finite differences.',
                  arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  sources={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                           (Path(__file__), Path(__file__).with_name('hard_prefill_grad_reference.py'))}, cases=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        for n in args.tokens:
            for tau in args.taus:
                for pattern in ('random', 'repeat', 'mismatch', 'periodic'):
                    report['cases'].append(check_case(pattern, n, tau, args.readout_dim, args.value_dim, seed))
            failures = sum(bool(c['maintained_oracle_strict_failures']) for c in report['cases'])
            fp32_failures = sum(bool(c['fp32_strict_failures']) for c in report['cases'])
            fd_failures = sum(bool(c['finite_difference_strict_failures']) for c in report['cases'])
            print(f'seed={seed} n={n}: {len(report["cases"])} exact-oracle/production-gate passes; '
                  f'{failures} maintained-oracle, {fp32_failures} FP32, {fd_failures} FD strict discrepancies', flush=True)
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    if args.gpu_check:
        report['gpu'] = [check_gpu_case(pattern, tau) for tau in (.5, 2.)
                         for pattern in ('random', 'repeat', 'mismatch', 'periodic')]
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
