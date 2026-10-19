"""Numerical/resource diagnostics for the opt-in Triton hard backward prototype."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np
import torch

from check_hard_prefill_training import assert_gradient, dense_oracle, fixture, metrics
from hard_prefill_backward_triton import HardPrefillBackward
from flash_dism.inference import HardPrefillPlan


def inspect_kernel(kernel, directory):
    directory.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(kernel.asm['cubin']).hexdigest()
    target = directory / f'{kernel.name}_{digest[:16]}.cubin'
    if not target.exists():
        target.write_bytes(kernel.asm['cubin'])
        sass = subprocess.check_output(['/usr/local/cuda/bin/cuobjdump', '-sass', str(target)], text=True)
        target.with_suffix('.sass').write_text(sass)
    else:
        sass = target.with_suffix('.sass').read_text()
    return dict(name=kernel.name, sha256=digest, registers=kernel.n_regs,
                spills=kernel.n_spills, shared_bytes=kernel.metadata.shared,
                calls=len(re.findall(r'\bCALL(?:\.|\s)', sass)),
                local_loads=len(re.findall(r'\bLDL(?:\.|\s)', sass)),
                local_stores=len(re.findall(r'\bSTL(?:\.|\s)', sass)),
                # Static instruction sites, not dynamic transactions. In
                # particular, branches/loops and predicates change execution
                # counts; shared_bytes above is capacity, not traffic.
                mma_instructions=len(re.findall(r'\bHMMA(?:\.|\s)', sass)),
                global_atomics=len(re.findall(r'\b(?:ATOMG|REDG)(?:\.|\s)', sass)),
                shared_loads=len(re.findall(r'\b(?:LDS|LDSM)(?:\.|\s)', sass)),
                shared_stores=len(re.findall(r'\bSTS(?:\.|\s)', sass)),
                global_loads=len(re.findall(r'\bLDG(?:\.|\s)', sass)),
                global_stores=len(re.findall(r'\bSTG(?:\.|\s)', sass)),
                shared_atomics=len(re.findall(r'\b(?:ATOMS|REDS)(?:\.|\s)', sass)))


def check_case(pattern, n, tau, *, r=32, dv=64, seed=318, precision='tf32x3',
               chunk_size=16, dtype=torch.bfloat16, resource_dir=None, fused=False):
    q, k, *vectors = fixture(pattern, n, r=r, dv=dv, seed=seed)
    inputs = [torch.tensor(x, device='cuda', dtype=dtype) for x in vectors]
    cpu = [x.float().cpu().double().numpy() for x in inputs]
    plan = HardPrefillBackward(q, k, tau, chunk_size=chunk_size, mma_precision=precision)
    if fused:
        from hard_prefill_backward_fused import FusedHardPrefillBackward
        plan = FusedHardPrefillBackward.from_plan(plan, plan.packed['offsets'])
    native = HardPrefillPlan(q, k, tau).native.chunks(chunk_size)
    for name in ('offsets', 'rows', 'lengths', 'resets'):
        np.testing.assert_array_equal(plan.packed[name], native[name])
    for name in ('prefixes', 'weights'):
        np.testing.assert_allclose(plan.packed[name], native[name], rtol=2e-6, atol=2e-7)
    _, reference = dense_oracle(q, k, *cpu[:3], tau, cpu[3], direction=bool(seed % 2))
    actual = plan.backward(*inputs)
    measurements, production_failures = {}, {}
    for name, expected in zip(('sq_vec', 'sk_vec', 'v', 'rtau'), reference):
        a = actual[name].cpu()
        assert torch.isfinite(a).all(), name
        measurements[name] = metrics(a.numpy(), expected)
        try:
            assert_gradient(a, torch.as_tensor(expected), name)
        except AssertionError as error:
            production_failures[name] = str(error)
        if name != 'rtau':
            assert_gradient(a, torch.as_tensor(expected), name)
    a, b = float(actual['rtau'].cpu()), float(reference[3])
    partials = actual['tau_partials'].cpu().double().numpy()
    # A diagnostic category, NOT a replacement for the unchanged production gate.
    # Record raw sign flips even below this absolute near-zero threshold.
    sign_floor = 1e-6
    flip = a*b < 0
    result = dict(pattern=pattern, n=n, tau=tau, r=r, dv=dv, seed=seed,
                  precision=precision, chunk_size=chunk_size, dtype=str(dtype),
                  fused_chunk_backward=fused,
                  streams=plan.streams, chunks=len(plan.packed['lengths']),
                  device_metadata_bytes=sum(x.numel()*x.element_size() for x in plan.data.values()),
                  matrix_snapshot_bytes=getattr(plan, 'matrix_snapshot_bytes', 0),
                  tau_partial_bytes=actual['tau_partials'].numel()*4,
                  gradients=measurements, production_acceptance_failures=production_failures,
                  tau_actual=a, tau_reference=b, tau_sign_flip=flip,
                  tau_sign_floor=sign_floor, tau_material_sign_flip=flip and abs(b) > sign_floor,
                  tau_fp64_partial_sum=float(partials.sum()),
                  tau_absolute_partial_sum=float(np.abs(partials).sum()))
    if resource_dir is not None:
        result['kernels'] = [inspect_kernel(kernel, resource_dir) for kernel in plan.last_kernels]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokens', nargs='+', type=int, default=[1, 17, 65, 256])
    parser.add_argument('--taus', nargs='+', type=float, default=[-2., 0., .5, 2., np.log(64.), 16.])
    parser.add_argument('--patterns', nargs='+', default=['random', 'repeat', 'mismatch', 'periodic'])
    parser.add_argument('--precisions', nargs='+', default=['tf32x3', 'bf16'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[318, 319])
    parser.add_argument('--readout-dim', type=int, default=32)
    parser.add_argument('--value-dim', type=int, default=64)
    parser.add_argument('--chunk-size', type=int, default=16)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('choose a fresh output path')
    torch.set_num_threads(1)
    report = dict(git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  device=torch.cuda.get_device_name(), torch=torch.__version__,
                  arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  sources={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                           (Path(__file__), Path(__file__).with_name('hard_prefill_backward_triton.py'),
                            Path(__file__).with_name('hard_prefill_grad_reference.py'))}, cases=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        for n in args.tokens:
            for tau in args.taus:
                for pattern in args.patterns:
                    for precision in args.precisions:
                        case = check_case(pattern, n, tau, r=args.readout_dim, dv=args.value_dim,
                                          seed=seed, precision=precision, chunk_size=args.chunk_size,
                                          resource_dir=args.output.with_suffix(''))
                        report['cases'].append(case)
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
            for precision in args.precisions:
                cases = [c for c in report['cases'] if c['precision'] == precision]
                print(f'seed={seed} n={n} {precision}: {len(cases)} vector passes; '
                      f'{sum(c["tau_sign_flip"] for c in cases)} raw / '
                      f'{sum(c["tau_material_sign_flip"] for c in cases)} material tau sign flips; '
                      f'{sum(bool(c["production_acceptance_failures"]) for c in cases)} original-gate failures', flush=True)


if __name__ == '__main__':
    main()
