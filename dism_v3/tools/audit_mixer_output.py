"""Bounded fused-output graph replay and SASS audit, suitable for Compute Sanitizer.

No checkpoints/datasets. Generated artifacts must remain under dism_v3/build.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import torch
from torch import nn
import triton
from fla.modules import FusedRMSNormGated
from flash_dism.kernels import mixer_output as kernels
from nanochat.models.dism.output_gate import LayerNormRawGate, RMSNormRawGate, SigmoidInputLinear


def check(tokens, width, rank, heads, dim, center):
    torch.manual_seed(613)
    down = nn.Linear(width, rank, device='cuda')
    up = SigmoidInputLinear(nn.Linear(rank, heads * dim, device='cuda'))
    original = FusedRMSNormGated(dim, eps=1e-5).cuda()
    norm = (LayerNormRawGate if center else RMSNormRawGate)(original, .53)
    x = torch.randn(1, tokens, width, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    raw = torch.randn(1, tokens, heads, dim, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    dy = torch.randn_like(raw)
    values = [x, raw, *down.parameters(), *up.parameters(), *norm.parameters()]
    def step():
        with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
            assert kernels.supported(x, raw, down.weight, down.bias, up.weight, up.bias, norm.weight)
            out = norm.fused_output(raw, x, down, up)
        out.backward(dy)
        return out
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            step()
    torch.cuda.current_stream().wait_stream(stream)
    for value in values:
        value.grad.zero_()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = step()
    for index in range(3):
        with torch.no_grad():
            x.add_(.1)
            raw.mul_(.9)
        if index == 2:
            dy.zero_()
        for value in values:
            value.grad.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.isfinite(output).all().item()
        for value in values:
            assert torch.isfinite(value.grad).all().item()
            if index == 2:
                assert torch.count_nonzero(value.grad).item() == 0
    return dict(tokens=tokens, width=width, rank=rank, heads=heads, dim=dim, center=center)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cuobjdump', default='/usr/local/cuda/bin/cuobjdump')
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1] / 'build'
    if a.output.exists() or not a.output.resolve().is_relative_to(root):
        p.error('output must be a fresh directory under dism_v3/build')
    torch.set_num_threads(1)
    cases = []
    for center in [True, False]:
        for shape in [(17, 128, 16, 4, 32), (65, 768, 96, 12, 64),
                      (129, 1024, 128, 8, 128), (19, 40, 17, 3, 33)]:
            cases.append(check(*shape, center))
    a.output.mkdir(parents=True)
    resources, tuning, seen = [], {}, set()
    for name, entry in vars(kernels).items():
        fn = entry
        if hasattr(entry, 'configs') and hasattr(entry, 'cache'):
            tuning[name] = {str(key): dict(kwargs=config.kwargs, num_warps=config.num_warps,
                                           num_stages=config.num_stages)
                            for key, config in entry.cache.items()}
        while not isinstance(fn, triton.runtime.JITFunction) and hasattr(fn, 'fn'):
            fn = fn.fn
        if not isinstance(fn, triton.runtime.JITFunction):
            continue
        for compiled in fn.device_caches[torch.cuda.current_device()][0].values():
            cubin = compiled.asm['cubin']
            key = hashlib.sha256(cubin).hexdigest()[:16]
            if key in seen:
                continue
            seen.add(key)
            path = a.output / f'{name}_{key}.cubin'
            path.write_bytes(cubin)
            sass = subprocess.check_output([a.cuobjdump, '--dump-sass', str(path)], text=True)
            path.with_suffix('.sass').write_text(sass)
            resources.append(dict(name=name, key=key, registers=compiled.n_regs,
                                  spills=compiled.n_spills, shared_bytes=compiled.metadata.shared,
                                  calls=len(re.findall(r'\bCALL(?:\.[A-Z]+)*\b', sass))))
    if not resources:
        raise RuntimeError('No compiled mixer output kernels found for SASS audit')
    report = dict(cases=cases, tuning=tuning, resources=resources)
    (a.output / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
