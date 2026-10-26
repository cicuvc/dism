"""Scratch CUDA Graph timing of the production mixer output variants.

No dataset/checkpoint access. Shapes and parameter dtypes match 123M training.
Reports isolated forward/backward and joint graph replay GPU time; not CPU time.
"""
import argparse
import gc
import json
from pathlib import Path
import statistics
import torch
from torch import nn
from torch.nn import functional as F
from fla.modules import FusedRMSNormGated
from nanochat.models.dism.output_gate import LayerNormRawGate, RMSNormRawGate, SigmoidInputLinear


class ExplicitOriginal(RMSNormRawGate):
    def forward(self, x, gate):
        value = x.float()
        normalized = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.eps)
        return (normalized * self.weight.float() * F.silu(gate.float())).to(x.dtype)


class OutputTail(nn.Module):
    def __init__(self, mode, scale):
        super().__init__()
        self.fused = mode.endswith('_fused')
        if self.fused:
            mode = mode.removesuffix('_fused')
        self.down = nn.Linear(768, 96)
        self.up = nn.Linear(96, 768)
        self.norm = FusedRMSNormGated(64, eps=1e-5)
        self.out = nn.Linear(768, 768, dtype=torch.bfloat16)
        if mode.endswith('sigmoid'):
            self.up = SigmoidInputLinear(self.up)
            kind = LayerNormRawGate if mode == 'layernorm_sigmoid' else RMSNormRawGate
            self.norm = kind(self.norm, scale)
        elif mode == 'rmsnorm_silu_explicit':
            self.norm = ExplicitOriginal(self.norm, 1.)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=.02)
                nn.init.zeros_(module.bias)

    def forward(self, stage, x, raw, gate):
        if stage == 'gate_projection':
            return self.up(self.down(x))
        if stage == 'norm_gate':
            return self.norm(raw, gate)
        if self.fused:
            return self.out(self.norm.fused_output(raw, x, self.down, self.up).flatten(2))
        g = self.up(self.down(x)).unflatten(-1, (12, 64))
        return self.out(self.norm(raw, g).flatten(2))


def capture(mode, stage, tokens, scale):
    torch.manual_seed(1741)
    module = OutputTail(mode, scale).cuda()
    x = torch.randn(1, tokens, 768, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    raw = torch.randn(1, tokens, 12, 64, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    gate = torch.randn_like(raw, requires_grad=True)
    inputs = [x, raw, gate]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    def forward():
        with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
            return module(stage, x, raw, gate)
    with torch.cuda.stream(stream):
        for _ in range(3):
            y = forward()
            dy = torch.randn_like(y)
            y.backward(dy)
        for value in list(module.parameters()) + inputs:
            if value.grad is not None:
                value.grad.zero_()
        fg, bg, joint = (torch.cuda.CUDAGraph() for _ in range(3))
        with torch.cuda.graph(fg, stream=stream):
            y = forward()
        with torch.cuda.graph(bg, stream=stream):
            y.backward(dy, retain_graph=True)
        with torch.cuda.graph(joint, stream=stream):
            z = forward()
            z.backward(dy)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    for value in list(module.parameters()) + inputs:
        if value.grad is not None and not torch.isfinite(value.grad).all().item():
            raise FloatingPointError((mode, stage))
    # Keep parameters, graph outputs and their autograd saved tensors alive.
    return {'forward': fg, 'backward': bg, 'joint': joint}, (module, inputs, y, z, dy)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tokens', type=int, default=16384)
    p.add_argument('--rounds', type=int, default=10)
    p.add_argument('--repeats', type=int, default=20)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1] / 'build'
    if args.output.exists() or not args.output.resolve().is_relative_to(root):
        p.error('output must be a fresh path under dism_v3/build')
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('high')
    modes = ['rmsnorm_silu', 'rmsnorm_silu_explicit', 'rmsnorm_sigmoid', 'layernorm_sigmoid']
    report = dict(tokens=args.tokens, heads=12, value_dim=64, width=768, rank=96,
                  scale=.53279290945938, rounds=args.rounds, repeats=args.repeats,
                  device=torch.cuda.get_device_name(), results={})
    for stage in ['gate_projection', 'norm_gate', 'full_output_tail']:
        stage_modes = modes + ['rmsnorm_sigmoid_fused', 'layernorm_sigmoid_fused'] if stage == 'full_output_tail' else modes
        graphs, keepalive, times = {}, {}, {}
        for mode in stage_modes:
            graphs[mode], keepalive[mode] = capture(mode, stage, args.tokens, report['scale'])
            times[mode] = {part: [] for part in graphs[mode]}
        for _ in range(20):
            for mode in stage_modes:
                graphs[mode]['joint'].replay()
        torch.cuda.synchronize()
        for index in range(args.rounds):
            order = stage_modes if index % 2 == 0 else list(reversed(stage_modes))
            for mode in order:
                for part, graph in graphs[mode].items():
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(args.repeats):
                        graph.replay()
                    end.record()
                    end.synchronize()
                    times[mode][part].append(start.elapsed_time(end) / args.repeats)
        result = {mode: {part: dict(mean_ms=statistics.mean(values),
                                  median_ms=statistics.median(values), samples_ms=values)
                         for part, values in parts.items()} for mode, parts in times.items()}
        report['results'][stage] = result
        print(json.dumps({stage: {mode: {part: entry['median_ms'] for part, entry in parts.items()}
                                 for mode, parts in result.items()}}), flush=True)
        del graphs, keepalive
        gc.collect()
        torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
