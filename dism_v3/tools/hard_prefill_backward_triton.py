"""Symbolic single-document fixture for the packaged backward kernels."""
import math
import numpy as np
import torch
from hard_prefill_grad_reference import SymbolicPlanner
from flash_dism.kernels.hard_backward_scan import (
    HardPrefillBackward as _NativeBackward, _query_backward, _key_backward)


def pack_program(program, chunk_size):
    """Native chunk convention, retaining analytic dlog(weight)/dtau per event."""
    if chunk_size not in (16, 32, 64):
        raise ValueError('chunk_size must be 16, 32 or 64')
    fields = {name: [] for name in ('rows', 'prefixes', 'weights', 'dlog', 'lengths', 'resets')}
    offsets = [0]
    for begin, end in zip(program.offsets[:-1], program.offsets[1:]):
        start = int(begin)
        while start < end:
            stop = min(start+chunk_size, int(end))
            for e in range(start+1, stop):
                if program.rows[e] >= 0 and program.decay[e] == 0:
                    stop = e
                    break
            reset = program.rows[start] >= 0 and program.decay[start] == 0
            prefix = 0.
            rows, prefixes, weights, derivs = [], [], [], []
            for e in range(start, stop):
                if program.rows[e] >= 0 and not (reset and e == start):
                    prefix += math.log(program.decay[e])
                rows.append(program.rows[e])
                prefixes.append(prefix)
                weights.append(program.weight[e])
                derivs.append(program.dlogweight[e])
            for name, values in zip(('rows', 'prefixes', 'weights', 'dlog'), (rows, prefixes, weights, derivs)):
                fields[name].extend(values + [0]*(chunk_size-len(values)))
            fields['lengths'].append(stop-start)
            fields['resets'].append(int(reset))
            start = stop
        offsets.append(len(fields['lengths']))
    fields['offsets'] = offsets
    return {name: np.asarray(value, dtype=np.int64 if name == 'offsets' else
                             np.int32 if name in ('rows', 'lengths', 'resets') else np.float32)
            for name, value in fields.items()}


class HardPrefillBackward(_NativeBackward):
    """Single document/head, prepared outside capture; no autograd integration.

    BF16/FP32 contiguous input vectors; FP32 output gradients. Tau is shared
    across this plan. Multiple documents/heads must be planned independently.
    """
    def __init__(self, queries, keys, tau, *, device='cuda', chunk_size=16, mma_precision='tf32x3'):
        if mma_precision not in ('bf16', 'tf32x3'):
            raise ValueError('mma_precision must be bf16 or tf32x3')
        self.device = torch.device(device)
        if self.device.type != 'cuda':
            raise ValueError('CUDA device required')
        if self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        self.program = SymbolicPlanner(queries, keys).compile(float(tau))
        self.n, self.c, self.precision = len(queries), chunk_size, mma_precision
        self.packed = pack_program(self.program, chunk_size)
        self.data = {k: torch.as_tensor(v, device=self.device).contiguous() for k, v in self.packed.items()}
        self.streams = len(self.packed['offsets'])-1
        self.last_kernels = []
