"""Exact zero-history single-K counterfactual via two linear symbol scans.

CPU diagnostic prototype. All other K labels stay committed; this does not
compute an expectation over jointly random K labels or native suffix history.
"""
import argparse
import time
from pathlib import Path

import torch


def prefix_control(iq, ik, sq, sk, v, gs, tau, positions, ids, vocab_size):
    n, rank = sq.shape; dv = v.shape[-1]; modes = len(gs)
    state = torch.zeros((vocab_size, rank, dv), dtype=sq.dtype)
    counts = torch.zeros(vocab_size, dtype=sq.dtype)
    outputs = torch.empty_like(v); denominators = torch.empty(n, dtype=sq.dtype)
    reads = torch.empty(n, dtype=torch.long)
    fallback = (-tau).exp()
    for i in range(n):
        key, query = int(ik[i]), int(iq[i])
        state[key] += sk[i, :, None] * v[i, None, :]
        counts[key] += 1
        reads[i] = counts[query].long()
        denominators[i] = fallback + counts[query]
        outputs[i] = (sq[i] @ state[query]) / denominators[i]
    del state
    # sign axis 0 = addition, 1 = removal; removal stores a positive contraction.
    matrices = torch.zeros((2, modes, vocab_size, rank, dv), dtype=sq.dtype)
    scalars = torch.zeros((2, modes, vocab_size), dtype=sq.dtype)
    rewards = torch.zeros((modes, len(positions), ids.shape[-1]), dtype=sq.dtype)
    selected = {int(j): a for a, j in enumerate(positions)}
    for i in range(n - 1, -1, -1):
        c = int(iq[i]); outer = sq[i, None, :, None] * gs[:, i, None, :]
        value = (gs[:, i] * outputs[i]).sum(-1)
        matrices[0, :, c] += outer / (denominators[i] + 1)
        scalars[0, :, c] += value / (denominators[i] + 1)
        if reads[i] > 0:
            matrices[1, :, c] += outer / (denominators[i] - 1)
            scalars[1, :, c] += value / (denominators[i] - 1)
        if i in selected:
            a = selected[i]; old = int(ik[i]); candidates = ids[a]
            add = torch.einsum('mcav,a,v->mc', matrices[0, :, candidates], sk[i], v[i]) - scalars[0, :, candidates]
            remove = torch.einsum('mav,a,v->m', matrices[1, :, old], sk[i], v[i]) - scalars[1, :, old]
            rewards[:, a] = torch.where(candidates[None] == old, 0., add - remove[:, None])
    return outputs, rewards


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact', type=Path)
    p.add_argument('--threads', type=int, default=2)
    a = p.parse_args(); torch.set_num_threads(a.threads); torch.set_num_interop_threads(1)
    result = torch.load(a.artifact, map_location='cpu', weights_only=False)
    report = result['report']; data = torch.load(report['input'], map_location='cpu', weights_only=False)
    h = report['head']; vals = data['inputs'][0]
    q, k, sq, sk, v = [x[0, :, h].double().contiguous() for x in vals[:5]]
    eq, ek = [x[h].bfloat16().double() for x in vals[5:7]]
    iq, ik = (q@eq.T).argmax(-1), (k@ek.T).argmax(-1)
    gs = torch.stack([data['g_'+m][0, :, h].double()*len(q) for m in ('hard', 'soft')])
    start = time.perf_counter()
    _, actual = prefix_control(iq, ik, sq, sk, v, gs, vals[7][h].double(), result['positions'], result['ids'], len(eq))
    expected = result['rewards']['zero']
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-9)
    print('PASS', a.artifact.stem, 'max_abs_error', float((actual-expected).abs().max()),
          'relative_l2', float((actual-expected).norm()/expected.norm().clamp_min(1e-300)),
          'seconds', round(time.perf_counter()-start, 3))


if __name__ == '__main__':main()
