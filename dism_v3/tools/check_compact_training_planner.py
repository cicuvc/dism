"""Compact planner equivalence and sanitizer workloads (current worktree only)."""
import argparse
import numpy as np
import torch

from flash_dism.inference.training_planner import COMPACT_WIDTHS, NativeTrainingPlanner
from flash_dism.kernels.hard_backward_fused import (
    CompactFusedHardPrefillBackward, FusedHardPrefillBackward)
from flash_dism.kernels.hard_backward_scan import HardPrefillBackward


def labels_for(pattern, b, h, n, seed):
    rng = np.random.default_rng(seed)
    if pattern in ('rand512', 'rand8'):
        return rng.integers(0, 512 if pattern == 'rand512' else 8,
                            (2, b, h, n), dtype=np.int32)
    if pattern == 'same':
        return np.zeros((2, b, h, n), dtype=np.int32)
    if pattern == 'period16':
        return np.broadcast_to(np.arange(n, dtype=np.int32) % 16,
                               (2, b, h, n)).copy()
    raise ValueError(pattern)


def docs_for(layout, b, n):
    # Deliberately unequal document lengths; sequence-level hard remains [B,H].
    boundaries = [0, max(1, n//4), n] if layout == 'ragged' and n >= 4 else [0, n]
    return np.asarray([(row, start, end) for row in range(b)
                       for start, end in zip(boundaries[:-1], boundaries[1:])],
                      dtype=np.int64)


def hard_for(kind, b, h):
    if kind == 'all':
        return np.ones((b, h), dtype=np.uint8)
    if kind == 'none':
        return np.zeros((b, h), dtype=np.uint8)
    if kind == 'alt':
        return (np.indices((b, h)).sum(0) % 2).astype(np.uint8)
    raise ValueError(kind)


def bucket_of(length):
    for i, width in enumerate(COMPACT_WIDTHS):
        if length <= width:
            return i
    raise ValueError(length)


def split_streams(offsets, lengths):
    single, multi = [], []
    for stream in range(len(offsets) - 1):
        idx = np.arange(offsets[stream], offsets[stream + 1])
        if idx.size == 1:
            single.append((stream, int(lengths[idx[0]])))
        else:
            multi.append((stream, idx))
    return single, multi


def check_cpu_case(pattern, n, tau, chunk, layout, hard_kind, b=2, h=3, per_row_tau=False):
    labels = labels_for(pattern, b, h, n, 1234 + n)
    docs = docs_for(layout, b, n)
    hard = hard_for(hard_kind, b, h)
    taus = np.full(h, tau, np.float64)
    if per_row_tau:
        taus = taus[None, :] + np.arange(b)[:, None]*.2
    planner = NativeTrainingPlanner(workers=3)
    classic = planner.prepare(labels, taus, documents=docs, hard=hard, chunk_size=chunk)
    compact = planner.prepare(labels, taus, documents=docs, hard=hard, chunk_size=chunk,
                              compact=True)
    c = {k: classic.data[k].numpy() for k in
         ('offsets', 'rows', 'prefixes', 'weights', 'dlog', 'lengths', 'resets', 'stream_heads')}
    single, multi = split_streams(c['offsets'], c['lengths'])
    expected = {w: [] for w in COMPACT_WIDTHS}
    for stream, length in single:
        expected[COMPACT_WIDTHS[bucket_of(length)]].append((stream, length))

    stats = compact.statistics
    assert list(stats['bucket_blocks']) == [len(expected[w]) for w in COMPACT_WIDTHS]
    assert stats['multi_streams'] == len(multi)
    assert stats['multi_chunks'] == sum(idx.size for _, idx in multi)
    assert stats['events'] == classic.statistics['events']

    # A compact block is exactly the classic padded chunk slice: same event
    # values and the same padding convention (rows padding = relabelled local
    # row 0, other arrays zero). Bucket width never exceeds chunk_size.
    for i, width in enumerate(COMPACT_WIDTHS):
        bucket = compact.buckets[i]
        pairs = expected[width]
        assert bucket['rows'].numel() == len(pairs)*width
        if not pairs:
            continue
        starts = np.array([c['offsets'][s] for s, _ in pairs], np.int64)*chunk
        gather = starts[:, None] + np.arange(width)[None, :]
        np.testing.assert_array_equal(bucket['rows'].numpy().reshape(-1, width), c['rows'][gather])
        np.testing.assert_array_equal(bucket['prefixes'].numpy().reshape(-1, width),
                                      c['prefixes'][gather])
        np.testing.assert_array_equal(bucket['weights'].numpy().reshape(-1, width),
                                      c['weights'][gather])
        np.testing.assert_array_equal(bucket['dlog'].numpy().reshape(-1, width), c['dlog'][gather])
        np.testing.assert_array_equal(bucket['lengths'].numpy(),
                                      np.array([L for _, L in pairs], np.int32))
        np.testing.assert_array_equal(bucket['stream_heads'].numpy(),
                                      c['stream_heads'][[s for s, _ in pairs]])

    m = compact.multi
    m_off = m['offsets'].numpy(); m_rows = m['rows'].numpy(); m_pre = m['prefixes'].numpy()
    m_w = m['weights'].numpy(); m_d = m['dlog'].numpy()
    m_len = m['lengths'].numpy(); m_res = m['resets'].numpy(); m_heads = m['stream_heads'].numpy()
    assert m_off.size == len(multi) + 1 and int(m_off[-1]) == stats['multi_chunks']
    expected_off = np.r_[0, np.cumsum([idx.size for _, idx in multi])].astype(np.int64)
    np.testing.assert_array_equal(m_off, expected_off)
    if multi:
        chunks = np.concatenate([idx for _, idx in multi])
        gather = chunks[:, None]*chunk + np.arange(chunk)[None, :]
        np.testing.assert_array_equal(m_rows.reshape(-1, chunk), c['rows'][gather])
        np.testing.assert_array_equal(m_pre.reshape(-1, chunk), c['prefixes'][gather])
        np.testing.assert_array_equal(m_w.reshape(-1, chunk), c['weights'][gather])
        np.testing.assert_array_equal(m_d.reshape(-1, chunk), c['dlog'][gather])
        np.testing.assert_array_equal(m_len, c['lengths'][chunks])
        np.testing.assert_array_equal(m_res, c['resets'][chunks])
        np.testing.assert_array_equal(m_heads, c['stream_heads'][[s for s, _ in multi]])


def check_gpu_case(pattern, n, chunk, layout, hard_kind, b=4, h=2, r=32, dv=64, seed=0, precision='bf16', per_row_tau=False):
    labels = labels_for(pattern, b, h, n, 99 + n)
    docs = docs_for(layout, b, n)
    hard = hard_for(hard_kind, b, h)
    tau = np.full(h, 0.5)
    if per_row_tau:
        tau = tau[None, :] + np.arange(b)[:, None]*.2
    planner = NativeTrainingPlanner(8)
    classic = planner.prepare(labels, tau, documents=docs, hard=hard, chunk_size=chunk)
    compact = planner.prepare(labels, tau, documents=docs, hard=hard, chunk_size=chunk,
                              compact=True)
    base = HardPrefillBackward.from_native(classic.upload(), mma_precision=precision)
    packed = FusedHardPrefillBackward.from_plan(
        base, classic.data['offsets'].numpy(), host_lengths=classic.data['lengths'].numpy(),
        long_mode='three_stage')
    bucketed = CompactFusedHardPrefillBackward.from_compact(
        compact.upload(), mma_precision=precision, long_mode='three_stage')
    total = b*h*n
    gen = torch.Generator(device='cuda').manual_seed(seed)
    scale = 0.2
    sq = (torch.randn(total, r, device='cuda', generator=gen)*scale).bfloat16()
    sk = (torch.randn(total, r, device='cuda', generator=gen)*scale).bfloat16()
    val = (torch.randn(total, dv, device='cuda', generator=gen)*scale).bfloat16()
    dout = (torch.randn(total, dv, device='cuda', generator=gen)*scale).bfloat16()
    out_packed = packed.backward(sq, sk, val, dout)
    out_compact = bucketed.backward(sq, sk, val, dout)
    torch.cuda.synchronize()
    worst = 0.0
    for key in ('sq_vec', 'sk_vec', 'v', 'rtau'):
        a = out_packed[key].float()
        c = out_compact[key].float()
        diff = (a - c).abs()
        denom = a.abs().clamp_min(1e-6)
        rel = (diff/denom).max().item()
        worst = max(worst, rel)
        print(f'    {key:8s} max_abs={diff.max().item():.3e} max_scaled={rel:.3e} '
              f'ref_norm={a.norm().item():.3e}', flush=True)
        # Do not treat a printed mismatch followed by exit(0) as a pass.
        # Vector differences are FP32 atomic ordering; tau also reduces across
        # bucket order. Preserve BF16 MMA in both arms.
        torch.testing.assert_close(c, a, rtol=1e-5,
                                   atol=1e-4 if key == 'rtau' else 1e-7)
    return worst


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    if not (args.cpu or args.gpu or args.smoke):
        parser.error('select --cpu, --gpu or --smoke')
    if args.cpu:
        count = 0
        for pattern in ('rand512', 'rand8', 'same', 'period16'):
            for n in (1, 2, 17, 65, 256, 2048):
                for tau in (0., .5, 2., 16.):
                    for chunk in (16, 32, 64):
                        for layout in ('uniform', 'ragged'):
                            for hard in ('all', 'alt'):
                                check_cpu_case(pattern, n, tau, chunk, layout, hard)
                                count += 1
        print(f'{count} CPU cases passed', flush=True)
    if args.gpu:
        count = 0
        for pattern in ('rand512', 'rand8', 'same'):
            for n in (256, 2048):
                for chunk in (16, 64):
                    for layout in ('uniform', 'ragged'):
                        for hard in ('all', 'alt'):
                            check_gpu_case(pattern, n, chunk, layout, hard)
                            count += 1
        print(f'{count} GPU cases passed', flush=True)
    if args.smoke:
        check_gpu_case('rand512', 256, 16, 'uniform', 'all', b=2, h=2, seed=1)
        check_gpu_case('rand8', 512, 64, 'ragged', 'alt', b=2, h=2, seed=2)
        print('sanitizer workload passed', flush=True)


if __name__ == '__main__':
    main()
