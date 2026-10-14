"""Opt-in chunk-parallel hard backward inspired by FLA's chunk_bwd_dqkwg.

The installed FLA simple GLA/RetNet shares dS=dO@V.T for dQ and dK.
Here a single event chunk has no incoming/outgoing matrix state; compute
all local vector gradients together and contract analytic tau pairwise.
Multi-chunk streams use parallel summaries, a matrix-elementwise state scan,
and parallel fused readout. Single-chunk streams only need the readout.
The previous register scans remain an explicit comparison option.
No FLA source is vendored. Used by the explicit eager async training adapter;
the default model dispatcher remains unchanged.
"""
import numpy as np
import torch
import triton as tr
import triton.language as tl

from .hard_backward_scan import HardPrefillBackward, _query_backward, _key_backward


@tr.jit
def _chunk_backward(SQ, SK, V, DO, DSQ, DSK, DV_OUT, TAU, IDS,
                           OFFSETS, ROWS, PREFIX, WEIGHT, DLOG, LENGTH,
                           R: tl.constexpr, DV: tl.constexpr, C: tl.constexpr,
                           BR: tl.constexpr, BD: tl.constexpr, BF16: tl.constexpr,
                           H=None, G=None, CHUNKS=None, RESET=None, BC: tl.constexpr=0,
                           LOCAL: tl.constexpr=0):
    # LOCAL marks a compact bucket arena: the program id is the block index
    # inside that arena and the arena itself is a padded plan of width C, so
    # chunk*C addressing and the per-block tau slot both stay valid unchanged.
    local = tl.program_id(0)
    if LOCAL:
        chunk = local
        stream = local
    else:
        stream = tl.load(IDS + local)
        if CHUNKS is None:
            chunk = tl.load(OFFSETS + stream)
        else:
            chunk = tl.load(CHUNKS + local).to(tl.int64)
    K: tl.constexpr = C if BC == 0 else BC
    t, r, d = tl.arange(0, K), tl.arange(0, BR), tl.arange(0, BD)
    length = tl.load(LENGTH + chunk)
    row = tl.load(ROWS + chunk*C + t)
    p = tl.load(PREFIX + chunk*C + t)
    w = tl.load(WEIGHT + chunk*C + t)
    dw = tl.load(DLOG + chunk*C + t)
    key, query = (t < length) & (row >= 0), (t < length) & (row < 0)
    ki, qi = tl.where(key, row, 0), tl.where(query, -row-1, 0)
    sq = tl.load(SQ + qi[:, None]*R + r[None, :], query[:, None] & (r[None, :] < R), 0).to(tl.float32)
    sk = tl.load(SK + ki[:, None]*R + r[None, :], key[:, None] & (r[None, :] < R), 0).to(tl.float32)
    v = tl.load(V + ki[:, None]*DV + d[None, :], key[:, None] & (d[None, :] < DV), 0).to(tl.float32)
    do = tl.load(DO + qi[:, None]*DV + d[None, :], query[:, None] & (d[None, :] < DV), 0).to(tl.float32)
    valid = query[:, None] & key[None, :] & (t[:, None] > t[None, :])
    pair = tl.where(valid, tl.exp(tl.minimum(p[:, None]-p[None, :], 0.))*w[:, None]*w[None, :], 0.)
    if BF16:
        ds = tl.dot(do.to(tl.bfloat16), tl.trans(v).to(tl.bfloat16))*pair
        score = tl.dot(sq.to(tl.bfloat16), tl.trans(sk).to(tl.bfloat16))
    else:
        ds = tl.dot(do, tl.trans(v), input_precision="tf32x3")*pair
        score = tl.dot(sq, tl.trans(sk), input_precision="tf32x3")
    # Consume the two FP32 pair matrices early to limit their live ranges.
    # Tau uses dS before the BF16 cast applied to vector MMAs.
    dtau = tl.sum(tl.sum(ds*score*(dw[:, None]+dw[None, :]), 1), 0)
    if BF16:
        dv = tl.dot(tl.trans((score*pair).to(tl.bfloat16)), do.to(tl.bfloat16))
    else:
        dv = tl.dot(tl.trans(score*pair), do, input_precision="tf32x3")
    tl.atomic_add(DV_OUT + ki[:, None]*DV + d[None, :], dv,
                  key[:, None] & (d[None, :] < DV), sem="relaxed")
    if BF16:
        ds_mma = ds.to(tl.bfloat16)
        dsq = tl.dot(ds_mma, sk.to(tl.bfloat16))
    else:
        dsq = tl.dot(ds, sk, input_precision="tf32x3")
    tl.atomic_add(DSQ + qi[:, None]*R + r[None, :], dsq,
                  query[:, None] & (r[None, :] < R), sem="relaxed")
    if BF16:
        dsk = tl.dot(tl.trans(ds_mma), sq.to(tl.bfloat16))
    else:
        dsk = tl.dot(tl.trans(ds), sq, input_precision="tf32x3")
    tl.atomic_add(DSK + ki[:, None]*R + r[None, :], dsk,
                  key[:, None] & (r[None, :] < R), sem="relaxed")
    if CHUNKS is not None:
        pos = tl.program_id(0).to(tl.int64)*R*DV
        last = tl.sum(tl.where(t == length-1, p, 0.), 0)
        reset = tl.load(RESET + chunk)
        qs = tl.where(query & (reset == 0), w*tl.exp(p), 0.)
        ks = tl.where(key, w*tl.exp(tl.minimum(last-p, 0.)), 0.)
        h = tl.load(H + pos + r[:, None]*DV + d[None, :],
                    (r[:, None] < R) & (d[None, :] < DV), 0)
        if BF16:
            hq = tl.dot(do.to(tl.bfloat16), tl.trans(h).to(tl.bfloat16))
        else:
            hq = tl.dot(do, tl.trans(h), input_precision="tf32x3")
        hq *= qs[:, None]
        dtau += tl.sum(dw*tl.sum(sq*hq, 1), 0)
        tl.atomic_add(DSQ + qi[:, None]*R + r[None, :], hq,
                      query[:, None] & (r[None, :] < R), sem="relaxed")
        g = tl.load(G + pos + r[:, None]*DV + d[None, :],
                    (r[:, None] < R) & (d[None, :] < DV), 0)
        if BF16:
            hk = tl.dot(v.to(tl.bfloat16), tl.trans(g).to(tl.bfloat16))
        else:
            hk = tl.dot(v, tl.trans(g), input_precision="tf32x3")
        hk *= ks[:, None]
        dtau += tl.sum(dw*tl.sum(sk*hk, 1), 0)
        tl.atomic_add(DSK + ki[:, None]*R + r[None, :], hk,
                      key[:, None] & (r[None, :] < R), sem="relaxed")
        if BF16:
            hv = tl.dot(sk.to(tl.bfloat16), g.to(tl.bfloat16))
        else:
            hv = tl.dot(sk, g, input_precision="tf32x3")
        tl.atomic_add(DV_OUT + ki[:, None]*DV + d[None, :], hv*ks[:, None],
                      key[:, None] & (d[None, :] < DV), sem="relaxed")
    if CHUNKS is None:
        tl.store(TAU + stream, dtau)
    else:
        tl.atomic_add(TAU + stream, dtau, sem="relaxed")


@tr.jit
def _chunk_summaries(SQ, SK, V, DO, H, G, A, CHUNKS, ROWS, PREFIX,
                     WEIGHT, LENGTH, RESET, R: tl.constexpr, DV: tl.constexpr,
                     C: tl.constexpr, BR: tl.constexpr, BD: tl.constexpr,
                     BF16: tl.constexpr):
    idx = tl.program_id(0)
    chunk = tl.load(CHUNKS + idx).to(tl.int64)
    t, r, d = tl.arange(0, C), tl.arange(0, BR), tl.arange(0, BD)
    length = tl.load(LENGTH + chunk)
    row = tl.load(ROWS + chunk*C + t)
    p = tl.load(PREFIX + chunk*C + t)
    w = tl.load(WEIGHT + chunk*C + t)
    reset = tl.load(RESET + chunk)
    key, query = (t < length) & (row >= 0), (t < length) & (row < 0)
    ki, qi = tl.where(key, row, 0), tl.where(query, -row-1, 0)
    sq = tl.load(SQ + qi[:, None]*R + r[None, :], query[:, None] & (r[None, :] < R), 0).to(tl.float32)
    sk = tl.load(SK + ki[:, None]*R + r[None, :], key[:, None] & (r[None, :] < R), 0).to(tl.float32)
    v = tl.load(V + ki[:, None]*DV + d[None, :], key[:, None] & (d[None, :] < DV), 0).to(tl.float32)
    do = tl.load(DO + qi[:, None]*DV + d[None, :], query[:, None] & (d[None, :] < DV), 0).to(tl.float32)
    last = tl.sum(tl.where(t == length-1, p, 0.), 0)
    ks = tl.where(key, w*tl.exp(tl.minimum(last-p, 0.)), 0.)
    qs = tl.where(query, w*tl.exp(p), 0.)
    if BF16:
        h = tl.dot(tl.trans((sk*ks[:, None]).to(tl.bfloat16)), v.to(tl.bfloat16))
        g = tl.dot(tl.trans(sq).to(tl.bfloat16), (do*qs[:, None]).to(tl.bfloat16))
    else:
        h = tl.dot(tl.trans(sk*ks[:, None]), v, input_precision="tf32x3")
        g = tl.dot(tl.trans(sq), do*qs[:, None], input_precision="tf32x3")
    pos = idx.to(tl.int64)*R*DV + r[:, None]*DV + d[None, :]
    mask = (r[:, None] < R) & (d[None, :] < DV)
    tl.store(H + pos, h, mask)
    # A reset kills the complete dependence on the preceding chunk, including
    # its local reverse summary, but not this chunk's forward key summary.
    tl.store(G + pos, tl.where(reset != 0, 0., g), mask)
    tl.store(A + idx, tl.where(reset != 0, 0., tl.exp(last)))


@tr.jit
def _summary_scan(H, G, A, OFFSETS, M: tl.constexpr, BLOCK: tl.constexpr):
    """Exclusive forward/reverse scan; overwrite summaries with boundaries.

    Each CTA owns disjoint matrix elements over its entire stream. No MMA,
    atomics, inter-CTA communication, or shared state in the serial loop.
    """
    stream, tile = tl.program_id(0), tl.program_id(1)
    begin = tl.load(OFFSETS + stream)
    end = tl.load(OFFSETS + stream + 1)
    x = tile*BLOCK + tl.arange(0, BLOCK)
    h, g = tl.full((BLOCK,), 0., tl.float32), tl.full((BLOCK,), 0., tl.float32)
    for step in range(end-begin):
        f, b = begin+step, end-1-step
        fp, bp = f.to(tl.int64)*M+x, b.to(tl.int64)*M+x
        sh, sg = tl.load(H+fp, x < M, 0.), tl.load(G+bp, x < M, 0.)
        ah, ag = tl.load(A+f), tl.load(A+b)
        tl.store(H+fp, h, x < M)
        tl.store(G+bp, g, x < M)
        h = ah*h+sh
        g = ag*g+sg


class FusedHardPrefillBackward(HardPrefillBackward):
    @classmethod
    def from_plan(cls, plan, host_offsets, *, host_lengths=None, num_warps=None, long_mode='three_stage', non_blocking=False):
        """Prepare stream buckets outside capture using existing CPU offsets.

        The caller must keep the original device-plan buffers alive; this
        object retains its metadata and upload event just like the base plan.
        No GPU metadata readback or event topology cache is introduced.
        """
        if num_warps not in (None, 2, 4, 8):
            raise ValueError('num_warps must be None, 2, 4 or 8')
        if long_mode not in ('scan', 'three_stage'):
            raise ValueError('long_mode must be scan or three_stage')
        offsets = np.asarray(host_offsets, dtype=np.int64)
        if offsets.ndim != 1 or len(offsets) != plan.streams+1 or offsets[0] != 0:
            raise ValueError('CPU offsets must match the source plan')
        if offsets[-1] != plan.data['lengths'].numel():
            raise ValueError('CPU offsets must cover the source chunks')
        spans = np.diff(offsets)
        if np.any(spans <= 0) or plan.streams > np.iinfo(np.int32).max:
            raise ValueError('expected nonempty int32-addressable streams')
        result = cls.__new__(cls)
        result.__dict__.update(plan.__dict__)
        result.bucket_hosts = []
        result.bucket_devices = []
        def upload(array):
            host = torch.from_numpy(np.ascontiguousarray(array, dtype=np.int32))
            if non_blocking:
                host = host.pin_memory()
                result.bucket_hosts.append(host)
            gpu = host.to(plan.device, non_blocking=non_blocking)
            result.bucket_devices.append(gpu)
            return gpu
        result.fused_warps = num_warps
        result.long_mode = long_mode
        result.single_count = int(np.count_nonzero(spans == 1))
        result.multi_count = int(np.count_nonzero(spans > 1))
        result.single_ids = upload(np.flatnonzero(spans == 1))
        result.multi_ids = upload(np.flatnonzero(spans > 1))
        if host_lengths is None and hasattr(plan, 'packed'):
            host_lengths = plan.packed['lengths']
        if host_lengths is None:
            result.single_buckets = [(plan.c, result.single_ids)] if result.single_count else []
        else:
            lengths = np.asarray(host_lengths, dtype=np.int64)
            if (lengths.ndim != 1 or len(lengths) != offsets[-1]
                    or np.any(lengths < 1) or np.any(lengths > plan.c)):
                raise ValueError('CPU lengths must match the source chunks')
            singles = np.flatnonzero(spans == 1)
            sizes = lengths[offsets[singles]]
            result.single_buckets = []
            for width in (16, 32, 64):
                selected = singles[(sizes <= width) & (sizes > (0 if width == 16 else width//2))]
                if len(selected):
                    result.single_buckets.append((width, upload(selected)))
        multis = np.flatnonzero(spans > 1)
        counts = spans[multis]
        compact_offsets = np.r_[0, np.cumsum(counts)]
        result.multi_chunks = int(compact_offsets[-1])
        if result.multi_chunks > np.iinfo(np.int32).max or offsets[-1] > np.iinfo(np.int32).max:
            raise ValueError('expected int32-addressable chunks')
        # CPU-only bucketing; no synchronization or metadata readback from GPU.
        owners = np.repeat(multis, counts)
        chunk_ids = (np.arange(result.multi_chunks)+
                     np.repeat(offsets[multis]-compact_offsets[:-1], counts))
        for name, array in (('multi_offsets', compact_offsets), ('chunk_ids', chunk_ids),
                            ('chunk_streams', owners)):
            setattr(result, name, upload(array))
        result.bucket_ready = torch.cuda.Event() if non_blocking else None
        if result.bucket_ready is not None:
            result.bucket_ready.record(torch.cuda.current_stream(plan.device))
        return result

    @torch.no_grad()
    def backward(self, sq, sk, value, dout):
        if hasattr(self, 'device_plan'):
            self.device_plan.wait()
        stream = torch.cuda.current_stream(self.device)
        if self.bucket_ready is not None:
            stream.wait_event(self.bucket_ready)
        for tensor in self.bucket_devices:
            tensor.record_stream(stream)
        if (sq.ndim != 2 or sq.shape[0] != self.n or sk.shape != sq.shape
                or value.ndim != 2 or value.shape[0] != self.n or dout.shape != value.shape):
            raise ValueError('expected SQ/SK [N,R], V/dO [N,DV]')
        if sq.shape[1] not in (16, 32) or value.shape[1] not in (32, 64):
            raise ValueError('prototype supports R16/32, DV32/64')
        for x in (sq, sk, value, dout):
            if x.device != self.device or not x.is_contiguous() or x.dtype not in (torch.float32, torch.bfloat16):
                raise ValueError('contiguous BF16/FP32 input on the plan device required')
        dsq, dsk, dv = [torch.zeros_like(x, dtype=torch.float32) for x in (sq, sk, value)]
        partials = torch.zeros((2, self.streams), device=self.device, dtype=torch.float32)
        metadata = [self.data[k] for k in ('offsets', 'rows', 'prefixes', 'weights', 'dlog', 'lengths', 'resets')]
        r, v = sq.shape[1], value.shape[1]
        bd = v if self.precision == 'bf16' else 64
        config = (r, v, self.c, max(32, r), bd, self.precision == 'bf16')
        self.last_kernels = []
        for width, ids in self.single_buckets:
            warps = self.fused_warps or (2 if width == 16 else 4)
            self.last_kernels.append(_chunk_backward[(ids.numel(),)](
                sq, sk, value, dout, dsq, dsk, dv, partials[0], ids,
                *metadata[:-1], *config, BC=width, num_warps=warps, num_stages=1))
        self.matrix_snapshot_bytes = 0
        if self.multi_count and self.long_mode == 'three_stage':
            states = torch.empty((2, self.multi_chunks, r, v), device=self.device, dtype=torch.float32)
            alpha = torch.empty(self.multi_chunks, device=self.device, dtype=torch.float32)
            self.matrix_snapshot_bytes = states.numel()*states.element_size()
            h, g = states[0], states[1]
            warps = self.fused_warps or (2 if self.c == 16 else 4)
            self.last_kernels.append(_chunk_summaries[(self.multi_chunks,)](
                sq, sk, value, dout, h, g, alpha, self.chunk_ids,
                self.data['rows'], self.data['prefixes'], self.data['weights'],
                self.data['lengths'], self.data['resets'], *config,
                num_warps=warps, num_stages=1))
            # An entire R*DV matrix fits comfortably as elementwise state;
            # avoid eight tiny CTAs per stream at the R32/DV64 baseline.
            self.last_kernels.append(_summary_scan[(self.multi_count, 1)](
                h, g, alpha, self.multi_offsets, r*v, r*v, num_warps=4))
            self.last_kernels.append(_chunk_backward[(self.multi_chunks,)](
                sq, sk, value, dout, dsq, dsk, dv, partials[0], self.chunk_streams,
                *metadata[:-1], *config, H=h, G=g, CHUNKS=self.chunk_ids,
                RESET=self.data['resets'], num_warps=warps, num_stages=1))
        elif self.multi_count:
            self.last_kernels.append(_query_backward[(self.multi_count, 1)](
                sq, sk, value, dout, dsq, partials[0], *metadata, *config,
                STREAM_IDS=self.multi_ids, num_warps=8, num_stages=1))
            self.last_kernels.append(_key_backward[(self.multi_count, 1)](
                sq, sk, value, dout, dsk, dv, partials[1], *metadata, *config,
                STREAM_IDS=self.multi_ids, num_warps=8, num_stages=1))
        if hasattr(self, 'tau_groups'):
            contributions = partials[0]+partials[1]
            dtau = torch.zeros(self.tau_groups, device=self.device)
            dtau.scatter_add_(0, self.data['stream_heads'].long(), contributions)
        else:
            dtau = (partials[0]+partials[1]).sum()
        return dict(sq_vec=dsq, sk_vec=dsk, v=dv, rtau=dtau, tau_partials=partials)


class CompactFusedHardPrefillBackward(FusedHardPrefillBackward):
    """Fused consumer for a bucketed compact plan.

    Single-chunk streams come from dense per-width bucket arenas, so no padded
    single-chunk slots are uploaded; multi-chunk streams use the plan's ordinary
    padded arrays. Same math and launch shapes as the packed consumer.
    """
    @classmethod
    def from_compact(cls, plan, *, mma_precision='bf16', num_warps=None,
                     long_mode='three_stage', non_blocking=False):
        if mma_precision not in ('bf16', 'tf32x3'):
            raise ValueError('mma_precision must be bf16 or tf32x3')
        if num_warps not in (None, 2, 4, 8) or long_mode not in ('scan', 'three_stage'):
            raise ValueError('num_warps must be None/2/4/8 and long_mode scan/three_stage')
        stats = plan.statistics
        # Upload may have run on another stream. tau_heads below reads device
        # metadata immediately, so waiting only in backward would be too late.
        plan.wait()
        result = cls.__new__(cls)
        result.storage, result.ready = plan.storage, plan.ready
        result.device = plan.storage.device
        result.n = stats['batch']*stats['heads']*stats['n']
        result.c = stats['chunk_size']
        result.precision, result.fused_warps, result.long_mode = mma_precision, num_warps, long_mode
        result.tau_groups = stats['tau_rows']*stats['heads']
        result.buckets, result.multi = plan.buckets, plan.multi
        result.last_kernels = []
        result.matrix_snapshot_bytes = 0
        blocks = [int(bucket['lengths'].numel()) for bucket in plan.buckets]
        result.block_base = np.cumsum([0] + blocks).astype(np.int64)
        result.single_blocks = int(result.block_base[-1])
        offsets = np.asarray(plan.host_multi_offsets.numpy(), np.int64)
        spans = np.diff(offsets)
        if np.any(spans < 1):
            raise ValueError('multi-chunk stream offsets must be strictly increasing')
        result.multi_streams, result.multi_chunks = int(spans.size), int(offsets[-1])
        result.streams = result.single_blocks + result.multi_streams
        result.bucket_hosts, result.bucket_devices = [], []
        def upload(array):
            host = torch.from_numpy(np.ascontiguousarray(array))
            if non_blocking:
                host = host.pin_memory()
                result.bucket_hosts.append(host)
            gpu = host.to(result.device, non_blocking=non_blocking)
            result.bucket_devices.append(gpu)
            return gpu
        if result.multi_chunks:
            result.multi_chunk_ids = upload(np.arange(result.multi_chunks, dtype=np.int32))
            result.multi_chunk_streams = upload(
                np.repeat(np.arange(result.multi_streams, dtype=np.int32), spans))
            result.multi_stream_offsets = upload(np.r_[0, np.cumsum(spans)].astype(np.int64))
        else:
            result.multi_chunk_ids = result.multi_chunk_streams = None
            result.multi_stream_offsets = upload(np.zeros(1, np.int64))
        heads = [bucket['stream_heads'] for bucket in plan.buckets]
        heads.append(plan.multi['stream_heads'])
        result.tau_heads = torch.cat([h.reshape(-1) for h in heads]).to(result.device).long()
        # cat/long are asynchronous even when host uploads were blocking.
        result.bucket_ready = torch.cuda.Event()
        result.bucket_ready.record(torch.cuda.current_stream(result.device))
        return result

    @torch.no_grad()
    def backward(self, sq, sk, value, dout):
        stream = torch.cuda.current_stream(self.device)
        stream.wait_event(self.ready)
        self.storage.record_stream(stream)
        self.tau_heads.record_stream(stream)
        if self.bucket_ready is not None:
            stream.wait_event(self.bucket_ready)
        for tensor in self.bucket_devices:
            tensor.record_stream(stream)
        if (sq.ndim != 2 or sq.shape[0] != self.n or sk.shape != sq.shape
                or value.ndim != 2 or value.shape[0] != self.n or dout.shape != value.shape):
            raise ValueError('expected SQ/SK [N,R], V/dO [N,DV]')
        if sq.shape[1] not in (16, 32) or value.shape[1] not in (32, 64):
            raise ValueError('prototype supports R16/32, DV32/64')
        for x in (sq, sk, value, dout):
            if x.device != self.device or not x.is_contiguous() or x.dtype not in (torch.float32, torch.bfloat16):
                raise ValueError('contiguous BF16/FP32 input on the plan device required')
        dsq, dsk, dv = [torch.zeros_like(x, dtype=torch.float32) for x in (sq, sk, value)]
        partials = torch.zeros((2, self.streams), device=self.device, dtype=torch.float32)
        r, v = sq.shape[1], value.shape[1]
        bd = v if self.precision == 'bf16' else 64
        config = (r, v, self.c, max(32, r), bd, self.precision == 'bf16')
        self.last_kernels = []
        for i, bucket in enumerate(self.buckets):
            blocks = int(bucket['lengths'].numel())
            if not blocks:
                continue
            width = bucket['width']
            warps = self.fused_warps or (2 if width == 16 else 4)
            lo = int(self.block_base[i])
            self.last_kernels.append(_chunk_backward[(blocks,)](
                sq, sk, value, dout, dsq, dsk, dv, partials[0, lo:lo + blocks], None,
                None, bucket['rows'], bucket['prefixes'], bucket['weights'], bucket['dlog'],
                bucket['lengths'], r, v, width, max(32, r), bd, self.precision == 'bf16',
                BC=width, LOCAL=1, num_warps=warps, num_stages=1))
        base = self.single_blocks
        if self.multi_chunks:
            m = self.multi
            metadata = [m['offsets'], None, m['rows'], m['prefixes'], m['weights'],
                        m['dlog'], m['lengths'], m['resets']]
            warps = self.fused_warps or (2 if self.c == 16 else 4)
            if self.long_mode == 'three_stage':
                states = torch.empty((2, self.multi_chunks, r, v), device=self.device,
                                     dtype=torch.float32)
                alpha = torch.empty(self.multi_chunks, device=self.device, dtype=torch.float32)
                self.matrix_snapshot_bytes = states.numel()*states.element_size()
                h, g = states[0], states[1]
                self.last_kernels.append(_chunk_summaries[(self.multi_chunks,)](
                    sq, sk, value, dout, h, g, alpha, self.multi_chunk_ids, m['rows'],
                    m['prefixes'], m['weights'], m['lengths'], m['resets'], *config,
                    num_warps=warps, num_stages=1))
                self.last_kernels.append(_summary_scan[(self.multi_streams, 1)](
                    h, g, alpha, self.multi_stream_offsets, r*v, r*v, num_warps=4))
                self.last_kernels.append(_chunk_backward[(self.multi_chunks,)](
                    sq, sk, value, dout, dsq, dsk, dv, partials[0, base:],
                    self.multi_chunk_streams, m['offsets'], m['rows'], m['prefixes'],
                    m['weights'], m['dlog'], m['lengths'], *config, H=h, G=g,
                    CHUNKS=self.multi_chunk_ids, RESET=m['resets'],
                    num_warps=warps, num_stages=1))
            else:
                raise ValueError('compact plans require long_mode three_stage')
        contributions = partials[0] + partials[1]
        dtau = torch.zeros(self.tau_groups, device=self.device, dtype=torch.float32)
        dtau.scatter_add_(0, self.tau_heads, contributions)
        return dict(sq_vec=dsq, sk_vec=dsk, v=dv, rtau=dtau, tau_partials=partials)
