"""Native-plan hard backward kernels. Global FP32 vector atomics only."""
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _query_backward(SQ, SK, V, DO, DSQ, TAU, OFFSETS, ROWS, PREFIX, WEIGHT,
                    DLOG, LENGTH, RESET, R: tl.constexpr, DV: tl.constexpr,
                    C: tl.constexpr, BR: tl.constexpr, BD: tl.constexpr,
                    BF16: tl.constexpr, STREAM_IDS = None):
    stream = tl.program_id(0)
    if STREAM_IDS is not None:
        stream = tl.load(STREAM_IDS + stream)
    begin = tl.load(OFFSETS + stream)
    end = tl.load(OFFSETS + stream + 1)
    tile = tl.program_id(1)
    t, r, d = tl.arange(0, C), tl.arange(0, BR), tile*BD + tl.arange(0, BD)
    # Transposed forward state: sum V * SK^T, so the readout is dO @ state.
    state = tl.full((BD, BR), 0, tl.float32)
    dtau = tl.full((), 0, tl.float32)
    for chunk in range(begin, end):
        length = tl.load(LENGTH + chunk)
        reset = tl.load(RESET + chunk)
        row = tl.load(ROWS + chunk*C + t)
        p = tl.load(PREFIX + chunk*C + t)
        w = tl.load(WEIGHT + chunk*C + t)
        key, query = (t < length) & (row >= 0), (t < length) & (row < 0)
        ki, qi = tl.where(key, row, 0), tl.where(query, -row-1, 0)
        do = tl.load(DO + qi[:, None]*DV + d[None, :], query[:, None] & (d[None, :] < DV), 0).to(tl.float32)
        v = tl.load(V + ki[:, None]*DV + d[None, :], key[:, None] & (d[None, :] < DV), 0).to(tl.float32)
        sk = tl.load(SK + ki[:, None]*R + r[None, :], key[:, None] & (r[None, :] < R), 0).to(tl.float32)
        if BF16:
            history = tl.dot(do.to(tl.bfloat16), state.to(tl.bfloat16))
            score = tl.dot(do.to(tl.bfloat16), tl.trans(v).to(tl.bfloat16))
        else:
            history = tl.dot(do, state, input_precision="tf32x3")
            score = tl.dot(do, tl.trans(v), input_precision="tf32x3")
        history *= (tl.where(reset != 0, 0., tl.exp(p))*w)[:, None]
        valid = query[:, None] & key[None, :] & (t[:, None] > t[None, :])
        pair = tl.where(valid, tl.exp(tl.minimum(p[:, None]-p[None, :], 0.))*w[:, None]*w[None, :], 0.)
        if BF16:
            local = tl.dot((score*pair).to(tl.bfloat16), sk.to(tl.bfloat16))
        else:
            local = tl.dot(score*pair, sk, input_precision="tf32x3")
        grad = history + local
        tl.atomic_add(DSQ + qi[:, None]*R + r[None, :], grad,
                      query[:, None] & (r[None, :] < R), sem="relaxed")
        sq = tl.load(SQ + qi[:, None]*R + r[None, :], query[:, None] & (r[None, :] < R), 0).to(tl.float32)
        dw = tl.load(DLOG + chunk*C + t)
        dtau += tl.sum(tl.where(query, dw*tl.sum(sq*grad, 1), 0.), 0)
        last = tl.sum(tl.where(t == length-1, p, 0.), 0)
        scale = tl.where(key, w*tl.exp(tl.minimum(last-p, 0.)), 0.)
        if BF16:
            summary = tl.dot(tl.trans(v).to(tl.bfloat16), (sk*scale[:, None]).to(tl.bfloat16))
        else:
            summary = tl.dot(tl.trans(v), sk*scale[:, None], input_precision="tf32x3")
        state = tl.where(reset != 0, 0., tl.exp(last))*state + summary
    tl.store(TAU + stream*tl.cdiv(DV, BD) + tile, dtau)


@tr.jit
def _key_backward(SQ, SK, V, DO, DSK, DV_OUT, TAU, OFFSETS, ROWS, PREFIX,
                  WEIGHT, DLOG, LENGTH, RESET, R: tl.constexpr, DV: tl.constexpr,
                  C: tl.constexpr, BR: tl.constexpr, BD: tl.constexpr,
                  BF16: tl.constexpr, STREAM_IDS = None):
    stream = tl.program_id(0)
    if STREAM_IDS is not None:
        stream = tl.load(STREAM_IDS + stream)
    begin = tl.load(OFFSETS + stream)
    end = tl.load(OFFSETS + stream + 1)
    tile = tl.program_id(1)
    t, r, d = tl.arange(0, C), tl.arange(0, BR), tile*BD + tl.arange(0, BD)
    state = tl.full((BR, BD), 0, tl.float32)
    dtau = tl.full((), 0, tl.float32)
    for reverse in range(end-begin):
        chunk = end-1-reverse
        length = tl.load(LENGTH + chunk)
        reset = tl.load(RESET + chunk)
        row = tl.load(ROWS + chunk*C + t)
        p = tl.load(PREFIX + chunk*C + t)
        w = tl.load(WEIGHT + chunk*C + t)
        dw = tl.load(DLOG + chunk*C + t)
        key, query = (t < length) & (row >= 0), (t < length) & (row < 0)
        ki, qi = tl.where(key, row, 0), tl.where(query, -row-1, 0)
        sq = tl.load(SQ + qi[:, None]*R + r[None, :], query[:, None] & (r[None, :] < R), 0).to(tl.float32)
        sk = tl.load(SK + ki[:, None]*R + r[None, :], key[:, None] & (r[None, :] < R), 0).to(tl.float32)
        do = tl.load(DO + qi[:, None]*DV + d[None, :], query[:, None] & (d[None, :] < DV), 0).to(tl.float32)
        v = tl.load(V + ki[:, None]*DV + d[None, :], key[:, None] & (d[None, :] < DV), 0).to(tl.float32)
        last = tl.sum(tl.where(t == length-1, p, 0.), 0)
        key_scale = tl.where(key, w*tl.exp(tl.minimum(last-p, 0.)), 0.)
        valid = key[:, None] & query[None, :] & (t[:, None] < t[None, :])
        pair = tl.where(valid, tl.exp(tl.minimum(p[None, :]-p[:, None], 0.))*w[:, None]*w[None, :], 0.)
        if BF16:
            gsk = tl.dot(v.to(tl.bfloat16), tl.trans(state).to(tl.bfloat16))
            score_v = tl.dot(v.to(tl.bfloat16), tl.trans(do).to(tl.bfloat16))
            local_sk = tl.dot((score_v*pair).to(tl.bfloat16), sq.to(tl.bfloat16))
        else:
            gsk = tl.dot(v, tl.trans(state), input_precision="tf32x3")
            score_v = tl.dot(v, tl.trans(do), input_precision="tf32x3")
            local_sk = tl.dot(score_v*pair, sq, input_precision="tf32x3")
        gsk = gsk*key_scale[:, None] + local_sk
        tl.atomic_add(DSK + ki[:, None]*R + r[None, :], gsk,
                      key[:, None] & (r[None, :] < R), sem="relaxed")
        dtau += tl.sum(tl.where(key, dw*tl.sum(sk*gsk, 1), 0.), 0)
        if BF16:
            gv = tl.dot(sk.to(tl.bfloat16), state.to(tl.bfloat16))
            score_s = tl.dot(sk.to(tl.bfloat16), tl.trans(sq).to(tl.bfloat16))
            local_v = tl.dot((score_s*pair).to(tl.bfloat16), do.to(tl.bfloat16))
        else:
            gv = tl.dot(sk, state, input_precision="tf32x3")
            score_s = tl.dot(sk, tl.trans(sq), input_precision="tf32x3")
            local_v = tl.dot(score_s*pair, do, input_precision="tf32x3")
        gv = gv*key_scale[:, None] + local_v
        tl.atomic_add(DV_OUT + ki[:, None]*DV + d[None, :], gv,
                      key[:, None] & (d[None, :] < DV), sem="relaxed")
        query_scale = tl.where(query, w*tl.exp(p), 0.)
        if BF16:
            summary = tl.dot(tl.trans(sq).to(tl.bfloat16), (do*query_scale[:, None]).to(tl.bfloat16))
        else:
            summary = tl.dot(tl.trans(sq), do*query_scale[:, None], input_precision="tf32x3")
        # A reset at the first key kills ALL dependence on the preceding chunk.
        state = tl.where(reset != 0, 0., tl.exp(last)*state + summary)
    tl.store(TAU + stream*tl.cdiv(DV, BD) + tile, dtau)


class HardPrefillBackward:
    """Consume native CPU plans; vectors are flattened contiguous BNHD."""
    @classmethod
    def from_native(cls, prepared, *, mma_precision='tf32x3'):
        """Consume one uploaded native batch plan; vectors are flattened BNHD.

        rtau is returned as flat tau groups (H for head-shared input, B*H for
        explicit per-row/head input). Soft row/head outputs remain zero.
        """
        if mma_precision not in ('bf16', 'tf32x3'):
            raise ValueError('mma_precision must be bf16 or tf32x3')
        result = cls.__new__(cls)
        stats = prepared.statistics
        result.device = prepared.storage.device
        result.n = stats['batch']*stats['heads']*stats['n']
        result.c, result.precision = stats['chunk_size'], mma_precision
        result.data, result.streams = prepared.data, stats['streams']
        result.device_plan, result.tau_groups = prepared, stats['tau_rows']*stats['heads']
        result.last_kernels = []
        return result

    @torch.no_grad()
    def backward(self, sq, sk, value, dout):
        if hasattr(self, 'device_plan'):
            self.device_plan.wait()
        if (sq.ndim != 2 or sq.shape[0] != self.n or sk.shape != sq.shape
                or value.ndim != 2 or value.shape[0] != self.n or dout.shape != value.shape):
            raise ValueError('expected SQ/SK [N,R], V/dO [N,DV]')
        if sq.shape[1] not in (16, 32) or value.shape[1] not in (32, 64):
            raise ValueError('prototype supports R16/32, DV32/64')
        for x in (sq, sk, value, dout):
            if x.device != self.device or not x.is_contiguous() or x.dtype not in (torch.float32, torch.bfloat16):
                raise ValueError('contiguous BF16/FP32 input on the plan device required')
        dsq, dsk, dv = [torch.zeros_like(x, dtype=torch.float32) for x in (sq, sk, value)]
        # The wider padded TF32x3 MMA layout avoids the heavy layout-conversion
        # spills observed with a 32-wide value tile on sm120.
        bd = value.shape[1] if self.precision == 'bf16' else 64
        tiles = tr.cdiv(value.shape[1], bd)
        partials = torch.zeros((2, self.streams*tiles), device=self.device, dtype=torch.float32)
        self.last_kernels = []
        if self.streams:
            metadata = [self.data[k] for k in ('offsets', 'rows', 'prefixes', 'weights', 'dlog', 'lengths', 'resets')]
            config = (sq.shape[1], value.shape[1], self.c, max(32, sq.shape[1]), bd, self.precision == 'bf16')
            warps = 8
            self.last_kernels.append(_query_backward[(self.streams, tiles)](
                sq, sk, value, dout, dsq, partials[0], *metadata, *config, num_warps=warps, num_stages=1))
            self.last_kernels.append(_key_backward[(self.streams, tiles)](
                sq, sk, value, dout, dsk, dv, partials[1], *metadata, *config, num_warps=warps, num_stages=1))
        # Pair query/key terms per stream before summing across streams.
        if hasattr(self, 'tau_groups'):
            contributions = (partials[0]+partials[1]).reshape(self.streams, tiles).sum(-1)
            dtau = torch.zeros(self.tau_groups, device=self.device, dtype=torch.float32)
            dtau.scatter_add_(0, self.data['stream_heads'].long(), contributions)
        else:
            dtau = (partials[0] + partials[1]).sum()
        return dict(sq_vec=dsq, sk_vec=dsk, v=dv, rtau=dtau, tau_partials=partials)
