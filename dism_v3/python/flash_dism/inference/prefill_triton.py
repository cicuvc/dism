"""Experimental chunked vector executor for HardPrefillPlan.

One CTA per event stream, sequential chunks, register-carried R*DV state.
No per-event vector partials or per-chunk matrix snapshots in global memory.
"""
import numpy as np
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _execute(SQ, SK, V, OUT, OFFSETS, STREAM_IDS, ROWS, LOG_PREFIX, WEIGHTS,
             LENGTHS, RESET, R: tl.constexpr, DV: tl.constexpr,
             C: tl.constexpr, BR: tl.constexpr, BD: tl.constexpr, BF16: tl.constexpr):
    stream = tl.load(STREAM_IDS + tl.program_id(0))
    begin = tl.load(OFFSETS + stream)
    end = tl.load(OFFSETS + stream + 1)
    t = tl.arange(0, C)
    r = tl.arange(0, BR)
    d = tl.arange(0, BD)
    state = tl.full((BR, BD), 0, tl.float32)
    for chunk in range(begin, end):
        length = tl.load(LENGTHS + chunk)
        reset = tl.load(RESET + chunk)
        row = tl.load(ROWS + chunk*C + t)
        prefix = tl.load(LOG_PREFIX + chunk*C + t)
        weight = tl.load(WEIGHTS + chunk*C + t)
        is_key = (t < length) & (row >= 0)
        is_query = (t < length) & (row < 0)
        ki = tl.where(is_key, row, 0)
        qi = tl.where(is_query, -row-1, 0)
        sq = tl.load(SQ + qi[:, None]*R + r[None, :],
                     is_query[:, None] & (r[None, :] < R), 0).to(tl.float32)
        sk = tl.load(SK + ki[:, None]*R + r[None, :],
                     is_key[:, None] & (r[None, :] < R), 0).to(tl.float32)
        v = tl.load(V + ki[:, None]*DV + d[None, :],
                    is_key[:, None] & (d[None, :] < DV), 0).to(tl.float32)

        # Prefix log decays are non-increasing; no inverse small weights.
        history_scale = tl.where(reset != 0, 0., tl.exp(prefix))
        if BF16:
            history = tl.dot(sq.to(tl.bfloat16), state.to(tl.bfloat16))
        else:
            history = tl.dot(sq, state, input_precision="tf32x3")
        history *= (history_scale*weight)[:, None]
        valid_pair = is_query[:, None] & is_key[None, :] & (t[:, None] > t[None, :])
        log_ratio = tl.minimum(prefix[:, None]-prefix[None, :], 0.)
        pair = tl.where(valid_pair, tl.exp(log_ratio)*weight[:, None]*weight[None, :], 0.)
        if BF16:
            score = tl.dot(sq.to(tl.bfloat16), tl.trans(sk).to(tl.bfloat16))
            local = tl.dot((score*pair).to(tl.bfloat16), v.to(tl.bfloat16))
        else:
            score = tl.dot(sq, tl.trans(sk), input_precision="tf32x3")
            local = tl.dot(score*pair, v, input_precision="tf32x3")
        tl.atomic_add(OUT + qi[:, None]*DV + d[None, :], history+local,
                      is_query[:, None] & (d[None, :] < DV), sem="relaxed")

        last_prefix = tl.sum(tl.where(t == length-1, prefix, 0.), 0)
        key_scale = tl.where(is_key, weight*tl.exp(tl.minimum(last_prefix-prefix, 0.)), 0.)
        if BF16:
            summary = tl.dot(tl.trans(sk).to(tl.bfloat16), (v*key_scale[:, None]).to(tl.bfloat16))
        else:
            summary = tl.dot(tl.trans(sk), v*key_scale[:, None], input_precision="tf32x3")
        state = tl.where(reset != 0, 0., tl.exp(last_prefix))*state + summary


class TritonPrefillPlan:
    """Explicitly prepared/uploaded plan. execute() performs no host transfers.

    Labels/scalar planning remains CPU-side. Construct outside graph capture.
    Supports contiguous FP32/BF16 vectors on one CUDA device, R/DV <= 128.
    Atomic FP32 output is nondeterministic in the last few bits; inference only.
    """
    def __init__(self, plan, *, device="cuda", chunk_size=16, mma_precision="bf16",
                 dispatch_threshold=64):
        self._initialize(plan.native.chunks(chunk_size), plan.native.n,
                         device, chunk_size, mma_precision, dispatch_threshold)

    @classmethod
    def from_packed(cls, packed, n, *, device="cuda", chunk_size=16, mma_precision="bf16",
                    dispatch_threshold=64):
        """Internal trusted native chunk interface: no second chunk packing."""
        result = cls.__new__(cls)
        result._initialize(packed, n, device, chunk_size, mma_precision, dispatch_threshold)
        return result

    def _initialize(self, packed, n, device, chunk_size, mma_precision, dispatch_threshold):
        if chunk_size not in (16, 32, 64):
            raise ValueError("chunk_size must be 16, 32 or 64")
        if mma_precision not in ("tf32x3", "bf16"):
            raise ValueError("mma_precision must be tf32x3 or bf16")
        self.mma_precision = mma_precision
        self.n = n
        self.c = chunk_size
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("CUDA device required")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.streams = len(packed["offsets"])-1
        self.chunks = len(packed["lengths"])
        def upload(data, dtype):
            return torch.as_tensor(np.asarray(data), dtype=dtype, device=self.device).contiguous()
        self.offsets = upload(packed["offsets"], torch.int64)
        self.rows = upload(packed["rows"], torch.int32)
        self.prefixes = upload(packed["prefixes"], torch.float32)
        self.weights = upload(packed["weights"], torch.float32)
        self.lengths = upload(packed["lengths"], torch.int32)
        self.resets = upload(packed["resets"], torch.int32)
        # Dispatch routing. The planner already gives per-stream chunk counts via
        # offsets; split streams into a serial group and a chunk-parallel group.
        counts = np.diff(np.asarray(packed["offsets"]))
        long_mask = counts > int(dispatch_threshold)
        long_streams = np.nonzero(long_mask)[0]
        bounds = np.asarray(packed["offsets"])
        long_chunks = [np.arange(bounds[s], bounds[s+1], dtype=np.int64) for s in long_streams]
        lens = [int(c.size) for c in long_chunks]
        def upload_ids(values):
            return torch.as_tensor(np.asarray(values, dtype=np.int64), device=self.device)
        self.dispatch_threshold = int(dispatch_threshold)
        self.stream_ids = upload_ids(np.arange(self.streams))
        self.chunk_ids = upload_ids(np.arange(self.chunks))
        self.local_offsets = upload_ids(bounds[:-1])
        self.short_stream_ids = upload_ids(np.nonzero(~long_mask)[0])
        self.long_stream_ids = upload_ids(long_streams)
        self.long_chunk_ids = upload_ids(np.concatenate(long_chunks) if long_chunks else np.zeros(0, np.int64))
        self.long_local_offsets = upload_ids(np.concatenate(([0], np.cumsum(lens)[:-1])) if lens else np.zeros(0, np.int64))
        self.last_kernel = None

    @torch.no_grad()
    def _check_vectors(self, sq, sk, value):
        if sq.ndim != 2 or sk.shape != sq.shape or value.ndim != 2 or sq.shape[0] != self.n or value.shape[0] != self.n:
            raise ValueError("expected sq/sk [N,R], value [N,DV]")
        if not (1 <= sq.shape[1] <= 128 and 1 <= value.shape[1] <= 128):
            raise ValueError("R/DV must be in [1,128]")
        for x in (sq, sk, value):
            if x.device != self.device or not x.is_contiguous() or x.dtype not in (torch.float32, torch.bfloat16):
                raise ValueError("contiguous FP32/BF16 tensors on the plan device required")
            if x.requires_grad:
                raise ValueError("inference only")

    @torch.no_grad()
    def execute(self, sq, sk, value):
        self._check_vectors(sq, sk, value)
        out = torch.zeros((self.n, value.shape[1]), device=self.device, dtype=torch.float32)
        if self.streams:
            self.last_kernel = _execute[(self.streams,)](
                sq, sk, value, out, self.offsets, self.stream_ids, self.rows, self.prefixes, self.weights,
                self.lengths, self.resets, sq.shape[1], value.shape[1], self.c,
                max(32, tr.next_power_of_2(sq.shape[1])), max(32, tr.next_power_of_2(value.shape[1])),
                self.mma_precision == "bf16",
                num_warps=8, num_stages=1)
        return out

    @torch.no_grad()
    def execute_parallel(self, sq, sk, value):
        """Three-phase chunk-parallel forward over the whole plan.

        Equivalent to execute() up to FP32 atomic ordering and MMA tiling.
        Materializes one [BR, BD] FP32 state per chunk.
        """
        self._check_vectors(sq, sk, value)
        from .prefill_triton_parallel import run_parallel
        self.last_kernel = None
        return run_parallel(self, sq, sk, value, self.chunk_ids, self.stream_ids, self.local_offsets)

    @torch.no_grad()
    def execute_dispatch(self, sq, sk, value):
        """Route streams by chunk count: serial carry vs chunk-parallel phases.

        Streams with at most `dispatch_threshold` chunks keep the original
        register-carried `_execute`; longer streams go through summary -> state
        passing -> per-chunk output. Both write the same FP32 output.
        """
        self._check_vectors(sq, sk, value)
        from .prefill_triton_parallel import run_parallel
        out = torch.zeros((self.n, value.shape[1]), device=self.device, dtype=torch.float32)
        if self.short_stream_ids.numel():
            _execute[(int(self.short_stream_ids.numel()),)](
                sq, sk, value, out, self.offsets, self.short_stream_ids, self.rows, self.prefixes,
                self.weights, self.lengths, self.resets, sq.shape[1], value.shape[1], self.c,
                max(32, tr.next_power_of_2(sq.shape[1])), max(32, tr.next_power_of_2(value.shape[1])),
                self.mma_precision == "bf16", num_warps=8, num_stages=1)
        if self.long_stream_ids.numel():
            run_parallel(self, sq, sk, value, self.long_chunk_ids, self.long_stream_ids,
                         self.long_local_offsets, out)
        self.last_kernel = None
        return out

    def statistics(self):
        tensors = (self.offsets, self.rows, self.prefixes, self.weights, self.lengths, self.resets)
        return dict(streams=self.streams, chunks=self.chunks, chunk_size=self.c, mma_precision=self.mma_precision,
                    dispatch_threshold=self.dispatch_threshold,
                    short_streams=int(self.short_stream_ids.numel()),
                    long_streams=int(self.long_stream_ids.numel()),
                    long_chunks=int(self.long_chunk_ids.numel()),
                    device_metadata_bytes=sum(x.numel()*x.element_size() for x in tensors),
                    vector_snapshot_bytes=0)
