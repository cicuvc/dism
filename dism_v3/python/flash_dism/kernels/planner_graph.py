"""Fixed-grid compact consumers. Only scheduling changes; reuse eager math.

Descriptor ABI v1: four records of 16 int64 words (16/32/64/multi).
Each record is [chunks, streams, rows, prefixes, weights, dlog, lengths,
heads, offsets, resets, chunk_ids, chunk_owners, H, G, alpha, tau_partials].
Words 2..15 are device pointers. Empty records may contain null pointers.
The descriptor and its owners must remain live until GPU AND upload retire.
"""
import torch
import triton as tr
import triton.language as tl

from .hard_backward_fused import _chunk_backward, _chunk_summaries, _summary_scan


@tr.jit
def acquire(ptr):
    return tl.inline_asm_elementwise('ld.acquire.sys.global.u32 $0, [$1];',
        constraints='=r,l', args=[ptr], dtype=tl.uint32, is_pure=False, pack=1)


@tr.jit
def sample_route(Ready, Expected, Choice):
    seen = acquire(Ready)
    epoch = tl.load(Expected).to(tl.uint32)
    selected = ((epoch != 0) & (seen == epoch)).to(tl.uint32)
    tl.inline_asm_elementwise(
        '{ .reg .pred p; .reg .u32 t; mov.u32 t, %tid.x; '
        'setp.eq.u32 p, t, 0; @p st.release.sys.global.u32 [$1], $2; '
        'mov.u32 $0, 0; }', constraints='=r,l,r', args=[Choice, selected],
        dtype=tl.uint32, is_pure=False, pack=1)


@tr.jit
def pointer(D, offset: tl.constexpr, dtype: tl.constexpr):
    return tl.load(D+offset).to(tl.pointer_type(dtype))


@tr.jit
def compact_stage(D, Choice, SQ, SK, V, DO, DQ, DK, DV_OUT,
                  R: tl.constexpr, DV: tl.constexpr, C: tl.constexpr,
                  STAGE: tl.constexpr):
    # A CTA cannot read even the descriptor until it acquires the frozen choice.
    if acquire(Choice) == 1:
        count = tl.load(D + (1 if STAGE == 2 else 0)).to(tl.int32)
        rows = pointer(D, 2, tl.int32)
        prefix, weight, dlog = pointer(D, 3, tl.float32), pointer(D, 4, tl.float32), pointer(D, 5, tl.float32)
        length = pointer(D, 6, tl.int32)
        offsets, reset = pointer(D, 8, tl.int64), pointer(D, 9, tl.int32)
        ids, owners = pointer(D, 10, tl.int32), pointer(D, 11, tl.int32)
        h, g, a = pointer(D, 12, tl.float32), pointer(D, 13, tl.float32), pointer(D, 14, tl.float32)
        tau = pointer(D, 15, tl.float32)
        for task in range(tl.program_id(0), count, tl.num_programs(0)):
            if STAGE == 0:
                _chunk_backward(SQ, SK, V, DO, DQ, DK, DV_OUT, tau, None,
                    None, rows, prefix, weight, dlog, length, R, DV, C, 32, DV, True,
                    BC=C, LOCAL=1, TASK=task)
            elif STAGE == 1:
                _chunk_summaries(SQ, SK, V, DO, h, g, a, ids, rows, prefix,
                    weight, length, reset, R, DV, C, 32, DV, True, TASK=task)
            elif STAGE == 2:
                _summary_scan(h, g, a, offsets, R*DV, R*DV, TASK=task)
            else:
                _chunk_backward(SQ, SK, V, DO, DQ, DK, DV_OUT, tau, owners,
                    offsets, rows, prefix, weight, dlog, length, R, DV, C, 32, DV, True,
                    H=h, G=g, CHUNKS=ids, RESET=reset, TASK=task)


@tr.jit
def tau_stage(D, Choice, OUT, ZERO: tl.constexpr, BLOCK: tl.constexpr):
    if acquire(Choice) == 1:
        count = tl.load(D+1)
        tau, heads = pointer(D, 15, tl.float32), pointer(D, 7, tl.int32)
        for tile in range(tl.program_id(0), tl.cdiv(count, BLOCK), tl.num_programs(0)):
            i = tile*BLOCK + tl.arange(0, BLOCK)
            if ZERO:
                tl.store(tau+i, 0., i < count)
            else:
                head = tl.load(heads+i, i < count, 0)
                val = tl.load(tau+i, i < count, 0.)
                tl.atomic_add(OUT+head, val, i < count, sem='relaxed')


class GraphCompactBackward:
    """Explicit P3 API, C64/BF16 R32/DV64 only; no model dispatch changes."""
    def __init__(self, descriptor, choice, *, ctas=32):
        if (descriptor.dtype != torch.int64 or descriptor.shape != (64,)
                or not descriptor.is_cuda or not descriptor.is_contiguous()
                or choice.device != descriptor.device or choice.dtype != torch.int32
                or choice.numel() != 1 or not choice.is_contiguous() or ctas < 1):
            raise ValueError('CUDA int64[64] descriptor, int32 choice, positive grid required')
        self.descriptor, self.choice, self.ctas = descriptor, choice, ctas
        self.last_kernels = []

    @torch.no_grad()
    def backward(self, sq, sk, value, dout, *, heads):
        if (sq.ndim != 2 or sq.shape[1] != 32 or sk.shape != sq.shape
                or value.shape != (sq.shape[0], 64) or dout.shape != value.shape
                or heads < 1 or any(x.device != self.descriptor.device or
                    x.dtype != torch.bfloat16 or not x.is_contiguous()
                    for x in (sq, sk, value, dout))):
            raise ValueError('contiguous BF16 SQ/SK [N,32], V/dO [N,64] required')
        dq, dk, dv = [torch.zeros_like(x, dtype=torch.float32) for x in (sq, sk, value)]
        dt = torch.zeros(heads, dtype=torch.float32, device=sq.device)
        args = (self.choice, sq, sk, value, dout, dq, dk, dv, 32, 64)
        self.last_kernels = []
        for i, width in enumerate((16, 32, 64, 64)):
            desc = self.descriptor[i*16:]
            self.last_kernels.append(tau_stage[(self.ctas,)](
                desc, self.choice, dt, True, 256))
            for stage in ((0,) if i < 3 else (1, 2, 3)):
                self.last_kernels.append(compact_stage[(self.ctas, 1)](
                    desc, *args, width, stage,
                    num_warps=2 if width == 16 else 4, num_stages=1))
            self.last_kernels.append(tau_stage[(self.ctas,)](
                desc, self.choice, dt, False, 256))
        return dict(sq_vec=dq, sk_vec=dk, v=dv, rtau=dt)
