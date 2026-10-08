"""Fused vocabulary/core layout glue; all arithmetic rounding is intentional."""
import torch
import triton
import triton.language as tl


@triton.jit
def _prepare_vocab(EQ, EK, OEQ, OEK,
                   H: tl.constexpr, D: tl.constexpr, V: tl.constexpr,
                   QH: tl.constexpr, QV: tl.constexpr, QD: tl.constexpr,
                   KH: tl.constexpr, KV: tl.constexpr, KD: tl.constexpr,
                   BLOCK: tl.constexpr):
    i = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
    d = i % D
    h = i // (V*D)
    v = i // D % V
    q = tl.load(EQ+h*QH+v*QV+d*QD, i < H*V*D, other=0)
    k = tl.load(EK+h*KH+v*KV+d*KD, i < H*V*D, other=0)
    tl.store(OEQ+i, q, i < H*V*D)
    tl.store(OEK+i, k, i < H*V*D)


def prepare_embedding(q, k, eq, ek):
    """Keep BNHD token inputs as aliases; only materialize BF16 codebooks."""
    _, _, h, d = q.shape
    v = eq.shape[-2]
    qe = torch.empty((h,v,d), dtype=torch.bfloat16, device=q.device)
    ke = torch.empty_like(qe)
    strides = lambda t: ((0, *t.stride()) if t.ndim == 2 else t.stride())
    _prepare_vocab[(triton.cdiv(qe.numel(), 256),)](
        eq, ek, qe, ke, h, d, v, *strides(eq), *strides(ek), 256)
    return q, k, qe, ke


@triton.jit
def _select(Q, K, QFK, KFQ, Direction, A, B, N: tl.constexpr, H: tl.constexpr,
            D: tl.constexpr, TOTAL: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    h = i // D % H
    b = i // (D*H*N)
    choose = tl.load(Direction+b*H+h, i < TOTAL, other=0)
    q = tl.load(Q+i, i < TOTAL, other=0)
    k = tl.load(K+i, i < TOTAL, other=0)
    qfk = tl.load(QFK+i, i < TOTAL, other=0)
    kfq = tl.load(KFQ+i, i < TOTAL, other=0)
    tl.store(A+i, tl.where(choose, q, kfq), i < TOTAL)
    tl.store(B+i, tl.where(choose, qfk, k), i < TOTAL)


def select_operands(q, k, qfk, kfq, direction):
    """BNHD originals + BNHD interpolants -> two contiguous BNHD operands."""
    q, k = q.contiguous(), k.contiguous()
    qfk, kfq = qfk.contiguous(), kfq.contiguous()
    direction = direction.contiguous()
    b, n, h, d = q.shape
    a, out_b = torch.empty_like(q), torch.empty_like(k)
    if not q.numel():
        return a, out_b
    _select[(triton.cdiv(q.numel(), 256),)](
        q, k, qfk, kfq, direction, a, out_b, n, h, d, q.numel(), 256)
    return a, out_b


@triton.jit
def _split(A, B, Direction, OQ, OK, N: tl.constexpr, H: tl.constexpr,
           D: tl.constexpr, TOTAL: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
    h = i // D % H
    b = i // (D*H*N)
    choose = tl.load(Direction+b*H+h, i < TOTAL, other=0)
    # Core dA is FP32, dB is BF16. Cast BEFORE selection/interpolation.
    da = tl.load(A+i, i < TOTAL, other=0).to(tl.bfloat16)
    db = tl.load(B+i, i < TOTAL, other=0).to(tl.bfloat16)
    tl.store(OQ+i, tl.where(choose, db, 0), i < TOTAL)
    tl.store(OK+i, tl.where(choose, 0, da), i < TOTAL)


def split_interpolation_gradients(da, db, direction):
    b, n, h, d = da.shape
    doq = torch.empty((b, n, h, d), dtype=torch.bfloat16, device=da.device)
    dok = torch.empty_like(doq)
    if not da.numel():
        return doq, dok
    _split[(triton.cdiv(da.numel(), 256),)](
        da.contiguous(), db.contiguous(), direction.contiguous(), doq, dok,
        n, h, d, da.numel(), 256)
    return doq, dok


@triton.jit
def _merge(A, B, IQ, IK, SQ, Direction, DQ, DK, DSQ,
           N: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
           TOTAL: tl.constexpr, SQ_TOTAL: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
    h = i // D % H
    b = i // (D*H*N)
    choose = tl.load(Direction+b*H+h, i < TOTAL, other=0)
    # Match two BF16 add operands, FP32 addition, then BF16 output rounding.
    da = tl.load(A+i, i < TOTAL, other=0).to(tl.bfloat16).to(tl.float32)
    db = tl.load(B+i, i < TOTAL, other=0).to(tl.bfloat16).to(tl.float32)
    iq = tl.load(IQ+i, i < TOTAL, other=0).to(tl.bfloat16).to(tl.float32)
    ik = tl.load(IK+i, i < TOTAL, other=0).to(tl.bfloat16).to(tl.float32)
    tl.store(DQ+i, iq + tl.where(choose, da, 0.), i < TOTAL)
    tl.store(DK+i, ik + tl.where(choose, 0., db), i < TOTAL)
    dsq = tl.load(SQ+i, i < SQ_TOTAL, other=0)
    tl.store(DSQ+i, dsq, i < SQ_TOTAL)


def merge_token_gradients(da, db, iq, ik, dsq, direction):
    b, n, h, d = da.shape
    dq = torch.empty(da.shape, dtype=torch.bfloat16, device=da.device)
    dk = torch.empty_like(dq)
    sq = torch.empty(dsq.shape, dtype=torch.bfloat16, device=dsq.device)
    if not da.numel():
        return dq, dk, sq
    _merge[(triton.cdiv(max(da.numel(), dsq.numel()), 256),)](
        da.contiguous(), db.contiguous(), iq.contiguous(), ik.contiguous(),
        dsq.contiguous(), direction.contiguous(), dq, dk, sq,
        n, h, d, da.numel(), dsq.numel(), 256)
    return dq, dk, sq


@triton.jit
def _vocab_cast(Q, K, OQ, OK, TOTAL: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
    q = tl.load(Q+i, i < TOTAL, other=0).to(tl.bfloat16).to(tl.float32)
    k = tl.load(K+i, i < TOTAL, other=0).to(tl.bfloat16).to(tl.float32)
    tl.store(OQ+i, q, i < TOTAL)
    tl.store(OK+i, k, i < TOTAL)


def cast_vocabulary_gradients(dq, dk, q, k):
    # Shared-vocabulary reduction remains FP32 and retains Torch's reduction
    # order; it is uncommon in the current independent-per-head LM setting.
    oq = torch.empty(dq.shape, dtype=q.dtype, device=dq.device)
    ok = torch.empty(dk.shape, dtype=k.dtype, device=dk.device)
    _vocab_cast[(triton.cdiv(dq.numel(), 256),)](
        dq.contiguous(), dk.contiguous(), oq, ok, dq.numel(), 256)
    return (oq.sum(0) if q.ndim == 2 else oq,
            ok.sum(0) if k.ndim == 2 else ok)
