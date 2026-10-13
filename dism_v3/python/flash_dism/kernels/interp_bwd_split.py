"""Experimental FA2-style interpolation backward; production dispatch unchanged.

Two attentions: (q, E_q, E_k) and (k, E_k, E_q). Row-owned dQ and
vocabulary-owned dK/dV are separate launches. Vocabulary gradients optionally
split the token reduction into independent FP32 partials, then reduce without
atomics, or accumulate directly with global FP32 atomics. The extra dLSE term
and saved LSE-minus-tau contract are preserved.
"""
import torch
import triton
import triton.language as tl
import cu_flash_dism as cu

from flash_dism.emb_kernel import _interp_bwd_preprocess


@triton.jit
def _dq(X, Key, Value, DO, LSE, DLSE, Delta, DX, Tau,
        B: tl.constexpr, N: tl.constexpr, H: tl.constexpr, V: tl.constexpr,
        D: tl.constexpr, beta, HAS_TAU: tl.constexpr,
        xb: tl.constexpr, xn: tl.constexpr, xh: tl.constexpr,
        kh: tl.constexpr, kv: tl.constexpr, vh: tl.constexpr, vv: tl.constexpr,
        ob: tl.constexpr, on: tl.constexpr, oh: tl.constexpr,
        M: tl.constexpr, K: tl.constexpr):
    row = tl.program_id(0) * M + tl.arange(0, M)
    bh = tl.program_id(1)
    batch, head = bh // H, bh % H
    d = tl.arange(0, D)
    x = tl.load(X + batch * xb + row[:, None] * xn + head * xh + d[None, :],
                row[:, None] < N, 0)
    do = tl.load(DO + batch * ob + row[:, None] * on + head * oh + d[None, :],
                 row[:, None] < N, 0).to(Key.dtype.element_ty)
    offset = bh * N + row
    lse = tl.load(LSE + offset, row < N, 0)
    if HAS_TAU:
        lse += tl.load(Tau + head)
    lse *= 1.4426950408889634
    dlse = tl.load(DLSE + offset, row < N, 0)
    delta = tl.load(Delta + offset, row < N, 0)
    acc = tl.zeros((M, D), tl.float32)
    for start in range(tl.cdiv(V, K)):
        col = start * K + tl.arange(0, K)
        key = tl.load(Key + head * kh + col[:, None] * kv + d[None, :], col[:, None] < V, 0)
        value = tl.load(Value + head * vh + col[:, None] * vv + d[None, :], col[:, None] < V, 0)
        score = tl.dot(x, tl.trans(key))
        score = tl.where((row[:, None] < N) & (col[None, :] < V), score, float('-inf'))
        p = tl.exp2(score * (beta * 1.4426950408889634) - lse[:, None])
        dp = tl.dot(do, tl.trans(value))
        ds = p * (dp - delta[:, None]) + p * dlse[:, None]
        acc += tl.dot(ds.to(Key.dtype.element_ty), key) * beta
    tl.store(DX + ((batch * N + row[:, None]) * H + head) * D + d[None, :], acc, row[:, None] < N)


@triton.jit
def _dkv(X, Key, Value, DO, LSE, DLSE, Delta, DK, DV, Tau,
         B: tl.constexpr, N: tl.constexpr, H: tl.constexpr, V: tl.constexpr,
         D: tl.constexpr, beta, HAS_TAU: tl.constexpr,
         xb: tl.constexpr, xn: tl.constexpr, xh: tl.constexpr,
         kh: tl.constexpr, kv: tl.constexpr, vh: tl.constexpr, vv: tl.constexpr,
         ob: tl.constexpr, on: tl.constexpr, oh: tl.constexpr,
         M: tl.constexpr, K: tl.constexpr, SPLITS: tl.constexpr, ATOMIC: tl.constexpr):
    head = tl.program_id(1)
    split = tl.program_id(2)
    col = tl.program_id(0) * K + tl.arange(0, K)
    d = tl.arange(0, D)
    key = tl.load(Key + head * kh + col[:, None] * kv + d[None, :], col[:, None] < V, 0)
    value = tl.load(Value + head * vh + col[:, None] * vv + d[None, :], col[:, None] < V, 0)
    key_grad = tl.zeros((K, D), tl.float32)
    value_grad = tl.zeros((K, D), tl.float32)
    # Split in tiles; masks handle incomplete token/vocabulary tiles and B>1.
    TILES: tl.constexpr = tl.cdiv(B * N, M * SPLITS)
    for tile in range(TILES):
        row = (split * TILES + tile) * M + tl.arange(0, M)
        batch, token = row // N, row % N
        valid = row < B * N
        x = tl.load(X + batch[:, None] * xb + token[:, None] * xn + head * xh + d[None, :],
                    valid[:, None], 0)
        do = tl.load(DO + batch[:, None] * ob + token[:, None] * on + head * oh + d[None, :],
                     valid[:, None], 0).to(Key.dtype.element_ty)
        offset = (batch * H + head) * N + token
        lse = tl.load(LSE + offset, valid, 0)
        if HAS_TAU:
            lse += tl.load(Tau + head)
        lse *= 1.4426950408889634
        dlse = tl.load(DLSE + offset, valid, 0)
        delta = tl.load(Delta + offset, valid, 0)
        score = tl.dot(x, tl.trans(key))
        score = tl.where(valid[:, None] & (col[None, :] < V), score, float('-inf'))
        p = tl.exp2(score * (beta * 1.4426950408889634) - lse[:, None])
        dp = tl.dot(do, tl.trans(value))
        ds = p * (dp - delta[:, None]) + p * dlse[:, None]
        key_grad += tl.dot(tl.trans(ds).to(Key.dtype.element_ty), x) * beta
        value_grad += tl.dot(tl.trans(p).to(Key.dtype.element_ty), do)
    if ATOMIC:
        offset = (head * V + col[:, None]) * D + d[None, :]
        # Global FP32 atomics only; zeroed outputs, no shared atomics.
        tl.atomic_add(DK + offset, key_grad, col[:, None] < V, sem='relaxed')
        tl.atomic_add(DV + offset, value_grad, col[:, None] < V, sem='relaxed')
    else:
        offset = ((split * H + head) * V + col[:, None]) * D + d[None, :]
        tl.store(DK + offset, key_grad, col[:, None] < V)
        tl.store(DV + offset, value_grad, col[:, None] < V)


@triton.jit
def _table_grad(X, OtherX, Key, Value, DO, OtherDO, LSE, OtherLSE, DLSE, Delta,
                Grad, Tau, B: tl.constexpr, N: tl.constexpr, H: tl.constexpr,
                V: tl.constexpr, D: tl.constexpr, beta, HAS_TAU: tl.constexpr,
                XS: tl.constexpr, OS: tl.constexpr, KS: tl.constexpr, VS: tl.constexpr,
                M: tl.constexpr, K: tl.constexpr):
    """One table's key-role and opposite attention's value-role, no partials."""
    head=tl.program_id(1)
    col=tl.program_id(0)*K+tl.arange(0,K)
    d=tl.arange(0,D)
    key=tl.load(Key+head*KS[0]+col[:,None]*KS[1]+d[None,:],col[:,None]<V,0)
    value=tl.load(Value+head*VS[0]+col[:,None]*VS[1]+d[None,:],col[:,None]<V,0)
    acc=tl.zeros((K,D),tl.float32)
    tau=tl.load(Tau+head) if HAS_TAU else 0.
    # Key-role contribution. Accumulator stays in registers across both scans.
    for tile in range(tl.cdiv(B*N,M)):
        row=tile*M+tl.arange(0,M)
        batch,token=row//N,row%N
        valid=row<B*N
        x=tl.load(X+batch[:,None]*XS[0]+token[:,None]*XS[1]+head*XS[2]+d[None,:],valid[:,None],0)
        do=tl.load(DO+(row[:,None]*H+head)*D+d[None,:],valid[:,None],0).to(Key.dtype.element_ty)
        offset=(batch*H+head)*N+token
        lse=(tl.load(LSE+offset,valid,0)+tau)*1.4426950408889634
        dlse=tl.load(DLSE+offset,valid,0)
        delta=tl.load(Delta+offset,valid,0)
        score=tl.dot(x,tl.trans(key))
        score=tl.where(valid[:,None] & (col[None,:]<V),score,float('-inf'))
        p=tl.exp2(score*(beta*1.4426950408889634)-lse[:,None])
        dp=tl.dot(do,tl.trans(value))
        ds=p*(dp-delta[:,None])+p*dlse[:,None]
        acc+=tl.dot(tl.trans(ds).to(Key.dtype.element_ty),x)*beta
    # Value-role contribution from the other attention. Its key is Value.
    # Recompute its probabilities instead of communicating a second gradient.
    for tile in range(tl.cdiv(B*N,M)):
        row=tile*M+tl.arange(0,M)
        batch,token=row//N,row%N
        valid=row<B*N
        x=tl.load(OtherX+batch[:,None]*OS[0]+token[:,None]*OS[1]+head*OS[2]+d[None,:],valid[:,None],0)
        do=tl.load(OtherDO+(row[:,None]*H+head)*D+d[None,:],valid[:,None],0).to(Key.dtype.element_ty)
        lse=(tl.load(OtherLSE+(batch*H+head)*N+token,valid,0)+tau)*1.4426950408889634
        score=tl.dot(x,tl.trans(value))
        score=tl.where(valid[:,None] & (col[None,:]<V),score,float('-inf'))
        p=tl.exp2(score*(beta*1.4426950408889634)-lse[:,None])
        acc+=tl.dot(tl.trans(p).to(Key.dtype.element_ty),do)
    tl.store(Grad+(head*V+col[:,None])*D+d[None,:],acc,col[:,None]<V)


@triton.jit
def _tables(Q,K,EQ,EK,DOQ,DOK,LQ,LK,DLQ,DLK,DeltaQ,DeltaK,DEQ,DEK,Tau,
            B: tl.constexpr,N: tl.constexpr,H: tl.constexpr,V: tl.constexpr,D: tl.constexpr,
            beta,HAS_TAU: tl.constexpr,QS: tl.constexpr,KS: tl.constexpr,
            EQS: tl.constexpr,EKS: tl.constexpr,M: tl.constexpr,BLOCK_V: tl.constexpr):
    # Both codebooks in one grid; avoid two underfilled sequential launches.
    if tl.program_id(2)==0:
        _table_grad(Q,K,EQ,EK,DOK,DOQ,LK,LQ,DLK,DeltaK,DEQ,Tau,
                    B,N,H,V,D,beta,HAS_TAU,QS,KS,EQS,EKS,M,BLOCK_V)
    else:
        _table_grad(K,Q,EK,EQ,DOQ,DOK,LQ,LK,DLQ,DeltaQ,DEK,Tau,
                    B,N,H,V,D,beta,HAS_TAU,KS,QS,EKS,EQS,M,BLOCK_V)


@triton.jit
def _reduce(QKey, QValue, KKey, KValue, DEQ, DEK,
            SIZE: tl.constexpr, SPLITS: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    eq = tl.full((BLOCK,), 0., tl.float32)
    ek = tl.full((BLOCK,), 0., tl.float32)
    for split in range(SPLITS):
        # E_q is the q-attention key and the k-attention value (and vice versa).
        eq += tl.load(QKey + split * SIZE + offset, offset < SIZE, 0)
        eq += tl.load(KValue + split * SIZE + offset, offset < SIZE, 0)
        ek += tl.load(KKey + split * SIZE + offset, offset < SIZE, 0)
        ek += tl.load(QValue + split * SIZE + offset, offset < SIZE, 0)
    tl.store(DEQ + offset, eq, offset < SIZE)
    tl.store(DEK + offset, ek, offset < SIZE)


def emb_bwd_split(q, k, q_emb, k_emb, out_q, out_k, olse_q, olse_k,
                  doq, dok, dlq, dlk, sm_scale=1., tau=None, *, splits=16,
                  dq_m=64, dq_v=64, dkv_m=64, dkv_v=32, num_warps=4, num_stages=3,
                  reduction='partial'):
    """Same inputs/outputs as emb_bwd_wrapper; opt-in experimental backend.

    FP32 reduction order differs for codebooks. No changed production tolerances,
    forward rounding, hard gradients, beta scaling, or tau gradient ownership.
    Atomic reduction is nondeterministic; partial reduction uses a fixed order.
    """
    cu.validate_interpolation(q, k, q_emb, k_emb, tau)
    b, n, h, d = q.shape
    vocab = q_emb.shape[1]
    if splits < 1:
        raise ValueError('splits must be positive')
    if reduction not in ('partial','atomic','register'):
        raise ValueError('reduction must be partial, atomic or register')
    if reduction == 'register' and splits != 1:
        raise ValueError('register reduction is unsplit; set splits=1')
    if dlq is None:
        dlq = torch.zeros((b, h, n), device=q.device, dtype=torch.float32)
    if dlk is None:
        dlk = torch.zeros((b, h, n), device=q.device, dtype=torch.float32)
    cu.validate_interpolation_grads(out_q, out_k, olse_q, olse_k, doq, dok, dlq, dlk)
    doq, dok, dlq, dlk = [t.contiguous() for t in (doq, dok, dlq, dlk)]
    olse_q, olse_k = olse_q.contiguous(), olse_k.contiguous()
    dq, dk = [torch.empty(t.shape, device=t.device, dtype=torch.float32) for t in (q, k)]
    deq, dek = [torch.empty(t.shape, device=t.device, dtype=torch.float32) for t in (q_emb, k_emb)]
    delta_q, delta_k = [torch.empty((b, h, n), device=q.device, dtype=torch.float32) for _ in range(2)]
    _interp_bwd_preprocess[(triton.cdiv(n, 64), b * h)](
        out_q, out_k, doq, dok, delta_q, delta_k, b, h, n,
        out_q.stride(0), out_q.stride(2), out_q.stride(1),
        doq.stride(0), doq.stride(2), doq.stride(1),
        dok.stride(0), dok.stride(2), dok.stride(1), h*n, n, d, 64)
    if reduction in ('atomic','register'):
        if reduction == 'atomic':
            deq.zero_()
            dek.zero_()
        partials = [deq,dek,dek,deq]
    else:
        partials = [torch.empty((splits, h, vocab, d), device=q.device, dtype=torch.float32) for _ in range(4)]
    for x, key, value, do, lse, dlse, delta, dx, dkey, dvalue in (
            (q, q_emb, k_emb, dok, olse_k, dlk, delta_k, dq, partials[0], partials[1]),
            (k, k_emb, q_emb, doq, olse_q, dlq, delta_q, dk, partials[2], partials[3])):
        common = (b, n, h, vocab, d, float(sm_scale), tau is not None,
                  x.stride(0), x.stride(1), x.stride(2),
                  key.stride(0), key.stride(1), value.stride(0), value.stride(1),
                  do.stride(0), do.stride(1), do.stride(2))
        _dq[(triton.cdiv(n, dq_m), b*h)](x, key, value, do, lse, dlse, delta, dx, tau,
                                       *common, dq_m, dq_v, num_warps=num_warps, num_stages=num_stages)
        if reduction != 'register':
            _dkv[(triton.cdiv(vocab, dkv_v), h, splits)](
                x, key, value, do, lse, dlse, delta, dkey, dvalue, tau,
                *common, dkv_m, dkv_v, splits, reduction == 'atomic', num_warps=num_warps, num_stages=num_stages)
    if reduction == 'partial':
        _reduce[(triton.cdiv(h*vocab*d, 256),)](*partials, deq, dek, h*vocab*d, splits, 256)
    elif reduction == 'register':
        _tables[(triton.cdiv(vocab,dkv_v),h,2)](
            q,k,q_emb,k_emb,doq,dok,olse_q,olse_k,dlq,dlk,delta_q,delta_k,deq,dek,tau,
            b,n,h,vocab,d,float(sm_scale),tau is not None,
            q.stride()[:3],k.stride()[:3],q_emb.stride()[:2],k_emb.stride()[:2],
            dkv_m,dkv_v,num_warps=num_warps,num_stages=num_stages)
    return dq, dk, deq, dek
