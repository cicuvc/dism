"""Quadratic hard-history blocks, with only one live boundary row per head."""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit(do_not_specialize=["START"])
def _history(Q, K, SEG, RESET, TAU, PREV, NEXT, LOGA, DLOGA,
             N: tl.constexpr, START, BQ: tl.constexpr, BD: tl.constexpr):
    g, tile = tl.program_id(0), tl.program_id(1)
    r = tl.arange(0, BQ)
    diagonal = tile * BD + tl.arange(0, BD)
    qi = START + r
    kj = qi[:, None] - diagonal[None, :]
    incoming_j = START - 1 - diagonal
    incoming = tl.load(PREV + g*N + incoming_j, incoming_j >= 0, 0)
    qprev = tl.load(Q + g*N + qi - 1, (qi > 0) & (qi <= N), -1)
    kprev = tl.load(K + g*N + kj - 1, (kj > 0) & (kj <= N), -2)
    qs = tl.load(SEG + g*N + qi, qi < N, -3)
    qps = tl.load(SEG + g*N + qi - 1, (qi > 0) & (qi <= N), -4)
    ks = tl.load(SEG + g*N + kj, (kj >= 0) & (kj < N), -5)
    kps = tl.load(SEG + g*N + kj - 1, (kj > 0) & (kj <= N), -6)
    reset = tl.load(RESET + g*N + qi, qi < N, 0)
    broken = ((r[:, None] > 0) & (qprev[:, None] != kprev)) | (reset[:, None] != 0)
    broken = broken | (qs[:, None] != qps[:, None]) | (ks != kps)
    last_bad = tl.associative_scan(tl.where(broken, r[:, None], -1), 0, _maximum)
    length = tl.where(last_bad >= 0, r[:, None] - last_bad, incoming[None, :] + r[:, None])
    terms = (length + 1).to(tl.float32)
    tau = tl.load(TAU + g)
    a = tl.abs(tau)
    # Geometric sum in log space. A short series avoids cancellation at tau=0.
    la = terms * a
    loggeom = tl.log(-libdevice.expm1(-la)) - tl.log(-libdevice.expm1(-a))
    logweight = tl.where(tau > 0, terms*tau, tau) + loggeom
    recip_a = tl.exp(-a) / (-libdevice.expm1(-a))
    recip_la = tl.exp(-la) / (-libdevice.expm1(-la))
    deriv = tl.where(tau > 0, terms + terms*recip_la - recip_a,
                     1. - terms*recip_la + recip_a)
    l2 = terms*terms
    series = tl.log(terms) + tau*(terms+1.)/2. + tau*tau*(l2-1.)/24.
    series -= tau*tau*tau*tau*(l2*l2-1.)/2880.
    dseries = (terms+1.)/2. + tau*(l2-1.)/12. - tau*tau*tau*(l2*l2-1.)/720.
    logweight = tl.where(la < .01, series, logweight)
    deriv = tl.where(la < .01, dseries, deriv)
    valid = (qi[:, None] < N) & (kj >= 0) & (kj < N) & (qs[:, None] == ks)
    address = (g*BQ + r[:, None])*N + kj
    tl.store(LOGA + address, tl.where(valid, logweight, -float('inf')),
             (qi[:, None] < N) & (kj >= 0) & (kj < N))
    tl.store(DLOGA + address, tl.where(valid, deriv, 0.),
             (qi[:, None] < N) & (kj >= 0) & (kj < N))
    last = tl.minimum(BQ, N - START) - 1
    previous_length = tl.sum(tl.where(r[:, None] == last, length, 0), 0)
    jlast = START + last - diagonal
    qlast = tl.load(Q + g*N + START + last)
    klast = tl.load(K + g*N + jlast, (jlast >= 0) & (jlast < N), -1)
    seglast = tl.load(SEG + g*N + START + last)
    kseg = tl.load(SEG + g*N + jlast, (jlast >= 0) & (jlast < N), -1)
    new_length = tl.where((qlast == klast) & (seglast == kseg), previous_length + 1, 0)
    tl.store(NEXT + g*N + jlast, new_length, (jlast >= 0) & (jlast < N))


@triton.jit
def _maximum(a, b):
    return tl.maximum(a, b)


def history_blocks(iq, ik, tau, segments, reset, block_size):
    """Yield log A and its exact tau log-derivative, regenerating hard history."""
    groups, n = iq.shape
    previous = torch.zeros_like(iq)
    for start in range(0, n, block_size):
        loga = torch.full((groups, block_size, n), -torch.inf, device=iq.device)
        derivative = torch.zeros_like(loga)
        following = torch.zeros_like(iq)
        _history[(groups, triton.cdiv(n, 32))](
            iq, ik, segments, reset, tau, previous, following, loga, derivative,
            n, start, block_size, 32, num_warps=8,
        )
        previous = following
        count = min(block_size, n - start)
        yield start, loga[:, :count].contiguous(), derivative[:, :count].contiguous()
