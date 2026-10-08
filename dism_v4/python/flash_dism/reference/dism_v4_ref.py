import torch
from torch import Tensor

def _float(x):
    return x.float() if x.dtype in (torch.float16, torch.bfloat16) else x

def dism_ref(q_vec: Tensor, k_vec: Tensor, sq_vec: Tensor, sk_vec: Tensor, q_lse: Tensor, k_lse: Tensor, idx_q: Tensor, idx_k: Tensor, direction: Tensor, hard: Tensor, delta: Tensor, v: Tensor, rtau: Tensor):
    """
    q_vec, k_vec: [B, N, H, C]
    q_lse, k_lse: [B, N, H]
    sq_vec, sk_vec: [B, N, H, R]
    v: [B, N, H, D]
    delta: [B, H, N], nonnegative attenuation = -log(g).
        0 preserves recurrence, +inf resets; use softplus(-gate_logits).
    sq_vec/sk_vec already include the caller's SiLU; do not activate twice.
    Accumulate BF16/FP16 inputs in FP32; preserve FP64 for gradcheck.
    direction: [B, H], true for k-to-qemb
    hard: [B, H, N]
    rtau: [H]
    idx_q, idx_k: [B, H, N]
    """
    if q_vec.ndim != 4 or q_vec.shape != k_vec.shape:
        raise ValueError('q_vec/k_vec must have equal [B,N,H,C] shapes')
    B, N, H, C = q_vec.shape
    if N == 0: raise ValueError('Empty sequences are not supported')
    if direction.shape != (B,H) or direction.dtype != torch.bool:
        raise ValueError('direction must be bool [B,H]')
    if hard.shape != (B,H,N) or hard.dtype != torch.bool:
        raise ValueError('hard must be bool [B,H,N]')
    if delta.shape != (B,H,N) or idx_q.shape != (B,H,N) or idx_k.shape != (B,H,N):
        raise ValueError('delta and indices must have shape [B,H,N]')
    if q_lse.shape != (B,N,H) or k_lse.shape != (B,N,H) or rtau.shape != (H,):
        raise ValueError('LSE must be [B,N,H], rtau [H]')
    if sq_vec.ndim != 4 or sq_vec.shape != sk_vec.shape or sq_vec.shape[:3] != (B,N,H):
        raise ValueError('Readout features must be equally shaped [B,N,H,R]')
    if v.ndim != 4 or v.shape[:3] != (B,N,H): raise ValueError('v must be [B,N,H,D]')
    q_vec,k_vec,sq_vec,sk_vec,q_lse,k_lse,delta,v,rtau = map(
        _float,(q_vec,k_vec,sq_vec,sk_vec,q_lse,k_lse,delta,v,rtau))
    raw_scores = torch.einsum('bnhc,bmhc->bhnm', q_vec, k_vec)
    # True: q @ q_from_k minus query LSE. False: k_from_q @ k minus key LSE.
    norm_score = raw_scores - torch.where(direction[..., None, None],
        q_lse.transpose(1, 2)[..., None], k_lse.transpose(1, 2)[..., None, :])
    
    hard_score = torch.zeros_like(raw_scores).masked_fill(
        idx_q[..., None] != idx_k[..., None, :], float('-inf'))
    mix_score = torch.where(hard[..., None], hard_score, norm_score)
    logM = mix_score + rtau[None, :, None, None] # [B, H, N, M]

    B, N, H, C = q_vec.shape
    rows: list[Tensor] = [torch.full_like(logM[:, :, 0:1, :], float('-inf'))]
    for i in range(N):
        shifted = torch.nn.functional.pad(rows[-1][..., :-1], (1, 0), value=float("-inf"))
        rows.append(logM[:, :, i:i+1, :] + torch.nn.functional.softplus(shifted - delta[:, :, i:i+1, None]))

    scores = torch.cat(rows[1:], dim=-2)
    causal = torch.ones((N, N), dtype=torch.bool, device=logM.device).tril()
    causal_cores = scores.masked_fill(~causal, float("-inf"))

    zmax = causal_cores.amax(dim=-1, keepdim=True).clamp_min(0.0).detach() # [B, H, N, 1]
    weight = torch.exp(causal_cores - zmax)
    fallback_weight = torch.exp(-zmax)

    s_weight = torch.einsum('bnhc,bmhc->bhnm', sq_vec, sk_vec)
    numerator = torch.einsum('bhnm,bmhc->bhnc', weight * s_weight, v)

    # Intentional numerator-only signed readout. SiLU is not nonnegative.
    # Denominator excludes s_weight; output need not be a convex combination of V.
    denominator = weight.sum(dim=-1, keepdim=True) + fallback_weight
    return (numerator / denominator).transpose(-2, -3)

@torch.no_grad()
def dism_ref_backward(q_vec: Tensor, k_vec: Tensor, sq_vec: Tensor, sk_vec: Tensor,
                      q_lse: Tensor, k_lse: Tensor, idx_q: Tensor, idx_k: Tensor,
                      direction: Tensor, hard: Tensor, delta: Tensor, v: Tensor,
                      rtau: Tensor, grad_out: Tensor):
    """Explicit first-order backward, with the SAME inputs/layouts as dism_ref.

    Returns a dict keyed by the nine differentiable input names. Gradients of
    FP16/BF16 inputs are intentionally returned in FP32 (cast at caller boundary
    if needed); FP64 is preserved. No autograd/backward calls or graph creation.
    sq/sk gradients are w.r.t. already-activated features, not pre-SiLU logits.
    q_lse/k_lse are independent inputs here: no embedding interpolation backward.
    Dense diagnostic implementation: O(B H N^2) storage, recomputes forward states.
    """
    # Validate via forward and obtain O for the denominator correction.
    out = dism_ref(q_vec,k_vec,sq_vec,sk_vec,q_lse,k_lse,idx_q,idx_k,
                   direction,hard,delta,v,rtau)
    if grad_out.shape != out.shape or grad_out.device != out.device:
        raise ValueError('grad_out must have output shape [B,N,H,D] and device')
    q_vec,k_vec,sq_vec,sk_vec,q_lse,k_lse,delta,v,rtau = map(
        _float,(q_vec,k_vec,sq_vec,sk_vec,q_lse,k_lse,delta,v,rtau))
    grad_out = _float(grad_out).to(out.dtype)
    B,N,H,_ = q_vec.shape
    raw = torch.einsum('bnhc,bmhc->bhnm',q_vec,k_vec)
    norm = raw - torch.where(direction[:,:,None,None],
        q_lse.transpose(1,2).unsqueeze(-1),k_lse.transpose(1,2).unsqueeze(-2))
    hs = torch.zeros_like(raw).masked_fill(idx_q.unsqueeze(-1)!=idx_k.unsqueeze(-2),-torch.inf)
    logm = torch.where(hard.unsqueeze(-1),hs,norm)+rtau[None,:,None,None]
    previous = torch.full_like(logm[:,:,0],-torch.inf)
    rows,alphas = [],[]
    for i in range(N):
        x = torch.nn.functional.pad(previous[...,:-1],(1,0),value=-torch.inf)-delta[:,:,i,None]
        # Match torch softplus's linear branch (default threshold=20) exactly.
        alphas.append(torch.where(x>20,torch.ones_like(x),torch.sigmoid(x)))
        previous = logm[:,:,i]+torch.nn.functional.softplus(x)
        rows.append(previous)
    w = torch.stack(rows,dim=-2)
    alpha = torch.stack(alphas,dim=-2)
    causal = torch.ones(N,N,device=w.device,dtype=torch.bool).tril()
    w = w.masked_fill(~causal,-torch.inf)
    maximum = w.amax(-1,keepdim=True).clamp_min(0.)
    ew = (w-maximum).exp()
    p = ew/(ew.sum(-1,keepdim=True)+(-maximum).exp())
    s = torch.einsum('bnhr,bmhr->bhnm',sq_vec,sk_vec)
    # E_ij = <dO_i,V_j>; dS=P*E. S is NOT in the denominator.
    e = torch.einsum('bnhd,bmhd->bhnm',grad_out,v)
    ds = p*e
    dv = torch.einsum('bhnm,bnhd->bmhd',p*s,grad_out)
    dsq = torch.einsum('bhnm,bmhr->bnhr',ds,sk_vec)
    dsk = torch.einsum('bhnm,bnhr->bmhr',ds,sq_vec)
    correction = (grad_out*out).sum(-1).transpose(1,2).unsqueeze(-1)
    # Local dW from readout, followed by reverse diagonal recurrence.
    total = p*(s*e-correction)
    ddelta = torch.zeros_like(delta)
    for i in range(N-1,-1,-1):
        propagated = total[:,:,i]*alpha[:,:,i]
        ddelta[:,:,i] = -propagated.sum(-1)
        if i:
            total[:,:,i-1,:-1] += propagated[...,1:]
    # total=dlogM after reverse scan. Hard rows still propagate and train tau/delta.
    dtau = total.sum((0,2,3))
    dsoft = total.masked_fill(hard.unsqueeze(-1),0.)
    dq = torch.einsum('bhnm,bmhc->bnhc',dsoft,k_vec)
    dk = torch.einsum('bhnm,bnhc->bmhc',dsoft,q_vec)
    dlq = torch.where(direction[:,:,None],-dsoft.sum(-1),0.).transpose(1,2)
    dlk = torch.where(direction[:,:,None],0.,-dsoft.sum(-2)).transpose(1,2)
    return dict(q_vec=dq,k_vec=dk,sq_vec=dsq,sk_vec=dsk,q_lse=dlq,k_lse=dlk,
                delta=ddelta,v=dv,rtau=dtau)


def dism_wrapper(q: Tensor, k: Tensor, sq_vec: Tensor, sk_vec: Tensor, q_weight: Tensor, k_weight: Tensor, delta: Tensor, v: Tensor, rtau: Tensor, *, direction: Tensor | None = None, hard: Tensor | None = None, hard_prob: float = .5, generator: torch.Generator | None = None):
    """
    q, k: [B, N, H, D]
    sq_vec, sk_vec: [B, N, H, R] soft readout, already SiLU activated
    q_weight, k_weight: [V, D] embedding table
    v: [B, N, H, D] values
    rtau: [H, ] additive match score in natural-log units
    delta: [B, H, N] attenuation -log(g), NOT raw gate logits
    direction: optional bool [B,H]; True = q @ q_from_k - q_lse.
        Default draws one GLOBAL direction per call, matching v2 semantics.
    hard: optional bool [B,H,N] for deterministic replay; otherwise sample
        with scalar hard_prob. This torch oracle does not reproduce CUDA Philox indexing.
    """
    if q.ndim != 4 or q.shape != k.shape: raise ValueError('Expected equal [B,N,H,D] q/k')
    B, N, H, D = q.shape
    if q_weight.ndim != 2 or q_weight.shape != k_weight.shape or q_weight.shape[1] != D:
        raise ValueError('Expected equally shaped shared tables [V,D]')
    if not 0 <= hard_prob <= 1: raise ValueError('hard_prob must be in [0,1]')
    q,k,q_weight,k_weight = map(_float,(q,k,q_weight,k_weight))

    k_scores = torch.einsum('bnhd,vd->bnhv', k, k_weight)
    q_scores = torch.einsum('bnhd,vd->bnhv', q, q_weight)
    k_lse, idx_k = torch.logsumexp(k_scores, -1), torch.argmax(k_scores, -1).transpose(-1, -2)
    q_lse, idx_q = torch.logsumexp(q_scores, -1), torch.argmax(q_scores, -1).transpose(-1, -2)

    q_emb = torch.einsum('bnhv,vd->bnhd', torch.softmax(k_scores, -1), q_weight)
    k_emb = torch.einsum('bnhv,vd->bnhd', torch.softmax(q_scores, -1), k_weight)
    
    if direction is None:
        direction = torch.randint(0, 2, (), device=q.device, generator=generator).bool().expand(B,H)
    if direction.shape != (B,H) or direction.dtype != torch.bool:
        raise ValueError('direction must be bool [B,H]')
    if hard is None:
        hard = (torch.full((B,H,N), bool(hard_prob), device=q.device, dtype=torch.bool)
                if hard_prob in (0.,1.) else
                torch.rand((B,H,N),device=q.device,generator=generator) < hard_prob)

    q_vec = torch.where(direction[:, None, :, None], q, k_emb)
    k_vec = torch.where(direction[:, None, :, None], q_emb, k)

    o = dism_ref(q_vec, k_vec, sq_vec, sk_vec, q_lse, k_lse, idx_q, idx_k, direction, hard, delta, v, rtau)

    return o
