import torch
import wandb
import math
import os
import random
import sys
from fla.modules import FusedRMSNormGated, ShortConvolution
from fla.layers.gated_deltanet import GatedDeltaNet
import torch.nn as nn
from flash_attn import flash_attn_func

from emb_kernel import EmbInterpFunction


def logcosh_approx(delta):
    return torch.relu(delta.abs() * 0.5 - math.log(2.0))


def _voc_interp(x: torch.Tensor, key_voc: torch.Tensor, val_voc: torch.Tensor):
    """
        x: [B, H, N, D] (queries)
        key_voc, val_voc: [H, V, D]
        returns out [B, H, N, D], lse [B, H, N]
    """
    B, H, N, D = x.shape
    V = key_voc.shape[1]
    DV = val_voc.shape[-1]
    if DV != D:
        assert DV < D
        val_voc = torch.nn.functional.pad(val_voc, (0, D - DV))
    kh = key_voc.transpose(0, 1).unsqueeze(0).expand(B, V, H, D).contiguous().to(x.dtype)
    vh = val_voc.transpose(0, 1).unsqueeze(0).expand(B, V, H, D).contiguous().to(x.dtype)
    out, lse, _ = flash_attn_func(x.transpose(1, 2).contiguous(), kh, vh, softmax_scale=1.0, return_attn_probs=True)
    return out.transpose(1, 2)[..., :DV], lse


def _dism_from_logm(logM: torch.Tensor, v: torch.Tensor):
    """Apply the causal softplus recurrence and zero-value fallback."""
    N = logM.shape[-1]
    weights = [logM[:, :, 0, None, :]]
    mask = torch.arange(0, N, device=logM.device) == 0
    umask = torch.tril(torch.ones((N, N), dtype=torch.bool, device=logM.device))

    for row in range(1, N):
        last_row = torch.roll(weights[-1], 1)
        last_row = torch.where(mask[None, None, None, :], -1e7, last_row)
        weights.append(torch.nn.functional.softplus(last_row) + logM[:, :, row, None, :])

    scores = torch.where(
        umask[None, None], torch.cat(weights, dim=-2), float('-inf')
    )
    zmax = torch.clamp_min(scores.max(-1, keepdim=True).values, 0.0).detach()
    attn = torch.exp(scores - zmax)
    fallback = torch.exp(-zmax)
    denominator = attn.sum(-1, keepdim=True) + fallback
    out = torch.matmul(attn.to(v.dtype), v) / denominator
    return out, attn / denominator


def voc_dism(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, rtau: torch.Tensor, q_voc: torch.Tensor, k_voc: torch.Tensor, hard: bool = False, lmb: float = 0.5, mix: str = "logaddexp", gen: torch.Generator = None, sketch: torch.Tensor = None, return_aux: bool = False, sm_scale: float = 1.0) -> torch.Tensor:
    """
        q, k: [B, H, N, D]
        q_voc, k_voc: [H, V, D]
        v: [B, H, N, C]
        rtau: [H, ]
    """
    B, H, N, D = q.shape

    
    # train-only path
    if hard:
        qq = torch.einsum('bhnd,hvd->bhnv', q, q_voc)
        kk = torch.einsum('bhnd,hvd->bhnv', k, k_voc)
        
        idx_q = torch.argmax(qq, -1)
        idx_k = torch.argmax(kk, -1)
        
        # inference path
        logM = rtau[None,:,None,None] + torch.where(idx_q[:,:,:,None] == idx_k[:,:,None,:], 0, -9999.0)

    else:
        score_mix = "random" if mix in (
            "confidence", "tail_dropout", "stratified", "stratified_hybrid",
            "random_logm_st", "random_leaky_vjp", "hard_leaky_vjp",
            "random_leaky_joint",
        ) else mix

        # emb kernel: q_embs = softmax(kk)@q_voc, k_embs = softmax(qq)@k_voc,
        #             k_lse = logsumexp(kk), q_lse = logsumexp(qq),
        #             ptop_k = top1(softmax(kk)), ptop_q = top1(softmax(qq)),
        #             idx_k = argmax(kk), idx_q = argmax(qq)
        q_embs, k_embs, k_lse, q_lse, ptop_k, ptop_q, idx_k, idx_q = EmbInterpFunction.apply(
            q, k, q_voc.to(q.dtype), k_voc.to(k.dtype), sm_scale)

        if mix == "tail_dropout":
            head_idx = torch.arange(H, device=q.device)[None, :, None]
            top_q_embs = q_voc[head_idx, idx_k.long()].to(q_embs.dtype)
            top_k_embs = k_voc[head_idx, idx_q.long()].to(k_embs.dtype)
            tail_mode = os.environ.get("TAIL_MODE", "bernoulli")
            tail_keep = float(os.environ.get("TAIL_KEEP", 0.5))
            rq = torch.rand(
                q_embs.shape[:-1] + (1,), device=q.device, generator=gen,
                dtype=torch.float32,
            )
            rk = torch.rand(
                k_embs.shape[:-1] + (1,), device=k.device, generator=gen,
                dtype=torch.float32,
            )
            if tail_mode == "bernoulli":
                rq = (rq < tail_keep).to(q_embs.dtype)
                rk = (rk < tail_keep).to(k_embs.dtype)
            elif tail_mode == "uniform":
                rq = rq.to(q_embs.dtype)
                rk = rk.to(k_embs.dtype)
            else:
                raise ValueError(tail_mode)
            q_embs = top_q_embs + rq * (q_embs - top_q_embs)
            k_embs = top_k_embs + rk * (k_embs - top_k_embs)

        logM = rtau[None,:,None,None] + torch.where(idx_q[:,:,:,None] == idx_k[:,:,None,:], 0, -9999.0)

        flip = int(torch.randint(0, 2, (1,), generator=gen, device=q.device).item()) if score_mix in ("random", "cumulant", "cumulant_sketch") else -1
        if score_mix not in ("cumulant", "cumulant_sketch") or flip == 0:
            if score_mix in ("logaddexp", "mean", "corrected", "min", "single_q", "random") or flip == 0:
                A = sm_scale * (q @ q_embs.transpose(-1, -2)) - q_lse[..., None]
        if score_mix not in ("cumulant", "cumulant_sketch") or flip == 1:
            if score_mix in ("logaddexp", "mean", "corrected", "min", "single_k", "random") or flip == 1:
                B = (sm_scale * (k @ k_embs.transpose(-1, -2)) - k_lse[..., None]).transpose(-1, -2)
        if score_mix == "logaddexp":
            s = torch.logaddexp(A, B) - math.log(2)
        elif score_mix == "mean":
            s = 0.5 * (A + B)
        elif score_mix == "corrected":
            s = 0.5 * (A + B) + logcosh_approx(A - B)
        elif score_mix == "min":
            s = torch.minimum(A, B)
        elif score_mix == "single_q":
            s = A
        elif score_mix == "single_k":
            s = B
        elif score_mix == "random":
            s = A if flip == 0 else B
        elif score_mix == "cumulant":
            # Prototype of log E[exp(X)] ~= E[X] + Var[X]/2.  We retain only
            # the diagonal of the embedding covariance, so the extra state is
            # O(D), independent of qk_vocab.  If useful, this moment belongs in
            # EmbInterpFunction's streaming kernel; _voc_interp is only used to
            # test the estimator before implementing its custom backward.
            if flip == 0:
                q2_embs, _ = _voc_interp(k, k_voc, q_voc.square())
                diag_var = (q2_embs.float() - q_embs.float().square()).clamp_min(0)
                var = q.float().square() @ diag_var.transpose(-1, -2)
                s = A.float() + float(os.environ.get("CUMULANT_COEF", 0.5)) * var
            else:
                k2_embs, _ = _voc_interp(q, q_voc, k_voc.square())
                diag_var = (k2_embs.float() - k_embs.float().square()).clamp_min(0)
                var = (k.float().square() @ diag_var.transpose(-1, -2)).transpose(-1, -2)
                s = B.float() + float(os.environ.get("CUMULANT_COEF", 0.5)) * var
            s = s.clamp_max(0)
        elif score_mix == "cumulant_sketch":
            assert sketch is not None
            coef = float(os.environ.get("CUMULANT_COEF", 0.5))
            if flip == 0:
                # z ~ N(0,I): E[((q.z)^2-||q||^2) Var(e.z)]/2
                # equals q^T Cov(e) q.  Averaging R sketches gives an O(RD)
                # estimator that retains off-diagonal covariance information.
                q_voc_z = torch.einsum('hvd,hdr->hvr', q_voc.float(), sketch)
                qz2_embs, _ = _voc_interp(k, k_voc, q_voc_z.square())
                mean_z = torch.einsum('bhnd,hdr->bhnr', q_embs.float(), sketch)
                var_z = (qz2_embs.float() - mean_z.square()).clamp_min(0)
                qz = torch.einsum('bhnd,hdr->bhnr', q.float(), sketch)
                centered_qz2 = qz.square() - q.float().square().sum(-1, keepdim=True)
                var = 0.5 * (centered_qz2 @ var_z.transpose(-1, -2)) / sketch.shape[-1]
                s = A.float() + coef * var.clamp_min(0)
            else:
                k_voc_z = torch.einsum('hvd,hdr->hvr', k_voc.float(), sketch)
                kz2_embs, _ = _voc_interp(q, q_voc, k_voc_z.square())
                mean_z = torch.einsum('bhnd,hdr->bhnr', k_embs.float(), sketch)
                var_z = (kz2_embs.float() - mean_z.square()).clamp_min(0)
                kz = torch.einsum('bhnd,hdr->bhnr', k.float(), sketch)
                centered_kz2 = kz.square() - k.float().square().sum(-1, keepdim=True)
                var = (0.5 * (centered_kz2 @ var_z.transpose(-1, -2)) / sketch.shape[-1]).transpose(-1, -2)
                s = B.float() + coef * var.clamp_min(0)
            s = s.clamp_max(0)
        else:
            raise ValueError(score_mix)
        logM_soft = rtau[None,:,None,None] + s
        drop = torch.rand(logM.shape[:-1]+(1,), generator=gen, device=logM.device, dtype=logM.dtype)
        if mix == "confidence":
            causal_count = torch.arange(1, N + 1, device=q.device, dtype=ptop_k.dtype)
            key_conf = ptop_k.cumsum(-1) / causal_count
            hard_prob = (ptop_q * key_conf).clamp_min(0).sqrt()[..., None]
        else:
            hard_prob = lmb
        use_soft = drop > hard_prob
        use_stratified = mix == "stratified" or (
            mix == "stratified_hybrid"
            and float(lmb) < float(os.environ.get("STRAT_UNTIL", 0.5))
        )
        if use_stratified:
            window = int(os.environ.get("SOFT_ANCHOR_WINDOW", 8))
            scope = os.environ.get("STRAT_SCOPE", "window")
            if scope == "window":
                assert N % window == 0, "window stratification requires N divisible by SOFT_ANCHOR_WINDOW"
                n_windows = N // window
                n_soft = max(1, round((1.0 - float(lmb)) * window))
                priority = torch.rand(
                    q.shape[0], q.shape[1], n_windows, window,
                    device=q.device, generator=gen,
                )
                rank = priority.argsort(dim=-1).argsort(dim=-1)
                use_soft = (rank < n_soft).reshape(q.shape[0], q.shape[1], N, 1)
            elif scope == "global":
                min_soft = math.ceil(N / window)
                n_soft = max(min_soft, round((1.0 - float(lmb)) * N))
                priority = torch.rand(
                    q.shape[0], q.shape[1], N, device=q.device, generator=gen,
                )
                rank = priority.argsort(dim=-1).argsort(dim=-1)
                use_soft = (rank < n_soft)[..., None]
            else:
                raise ValueError(scope)
        logM_hard = logM
        logM_mixed = torch.where(use_soft, logM_soft, logM_hard)
        if mix == "random_leaky_joint":
            eps = float(os.environ.get("LEAK_EPS", 0.03))
            if not 0.0 < eps <= 1.0:
                raise ValueError(f"LEAK_EPS must be in (0, 1], got {eps}")
            if eps == 1.0:
                leaky_hard = logM_soft
            else:
                hard_relative = logM_hard - rtau[None, :, None, None]
                leaky_relative = torch.logaddexp(
                    hard_relative + math.log1p(-eps),
                    s + math.log(eps),
                )
                leaky_hard = rtau[None, :, None, None] + leaky_relative
            # Unlike random_leaky_vjp, this is the sole training path.  The
            # leaky score changes the hard-row forward value and supplies the
            # q/k, temperature, and v gradients through the same recurrence.
            logM = torch.where(use_soft, logM_soft, leaky_hard)
        elif mix == "random_logm_st":
            # Row-random hard/soft forward, but use the selected direction's
            # soft logM as the surrogate in backward:
            #   forward:  logM_mixed
            #   d logM / d logM_soft = 1 for every row.
            # The downstream Jacobian is still evaluated at the mixed forward
            # state; this is a logM-level STE, not an output-level STE.
            logM = logM_soft + (logM_mixed - logM_soft).detach()
        else:
            logM = logM_mixed # the best one
        #logM = torch.where(torch.rand_like(logM) > lmb+0.5, logM_soft, logM_soft+(logM - logM_soft).detach())
        #logM = logM_soft + torch.where(torch.rand_like(logM) > lmb, 0, 1) * (logM - logM_soft).detach() # best
        #logM = logM_soft + lmb * (logM - logM_soft).detach() 

    # Diagnostic-only hook: publish the ACTUAL row decision before recurrence.
    observer = globals().get('_row_hard_observer')
    if observer is not None:
        observer(torch.ones((B,H,N),device=q.device,dtype=torch.bool)
                 if hard else (~use_soft).squeeze(-1))
    out, attn_prob = _dism_from_logm(logM, v)

    if not hard and mix in ("random_leaky_vjp", "hard_leaky_vjp"):
        eps = float(os.environ.get("LEAK_EPS", 0.03))
        if not 0.0 < eps <= 1.0:
            raise ValueError(f"LEAK_EPS must be in (0, 1], got {eps}")

        # Backward-only state.  In probability space, a hard row uses
        #   (1-eps) exp(hard_score) + eps exp(soft_score).
        # Soft rows supply frozen state so this auxiliary VJP adds gradients
        # only for hard rows.  Detaching rtau and v prevents duplicate
        # temperature/value gradients; q/k/codebook gradients flow through s.
        if eps == 1.0:
            leaky_hard = rtau.detach()[None, :, None, None] + s
        else:
            hard_relative = logM_hard - rtau[None, :, None, None]
            leaky_relative = torch.logaddexp(
                hard_relative.detach() + math.log1p(-eps),
                s + math.log(eps),
            )
            leaky_hard = rtau.detach()[None, :, None, None] + leaky_relative
        logM_backward = torch.where(use_soft, logM_soft.detach(), leaky_hard)
        out_backward, _ = _dism_from_logm(logM_backward, v.detach())
        out = out + out_backward - out_backward.detach()

    if return_aux:
        assert not hard
        aux = {
            "attn_prob": attn_prob,
            "soft_score": s,
            "q_lse": q_lse,
            "k_lse": k_lse,
            "idx_q": idx_q,
            "idx_k": idx_k,
        }
        if 'A' in locals() and 'B' in locals():
            aux["score_a"] = A
            aux["score_b"] = B
        return out, aux
    return out


class DismMHAttentionV3(nn.Module):
    def __init__(s, d_model: int, n_heads: int, qk_vocab: int, head_dim: int, mix: str = "logaddexp"):
        super().__init__()

        s.mix = mix
        s.qk_vocab = qk_vocab
        s.heads = n_heads
        s.head_dim = head_dim
        s.dc_dim = head_dim // 4
        s.conv_size = 4


        s.q_proj = nn.Linear(d_model, s.heads * s.head_dim, bias=False)
        s.k_proj = nn.Linear(d_model, s.heads * s.head_dim, bias=False)
        s.v_proj = nn.Linear(d_model, s.heads * s.head_dim, bias=False)
        s.o_proj=nn.Linear(n_heads * head_dim, d_model)


        s.norm = FusedRMSNormGated(s.heads * s.head_dim, eps=1e-5)

        # GDN style short conv
        s.v_dism = ShortConvolution(s.heads * s.head_dim, s.conv_size, activation='silu')
        s.qd_conv = ShortConvolution(s.heads * s.head_dim, s.conv_size, activation='swish')
        s.kd_conv = ShortConvolution(s.heads * s.head_dim, s.conv_size, activation='swish')

        # embedding table
        s.q_voc = nn.Parameter(torch.empty((s.heads, s.qk_vocab, s.head_dim), dtype = torch.float32), requires_grad=True)
        nn.init.normal_(s.q_voc)
        if mix in ("soft_tied", "soft_tied_margin"):
            s.k_voc = s.q_voc
        else:
            s.k_voc = nn.Parameter(torch.empty((s.heads, s.qk_vocab, s.head_dim), dtype = torch.float32), requires_grad=True)
            nn.init.normal_(s.k_voc)

        s.q_voc._no_weight_decay = True
        s.k_voc._no_weight_decay = True

        s.log_sel_tau = nn.Parameter(torch.empty(n_heads, dtype=torch.float32).uniform_(1, 4), requires_grad=True)
        s.log_sel_tau._no_weight_decay = True

        sketch_rank = int(os.environ.get("SKETCH_RANK", 8))
        sketch_gen = torch.Generator(device=s.q_voc.device)
        sketch_gen.manual_seed(20260824)
        s.register_buffer(
            "cov_sketch",
            torch.randn(s.heads, s.head_dim, sketch_rank, generator=sketch_gen),
            persistent=False,
        )
        usage_code = torch.randn(
            s.heads, s.qk_vocab, s.head_dim, generator=sketch_gen,
            device=s.q_voc.device,
        )
        usage_code = usage_code - usage_code.mean(dim=1, keepdim=True)
        usage_code = torch.nn.functional.normalize(usage_code, dim=-1)
        s.register_buffer("usage_code", usage_code, persistent=False)

        # output gate
        s.g_proj_down = nn.Linear(d_model, d_model // 8)
        s.g_proj_up = nn.Linear(d_model // 8, s.heads * s.head_dim)

        s.step = 0


    def forward(s, x):
        B, N, C = x.shape

        qp, kp, vp = s.q_proj(x), s.k_proj(x), s.v_proj(x)

        in_q = s.qd_conv(qp)[0].view(B, N, s.heads, s.head_dim).permute(0, 2, 1, 3).contiguous()
        in_k = s.kd_conv(kp)[0].view(B, N, s.heads, s.head_dim).permute(0, 2, 1, 3).contiguous()
        v_bhnc = s.v_dism(vp)[0].view(B, N, s.heads, s.head_dim).permute(0, 2, 1, 3).contiguous()
        temp = torch.nn.functional.softplus(s.log_sel_tau.float()).contiguous()

        lmb = min(s.step / TOTAL_STEPS, 1.0)
        if s.training and s.mix == "hard_leaky_vjp":
            lmb = 1.0
        if s.training:
            s.step += 1
        if s.training and s.mix in ("stratified", "stratified_hybrid"):
            window = int(os.environ.get("SOFT_ANCHOR_WINDOW", 8))
            hybrid_iid = (
                s.mix == "stratified_hybrid"
                and lmb >= float(os.environ.get("STRAT_UNTIL", 0.5))
            )
            if hybrid_iid:
                s.last_soft_fraction = 1.0 - lmb
            elif os.environ.get("STRAT_SCOPE", "window") == "global":
                min_soft = math.ceil(N / window)
                s.last_soft_fraction = max(min_soft, round((1.0 - lmb) * N)) / N
            else:
                s.last_soft_fraction = max(1.0 / window, round((1.0 - lmb) * window) / window)
        if s.training and s.mix in ("soft_tied", "tail_dropout"):
            o = voc_dism(
                in_q, in_k, v_bhnc, temp, s.q_voc, s.k_voc,
                hard=False, lmb=0.0,
                mix=("tail_dropout" if s.mix == "tail_dropout" else "random"),
                gen=getattr(s, 'gen', None),
                sketch=s.cov_sketch,
            )
        elif s.training and s.mix in ("soft_margin", "soft_tied_margin"):
            alpha = float(os.environ.get("MARGIN_ALPHA", 1.0))
            o, margin_aux = voc_dism(
                in_q, in_k, v_bhnc, temp, s.q_voc, s.k_voc,
                hard=False, lmb=0.0, mix="random", gen=getattr(s, 'gen', None),
                sketch=s.cov_sketch, return_aux=True, sm_scale=alpha,
            )
            head_idx = torch.arange(s.heads, device=x.device)[None, :, None]
            selected_q = s.q_voc[head_idx, margin_aux["idx_q"].long()]
            selected_k = s.k_voc[head_idx, margin_aux["idx_k"].long()]
            max_q = alpha * (in_q.float() * selected_q.float()).sum(-1)
            max_k = alpha * (in_k.float() * selected_k.float()).sum(-1)
            nll_top_q = margin_aux["q_lse"] - max_q
            nll_top_k = margin_aux["k_lse"] - max_k
            target_p = float(os.environ.get("MARGIN_TARGET_P", 0.5))
            max_nll = -math.log(target_p)
            s.margin_loss = (
                torch.relu(nll_top_q - max_nll).mean()
                + torch.relu(nll_top_k - max_nll).mean()
            )

            # Prototype differentiable usage balance.  Fixed centered random
            # signatures form a JL sketch of the aggregate code distribution.
            mu_q, _ = _voc_interp(in_q, s.q_voc, s.usage_code)
            mu_k, _ = _voc_interp(in_k, s.k_voc, s.usage_code)
            batch_mu_q = mu_q.float().mean(dim=(0, 2))
            batch_mu_k = mu_k.float().mean(dim=(0, 2))
            s.usage_loss = batch_mu_q.square().sum(-1).mean() + batch_mu_k.square().sum(-1).mean()
            s.last_top_p = 0.5 * (
                torch.exp(-nll_top_q.detach()).mean().item()
                + torch.exp(-nll_top_k.detach()).mean().item()
            )
            s.last_usage = s.usage_loss.detach().item()
        elif s.training and s.mix == "temp_consistency":
            direction = (
                "single_q" if int(torch.randint(0, 2, (1,), generator=getattr(s, 'gen', None)).item()) == 0
                else "single_k"
            )
            alpha_lo = float(os.environ.get("CONS_ALPHA_LO", 1.0))
            alpha_hi = float(os.environ.get("CONS_ALPHA_HI", 2.0))
            o_lo = voc_dism(
                in_q, in_k, v_bhnc, temp, s.q_voc, s.k_voc,
                hard=False, lmb=0.0, mix=direction, gen=getattr(s, 'aux_gen', None),
                sketch=s.cov_sketch, sm_scale=alpha_lo,
            )
            o_hi = voc_dism(
                in_q, in_k, v_bhnc, temp, s.q_voc, s.k_voc,
                hard=False, lmb=0.0, mix=direction, gen=getattr(s, 'aux_gen', None),
                sketch=s.cov_sketch, sm_scale=alpha_hi,
            )
            cos = torch.cosine_similarity(o_lo.float(), o_hi.float(), dim=-1, eps=1e-6)
            lo_norm = o_lo.float().norm(dim=-1)
            hi_norm = o_hi.float().norm(dim=-1)
            valid = (lo_norm > 1e-3) & (hi_norm > 1e-3)
            metric = os.environ.get("CONS_METRIC", "cosine")
            if metric == "relative":
                distance = (o_lo.float() - o_hi.float()).norm(dim=-1) / (
                    0.5 * (lo_norm + hi_norm) + 1e-4
                )
                target = float(os.environ.get("CONS_REL_TARGET", 0.25))
                penalty = torch.relu(distance - target)
                active_mask = distance > target
                s.last_cons_distance = distance[valid].detach().mean().item() if valid.any() else 0.0
            else:
                target = float(os.environ.get("CONS_COS_TARGET", 0.8))
                penalty = torch.relu(target - cos)
                active_mask = cos < target
                s.last_cons_distance = 1.0 - (cos[valid].detach().mean().item() if valid.any() else 0.0)
            s.aux_loss = penalty[valid].mean() if valid.any() else penalty.sum() * 0.0
            s.last_cons_cos = cos[valid].detach().mean().item() if valid.any() else 0.0
            s.last_cons_active = (
                (active_mask & valid).float().sum() / valid.float().sum().clamp_min(1)
            ).item()
            o = o_lo
        elif s.training and s.mix == "soft_renyi":
            buckets = int(os.environ.get("TEMP_BUCKETS", 4))
            alpha_min = float(os.environ.get("ALPHA_MIN", 0.5))
            alpha_max = float(os.environ.get("ALPHA_MAX", 2.0))
            alphas = torch.logspace(
                math.log10(alpha_min), math.log10(alpha_max), buckets,
                device=x.device,
            ).tolist()
            q_chunks = in_q.chunk(buckets, dim=0)
            k_chunks = in_k.chunk(buckets, dim=0)
            v_chunks = v_bhnc.chunk(buckets, dim=0)
            outputs = []
            h2_terms = []
            for qb, kb, vb, alpha in zip(q_chunks, k_chunks, v_chunks, alphas):
                ob, aux = voc_dism(
                    qb, kb, vb, temp, s.q_voc, s.k_voc,
                    hard=False, lmb=0.0, mix="random",
                    gen=getattr(s, 'gen', None), sketch=s.cov_sketch,
                    return_aux=True, sm_scale=alpha,
                )
                outputs.append(ob)
                q2e, k2e, k_lse2, q_lse2, *_ = EmbInterpFunction.apply(
                    qb, kb, s.q_voc.to(qb.dtype), s.k_voc.to(kb.dtype), 2.0 * alpha
                )
                # Zero-valued dependencies ensure the custom backward receives
                # tensor gradients for its interpolation outputs as well as LSE.
                zero = 0.0 * (q2e.sum() + k2e.sum())
                h2_q = 2.0 * aux["q_lse"] - q_lse2
                h2_k = 2.0 * aux["k_lse"] - k_lse2
                h2_terms.append(h2_q.clamp_min(0).mean() + h2_k.clamp_min(0).mean() + zero)
            o = torch.cat(outputs, dim=0)
            s.aux_loss = torch.stack(h2_terms).mean()
            s.last_h2 = s.aux_loss.detach().item()
            s.last_alpha_range = (alpha_min, alpha_max)
        elif s.training and s.mix == "align_distill":
            # Keep the successful random row-mixed task path unchanged.
            o = voc_dism(
                in_q, in_k, v_bhnc, temp, s.q_voc, s.k_voc,
                hard=False, lmb=lmb, mix="random", gen=getattr(s, 'gen', None),
                sketch=s.cov_sketch,
            )
            # A separate all-soft DP supplies long-range positive pairs.  Its
            # output is unused; only a contrastive score loss is retained.
            _, align_aux = voc_dism(
                in_q, in_k, v_bhnc.detach(), temp, s.q_voc, s.k_voc,
                hard=False, lmb=0.0, mix=os.environ.get("ALIGN_TEACHER_MIX", "mean"),
                gen=getattr(s, 'aux_gen', None),
                sketch=s.cov_sketch, return_aux=True,
            )
            teacher_attn = align_aux["attn_prob"].detach()
            score = align_aux["soft_score"]
            pos_idx = teacher_attn.argmax(-1)
            pos_weight = teacher_attn.gather(-1, pos_idx[..., None]).squeeze(-1)
            row_idx = torch.arange(N, device=x.device)[None, None, :]
            min_dist = int(os.environ.get("ALIGN_MIN_DIST", 8))
            min_weight = float(os.environ.get("ALIGN_MIN_WEIGHT", 0.05))
            valid = (row_idx - pos_idx >= min_dist) & (pos_weight >= min_weight)
            if "score_a" in align_aux and "score_b" in align_aux:
                agree_tol = float(os.environ.get("ALIGN_AGREE_TOL", 1.0))
                score_a = align_aux["score_a"].detach()
                score_b = align_aux["score_b"].detach()
                pos_a = score_a.gather(-1, pos_idx[..., None]).squeeze(-1)
                pos_b = score_b.gather(-1, pos_idx[..., None]).squeeze(-1)
                causal = torch.tril(torch.ones(N, N, dtype=torch.bool, device=x.device))
                max_a = score_a.masked_fill(~causal, float('-inf')).max(-1).values
                max_b = score_b.masked_fill(~causal, float('-inf')).max(-1).values
                valid = valid & (pos_a >= max_a - agree_tol) & (pos_b >= max_b - agree_tol)

            neg_u = torch.rand(
                pos_idx.shape, device=x.device, generator=getattr(s, 'aux_gen', None)
            )
            neg_idx = (neg_u * (row_idx + 1)).long()
            # Avoid selecting the positive where a second causal key exists.
            neg_idx = torch.where(
                (neg_idx == pos_idx) & (row_idx > 0), (neg_idx + 1) % (row_idx + 1), neg_idx
            )
            pos_score = score.gather(-1, pos_idx[..., None]).squeeze(-1)
            neg_score = score.gather(-1, neg_idx[..., None]).squeeze(-1)
            margin = float(os.environ.get("ALIGN_MARGIN", 1.0))
            rank_loss = torch.relu(margin + neg_score - pos_score)
            if valid.any():
                s.aux_loss = rank_loss[valid].mean()
                s.last_align_frac = valid.float().mean().item()
                s.last_align_pos = pos_score[valid].detach().mean().item()
                s.last_align_neg = neg_score[valid].detach().mean().item()
                copy_rows = (row_idx >= (N // 2 - 1)) & (row_idx < N - 1)
                copy_valid = valid & copy_rows
                expected_copy_idx = row_idx - (N // 2 - 1)
                s.last_align_copy_acc = (
                    (pos_idx[copy_valid] == expected_copy_idx.expand_as(pos_idx)[copy_valid])
                    .float().mean().item() if copy_valid.any() else 0.0
                )
                s.last_align_copy_near = (
                    ((pos_idx[copy_valid] - expected_copy_idx.expand_as(pos_idx)[copy_valid]).abs() <= 1)
                    .float().mean().item() if copy_valid.any() else 0.0
                )
                s.last_align_copy_offset = (
                    (pos_idx[copy_valid] - expected_copy_idx.expand_as(pos_idx)[copy_valid])
                    .float().mean().item() if copy_valid.any() else 0.0
                )
            else:
                s.aux_loss = score.sum() * 0.0
                s.last_align_frac = 0.0
                s.last_align_pos = 0.0
                s.last_align_neg = 0.0
                s.last_align_copy_acc = 0.0
                s.last_align_copy_near = 0.0
                s.last_align_copy_offset = 0.0
        elif s.training and s.mix == "dual_loss":
            o_hard = voc_dism(
                in_q, in_k, v_bhnc, temp, s.q_voc, s.k_voc,
                hard=True, gen=getattr(s, 'gen', None), sketch=s.cov_sketch,
            )
            o_soft = voc_dism(
                in_q, in_k, v_bhnc, temp, s.q_voc, s.k_voc,
                hard=False, lmb=0.0, mix="random", gen=getattr(s, 'gen', None),
                sketch=s.cov_sketch,
            )
            o = (o_hard, o_soft)
        elif s.training and s.mix == "agreement":
            o_soft = voc_dism(
                in_q, in_k, v_bhnc, temp, s.q_voc, s.k_voc,
                hard=False, lmb=0.0, mix="random", gen=getattr(s, 'gen', None),
                sketch=s.cov_sketch,
            )
            o_hard = voc_dism(
                in_q, in_k, v_bhnc, temp, s.q_voc, s.k_voc,
                hard=True, gen=getattr(s, 'gen', None), sketch=s.cov_sketch,
            )
            with torch.no_grad():
                soft_norm = o_soft.float().norm(dim=-1)
                hard_norm = o_hard.float().norm(dim=-1)
                relerr = (o_hard.float() - o_soft.float()).norm(dim=-1) / soft_norm.clamp_min(1e-4)
                err_threshold = float(os.environ.get("AGREE_ERR", 0.5))
                norm_ratio = float(os.environ.get("AGREE_NORM_RATIO", 0.25))
                use_hard = (
                    (soft_norm > 1e-3)
                    & (hard_norm > norm_ratio * soft_norm)
                    & (relerr < err_threshold)
                )
                s.last_hard_frac = use_hard.float().mean().item()
            o = torch.where(use_hard[..., None], o_hard, o_soft)
        elif s.training and s.mix == "random_v_aux":
            o = voc_dism(
                in_q, in_k, v_bhnc, temp, s.q_voc, s.k_voc,
                hard=False, lmb=lmb, mix="random", gen=getattr(s, 'gen', None),
                sketch=s.cov_sketch,
            )
            o_soft_v = voc_dism(
                in_q.detach(), in_k.detach(), v_bhnc, temp.detach(),
                s.q_voc.detach(), s.k_voc.detach(), hard=False, lmb=0.0,
                mix="random", gen=getattr(s, 'aux_gen', None), sketch=s.cov_sketch,
            )
            aux = float(os.environ.get("V_AUX_COEF", 0.1)) * lmb
            o = o + aux * (o_soft_v - o_soft_v.detach())
        elif s.training and s.mix in ("confidence_v", "random_v"):
            # Let q/k follow the actual confidence-gated mixed graph, but
            # replace its sparse v gradient with a dense all-soft v gradient.
            o_mix = voc_dism(
                in_q, in_k, v_bhnc.detach(), temp, s.q_voc, s.k_voc,
                hard=False, lmb=(lmb if s.mix == "random_v" else 0.0),
                mix=("random" if s.mix == "random_v" else "confidence"),
                gen=getattr(s, 'gen', None),
                sketch=s.cov_sketch,
            )
            o_soft_v = voc_dism(
                in_q.detach(), in_k.detach(), v_bhnc, temp.detach(),
                s.q_voc.detach(), s.k_voc.detach(), hard=False, lmb=0.0,
                mix="random", gen=getattr(s, 'gen', None), sketch=s.cov_sketch,
            )
            o = o_mix + o_soft_v - o_soft_v.detach()
        elif s.training and s.mix in ("hard_st", "confidence_st"):
            # Hard forward, soft backward.  Detaching every hard-path input
            # prevents sparse/duplicated gradients (especially for v); the
            # numerically equal zero term grafts the dense soft surrogate
            # gradient onto the hard output without a soft/hard schedule.
            with torch.no_grad():
                if s.mix == "hard_st":
                    o_hard = voc_dism(
                        in_q, in_k, v_bhnc, temp, s.q_voc, s.k_voc,
                        hard=True, gen=getattr(s, 'gen', None), sketch=s.cov_sketch,
                    )
                else:
                    o_hard = voc_dism(
                        in_q, in_k, v_bhnc, temp, s.q_voc, s.k_voc,
                        hard=False, mix="confidence", gen=getattr(s, 'gen', None),
                        sketch=s.cov_sketch,
                    )
            o_soft = voc_dism(
                in_q, in_k, v_bhnc, temp, s.q_voc, s.k_voc,
                hard=False, lmb=0.0, mix="random",
                gen=getattr(s, 'gen', None), sketch=s.cov_sketch,
            )
            o = o_hard + o_soft - o_soft.detach()
        else:
            o = voc_dism(in_q, in_k, v_bhnc, temp, s.q_voc, s.k_voc, hard=(not s.training), lmb=lmb, mix=s.mix, gen=getattr(s, 'gen', None), sketch=s.cov_sketch)

        #lens = torch.norm(o_hard.reshape(-1, s.head_dim), 2, -1) > 1e-3

        #print(f"Sim = {torch.cosine_similarity(o.reshape(-1, s.head_dim), o_hard.reshape(-1, s.head_dim))[lens].mean().item()}")

        g = s.g_proj_up((s.g_proj_down(x)))

        def finish(oi):
            oi = oi.permute(0, 2, 1, 3).reshape(B, N, -1)
            return s.o_proj(s.norm(oi, g))

        if isinstance(o, tuple):
            return finish(o[0]), finish(o[1])
        return finish(o)

@torch.compile
class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden_dim)
        self.w2 = nn.Linear(d_model, hidden_dim)
        self.proj = nn.Linear(hidden_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated = torch.nn.functional.silu(self.w1(x)) * self.w2(x)
        return self.proj(gated)

class DismTransformerBlock(nn.Module):
    def __init__(self, d_model, mix: str = "logaddexp") -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = DismMHAttentionV3(d_model, 4, 256, d_model // 4, mix=mix)
        self.ffn = SwiGLU(d_model, d_model * 4)

    def forward(
        self,
        x: torch.Tensor
    ):
        attn_out = self.attn(self.ln1(x))
        if isinstance(attn_out, tuple):
            branches = []
            for ao in attn_out:
                xb = x + ao
                xb = xb + self.ffn(self.ln2(xb))
                branches.append(xb)
            return torch.cat(branches, dim=0)
        x = x + attn_out
        return x + self.ffn(self.ln2(x))


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.qk = nn.Linear(d_model, 2*(self.n_heads * self.head_dim), bias=False)
        self.v = nn.Linear(d_model, self.n_heads * self.head_dim)
        self.out = nn.Linear(self.n_heads * self.head_dim, d_model)
        self.rope_theta = 10000.0
        self.register_buffer("_rope_cos", torch.empty(0), persistent=False)
        self.register_buffer("_rope_sin", torch.empty(0), persistent=False)

        self.norm = FusedRMSNormGated(self.n_heads * self.head_dim, eps=1e-5)
        self.gate = nn.Linear(d_model, self.n_heads * self.head_dim)

        nn.init.xavier_uniform_(self.qk.weight)
        nn.init.xavier_uniform_(self.v.weight)
        nn.init.zeros_(self.v.bias)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(
        self,
        x: torch.Tensor
    ):
        bsz, seq, dim = x.shape
        qk = self.qk(x)
        q, k = qk.chunk(2, dim=-1)
        q = q.view(bsz, seq, self.n_heads, self.head_dim)
        k = k.view(bsz, seq, self.n_heads, self.head_dim)
        v = self.v(x).view(bsz, seq, self.n_heads, self.head_dim)

        o = flash_attn_func(q.to(v.dtype), k.to(v.dtype), v, causal=True)
        
        out = o.view(bsz, seq, self.head_dim * self.n_heads)
        g = self.gate(x)
        return self.out(self.norm(out, g))

    
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.shape[1]].to(x.dtype)

class TransformerBlock(nn.Module):
    def __init__(self, d_model) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, 4)
        self.ffn = SwiGLU(d_model, d_model * 4)

    def forward(
        self,
        x: torch.Tensor
    ):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class GDNBlock(nn.Module):
    def __init__(self, d_model) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = GatedDeltaNet(d_model, 2, 64, 4)
        self.ffn = SwiGLU(d_model, d_model * 4)

    def forward(
        self,
        x: torch.Tensor
    ):
        x = x + self.attn(self.ln1(x))[0]
        x = x + self.ffn(self.ln2(x))
        return x

def _build_cosine_warmup_scheduler(
    optimizer: torch.optim.AdamW,
    max_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        progress_clamped = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress_clamped))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

EVAL_SEED = 424242

def eval(m: torch.nn.Module):
    m.eval()

    losses = []
    g_eval = torch.Generator(device='cuda')
    g_eval.manual_seed(EVAL_SEED)

    for i in range(10):
        with torch.no_grad():
            x = torch.randint(0, VOC - 1, (B, N // 2, ), generator=g_eval)
            x_in = torch.cat([x, x], -1)
            with torch.autocast('cuda', torch.bfloat16):
                y = m(x_in) # [B, N, VOC]
        
            pred = y[:, (N//2)-1:-1, :]
            gt = x_in[:, (N//2):]
        
            loss = nn.functional.cross_entropy(pred.reshape(-1, VOC), gt.flatten())
            losses.append(loss.item())
    el = sum(losses)/len(losses)
    print(f"Eval loss = {el:.4f}")
    m.train()
    return el

if __name__ == "__main__":
    # A toy task to copy long sequence

    MIX = sys.argv[1] if len(sys.argv) > 1 else "logaddexp"
    SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    DATA_SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 12345

    VOC = 128
    B = 64
    N = 128
    TOTAL_STEPS = int(os.environ.get("TOTAL_STEPS", 1000))
    torch.set_default_device('cuda')
    torch.manual_seed(SEED)
    random.seed(SEED)
    m = nn.Sequential(nn.Embedding(VOC, 128), DismTransformerBlock(128, mix=MIX), nn.RMSNorm((128,)), nn.Linear(128, VOC))

    g_data = torch.Generator(device='cuda')
    g_data.manual_seed(DATA_SEED)
    g_drop = torch.Generator(device='cuda')
    g_drop.manual_seed(777)
    m[1].attn.gen = g_drop
    g_aux = torch.Generator(device='cuda')
    g_aux.manual_seed(778)
    m[1].attn.aux_gen = g_aux
    #    m = nn.Sequential(nn.Embedding(VOC, 128), SinusoidalPositionalEncoding(128), TransformerBlock(128), nn.RMSNorm((128,)), nn.Linear(128, VOC))
    decay_params = [p for p in m.parameters() if not hasattr(p, '_no_weight_decay')]
    no_decay_params = [p for p in m.parameters() if hasattr(p, '_no_weight_decay')]
    optim = torch.optim.AdamW([
        {'params': no_decay_params, 'weight_decay': 0.0},
        {'params': decay_params, 'weight_decay': 1e-2},
    ], lr=5e-3)
    sched = _build_cosine_warmup_scheduler(optim, TOTAL_STEPS, 50, 0.1)

    evals = []
    for i in range(TOTAL_STEPS):
        x = torch.randint(0, VOC - 1, (B, N // 2, ), generator=g_data)
        x_in = torch.cat([x, x], -1)
        with torch.autocast('cuda', torch.bfloat16):
            y = m(x_in) # [B, N, VOC]

        if MIX == "dual_loss":
            y_hard, y_soft = y.chunk(2, dim=0)
            pred_hard = y_hard[:, (N//2)-1:-1, :]
            pred_soft = y_soft[:, (N//2)-1:-1, :]
            gt = x_in[:, (N//2):]
            loss_hard = nn.functional.cross_entropy(pred_hard.reshape(-1, VOC), gt.flatten())
            loss_soft = nn.functional.cross_entropy(pred_soft.reshape(-1, VOC), gt.flatten())
            dual_beta = float(os.environ.get("DUAL_BETA", 1.0))
            loss = (loss_hard + dual_beta * loss_soft) / (1.0 + dual_beta)
        else:
            pred = y[:, (N//2)-1:-1, :]
            gt = x_in[:, (N//2):]
            loss = nn.functional.cross_entropy(pred.reshape(-1, VOC), gt.flatten())
        if MIX == "align_distill":
            loss = loss + float(os.environ.get("ALIGN_COEF", 0.1)) * m[1].attn.aux_loss
        if MIX == "soft_renyi":
            loss = loss + float(os.environ.get("RENYI_COEF", 0.01)) * m[1].attn.aux_loss
        if MIX == "temp_consistency":
            loss = loss + float(os.environ.get("CONS_COEF", 0.1)) * m[1].attn.aux_loss
        if MIX in ("soft_margin", "soft_tied_margin"):
            margin_gate = torch.exp(-loss.detach()).clamp(max=1.0)
            usage_excess = torch.relu(
                m[1].attn.usage_loss - float(os.environ.get("USAGE_MAX", 0.1))
            )
            loss = (
                loss
                + float(os.environ.get("MARGIN_COEF", 0.01)) * margin_gate * m[1].attn.margin_loss
                + float(os.environ.get("USAGE_COEF", 0.1)) * usage_excess
            )
            m[1].attn.last_margin_gate = margin_gate.item()
        optim.zero_grad(True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        optim.step()
        sched.step()

        #wandb.log({'log': loss.item()})
        print(f"step = {i}, loss={loss.item():4f}, lr = {optim.param_groups[0]['lr']:.4f}, gnorm = {gnorm.mean().item():.4f}")

        if i % 100 == 0:
            if hasattr(m[1].attn, 'last_hard_frac'):
                print(f"train hard row fraction = {m[1].attn.last_hard_frac:.4f}")
            if hasattr(m[1].attn, 'last_align_frac'):
                print(
                    f"align fraction = {m[1].attn.last_align_frac:.4f}, "
                    f"pos = {m[1].attn.last_align_pos:.4f}, neg = {m[1].attn.last_align_neg:.4f}, "
                    f"copy acc = {m[1].attn.last_align_copy_acc:.4f}, "
                    f"copy near = {m[1].attn.last_align_copy_near:.4f}, "
                    f"copy offset = {m[1].attn.last_align_copy_offset:.4f}"
                )
            if hasattr(m[1].attn, 'last_h2'):
                print(
                    f"renyi h2 = {m[1].attn.last_h2:.4f}, "
                    f"alpha range = {m[1].attn.last_alpha_range}"
                )
            if hasattr(m[1].attn, 'last_cons_cos'):
                print(
                    f"temperature consistency cosine = {m[1].attn.last_cons_cos:.4f}, "
                    f"distance = {m[1].attn.last_cons_distance:.4f}, "
                    f"active fraction = {m[1].attn.last_cons_active:.4f}"
                )
            if hasattr(m[1].attn, 'last_top_p'):
                print(
                    f"mean top probability = {m[1].attn.last_top_p:.4f}, "
                    f"usage sketch loss = {m[1].attn.last_usage:.6f}, "
                    f"margin gate = {getattr(m[1].attn, 'last_margin_gate', 0.0):.4f}"
                )
            if hasattr(m[1].attn, 'last_soft_fraction'):
                print(f"stratified expected soft fraction = {m[1].attn.last_soft_fraction:.4f}")
            evals.append((i, eval(m)))

    final = sum(e for _, e in evals[-3:]) / 3
    print(f"RESULT mix={MIX} seed={SEED} final_eval={final:.4f} last_evals={[f'{i}:{e:.4f}' for i, e in evals[-5:]]}")
