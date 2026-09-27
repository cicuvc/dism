"""Single-recurrence joint-leaky Dism prototype.

The default ``LEAK_MODE=row`` writes the row switch and probability leakage as
one random interpolation coefficient::

    alpha = 1          with probability 1-lambda
            final_eps  with probability lambda
    logM = logaddexp(hard + log(1-alpha), soft + log(alpha)) + rtau

Thus ``use_soft`` is only the Bernoulli source for ``alpha``; there is no
separate outer hard/soft selection after interpolation.  ``scheduled`` removes
the row randomness, while ``bernoulli`` removes logaddexp.  Both are retained
as diagnostic simplifications, but the default row mode has the best 2000-step
result so far.
"""

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


def voc_dism(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, rtau: torch.Tensor, q_voc: torch.Tensor, k_voc: torch.Tensor, hard: bool = False, lmb: float = 0.5, gen: torch.Generator = None, sm_scale: float = 1.0) -> torch.Tensor:
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
        logM = rtau[None,:,None,None] + torch.where(idx_q[:,:,:,None] == idx_k[:,:,None,:], 0, -1e7)

    else:
        # emb kernel: q_embs = softmax(kk)@q_voc, k_embs = softmax(qq)@k_voc,
        #             k_lse = logsumexp(kk), q_lse = logsumexp(qq),
        #             ptop_k = top1(softmax(kk)), ptop_q = top1(softmax(qq)),
        #             idx_k = argmax(kk), idx_q = argmax(qq)
        q_embs, k_embs, k_lse, q_lse, ptop_k, ptop_q, idx_k, idx_q = EmbInterpFunction.apply(
            q, k, q_voc.to(q.dtype), k_voc.to(k.dtype), sm_scale)

        logM_hard_pre =torch.where(idx_q[:,:,:,None] == idx_k[:,:,None,:], 0, -1e7)

        flip = int(torch.randint(0, 2, (1,), generator=gen).item())
        if flip == 0:
            logM_soft_pre = (
                sm_scale * (q @ q_embs.transpose(-1, -2))
                - q_lse[..., None]
            )
        else:
            logM_soft_pre = (
                sm_scale * (k @ k_embs.transpose(-1, -2))
                - k_lse[..., None]
            ).transpose(-1, -2)

        use_soft = torch.rand(
            logM_soft_pre.shape[:-1] + (1,),
            generator=gen,
            device=logM_soft_pre.device,
        ) > lmb
        
        logM = torch.where(use_soft, logM_soft_pre, logM_hard_pre) + rtau[None, :, None, None]

    out, attn_prob = _dism_from_logm(logM, v)

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

        s.k_voc = nn.Parameter(torch.empty((s.heads, s.qk_vocab, s.head_dim), dtype = torch.float32),requires_grad=True)
        nn.init.normal_(s.k_voc)

        s.q_voc._no_weight_decay = True
        s.k_voc._no_weight_decay = True

        s.log_sel_tau = nn.Parameter(torch.empty(n_heads, dtype=torch.float32).uniform_(-4, 4), requires_grad=True)
        s.log_sel_tau._no_weight_decay = True

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

        if s.training:
            s.step += 1
        
        o = voc_dism(
            in_q, in_k, v_bhnc, temp, s.q_voc, s.k_voc,
            hard=(not s.training), lmb=lmb, gen=getattr(s, 'gen', None)
        )

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

def softcap_logits(logits: torch.Tensor) -> torch.Tensor:
    """Apply the optional final-logit soft cap immediately before CE."""
    cap = float(os.environ.get("LOGITS_SOFTCAP", 0.0))
    if cap <= 0.0:
        return logits
    return cap * torch.tanh(logits / cap)


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
                y = softcap_logits(y)
        
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
    NUM_LAYERS = int(os.environ.get("NUM_LAYERS", 1))
    if NUM_LAYERS < 1:
        raise ValueError(f"NUM_LAYERS must be positive, got {NUM_LAYERS}")
    torch.set_default_device('cuda')
    torch.manual_seed(SEED)
    random.seed(SEED)
    token_embedding = nn.Embedding(VOC, 128)
    blocks = [DismTransformerBlock(128, mix=MIX) for _ in range(NUM_LAYERS)]
    eps_by_layer_text = os.environ.get("LEAK_EPS_BY_LAYER", "")
    eps_by_layer = None
    if eps_by_layer_text:
        eps_by_layer = [float(value.strip()) for value in eps_by_layer_text.split(",")]
        if len(eps_by_layer) != NUM_LAYERS:
            raise ValueError(
                f"LEAK_EPS_BY_LAYER needs {NUM_LAYERS} values, got {eps_by_layer}"
            )
        for block, layer_eps in zip(blocks, eps_by_layer):
            block.attn.leak_final_eps = layer_eps
    m = nn.Sequential(
        token_embedding,
        *blocks,
        nn.RMSNorm((128,)),
        nn.Linear(128, VOC),
    )

    g_data = torch.Generator(device='cuda')
    g_data.manual_seed(DATA_SEED)
    g_drop = torch.Generator(device='cuda')
    g_drop.manual_seed(777)
    for block in blocks:
        block.attn.gen = g_drop
    g_aux = torch.Generator(device='cuda')
    g_aux.manual_seed(778)
    for block in blocks:
        block.attn.aux_gen = g_aux
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
            y = softcap_logits(y)


        pred = y[:, (N//2)-1:-1, :]
        gt = x_in[:, (N//2):]
        loss = nn.functional.cross_entropy(pred.reshape(-1, VOC), gt.flatten())
        
        optim.zero_grad(True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        optim.step()
        sched.step()

        #wandb.log({'log': loss.item()})
        print(f"step = {i}, loss={loss.item():4f}, lr = {optim.param_groups[0]['lr']:.4f}, gnorm = {gnorm.mean().item():.4f}")

        if i % 100 == 0:
            evals.append((i, eval(m)))

    final = sum(e for _, e in evals[-3:]) / 3
    print(
        f"RESULT mix={MIX} layers={NUM_LAYERS} "
        f"seed={SEED} final_eval={final:.4f} "
        f"last_evals={[f'{i}:{e:.4f}' for i, e in evals[-5:]]}"
    )
