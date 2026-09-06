import torch


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    stacked = torch.stack((-x_odd, x_even), dim=-1)
    return stacked.flatten(-2)


def build_rope_cache(
    seq_len: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    theta: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    freqs = torch.outer(pos, inv_freq)
    emb = torch.repeat_interleave(freqs, repeats=2, dim=-1)
    cos = emb.cos().to(dtype=dtype)
    sin = emb.sin().to(dtype=dtype)
    return cos, sin

@torch.compile
def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_out = (q * cos) + (_rotate_half(q) * sin)
    k_out = (k * cos) + (_rotate_half(k) * sin)
    return q_out, k_out

@torch.compile
def apply_rope_bnhc(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    q_out = (q * cos) + (_rotate_half(q) * sin)
    k_out = (k * cos) + (_rotate_half(k) * sin)
    return q_out, k_out
    

@torch.compile
def apply_rope_Z(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(0)
    sin = sin.unsqueeze(0)
    q_out = (q * cos) + (_rotate_half(q) * sin)
    k_out = (k * cos) + (_rotate_half(k) * sin)
    return q_out, k_out

@torch.compile
def apply_rope_H(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos[None,:,None]
    sin = sin[None,:,None]
    q_out = (q * cos) + (_rotate_half(q) * sin)
    k_out = (k * cos) + (_rotate_half(k) * sin)
    return q_out, k_out

@torch.compile
def apply_rope_vo(v: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    v_out = (v * cos) + (_rotate_half(v) * sin)
    return v_out.to(v.dtype)


@torch.compile
def apply_rope_vo_T(v: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    v_out = (v * cos) - (_rotate_half(v) * sin)
    return v_out.to(v.dtype)

@torch.compile
def apply_rope_vo_bnhc_T(v: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    split_pt = v.shape[-1]//2
    v_p, v_q = v[..., :split_pt], v[..., split_pt:]
    v_out = (v_p * cos) - (_rotate_half(v_p) * sin)
    return torch.cat([v_out, v_q], dim=-1)

@torch.compile
def apply_rope_vo_bnhc(v: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    v = (v * cos) - (_rotate_half(v) * sin)
    return v