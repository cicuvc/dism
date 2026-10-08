"""
Unified Flash Attention interface with automatic FA3/FA2/SDPA switching.

Exports `flash_attn` module that matches the FA3 API exactly, but falls back
to a locally installed FA2 and then PyTorch SDPA.

Usage (drop-in replacement for FA3):
    from nanochat.flash_attention import flash_attn

    # Training (no KV cache)
    y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)

    # Packed varlen training (compile-stable cu_seqlens + segment_ids)
    y = flash_attn.flash_attn_varlen_func(q, k, v, cu_seqlens, T, segment_ids,
                                          causal=True, window_size=window_size)

    # Inference (with KV cache)
    y = flash_attn.flash_attn_with_kvcache(q, k_cache, v_cache, k=k, v=v, ...)
"""
import torch
import torch.nn.functional as F


# =============================================================================
# Detection: Try to load FA3/FA2 on CUDA GPUs
# =============================================================================
def _load_flash_attention_3():
    """Try to load Flash Attention 3."""
    if not torch.cuda.is_available():
        return None
    try:
        major, _ = torch.cuda.get_device_capability()
        # FA3 kernels are currently compiled for Hopper (sm90), Ada (sm89) and Ampere (sm80/sm86)
        # Blackwell (sm100) needs SDPA fallback until FA3 is recompiled or FA4 is released
        import os
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        from kernels import get_kernel, has_kernel
        # The varunneal kernel obtains better results for H100/Hopper
        if major == 9:
            hf_kernel = "varunneal/flash-attention-3"
            return get_kernel(hf_kernel).flash_attn_interface
        else:
            hf_kernel = "kernels-community/flash-attn3"
            if has_kernel(hf_kernel):
                return get_kernel(hf_kernel).flash_attn_interface
            else:
                return None

    except Exception:
        return None


_fa3 = _load_flash_attention_3()
HAS_FA3 = _fa3 is not None


def _load_flash_attention_2():
    """Try to load a locally installed Flash Attention 2 package."""
    if not torch.cuda.is_available():
        return None
    try:
        import flash_attn as flash_attn_2
        assert hasattr(flash_attn_2, "flash_attn_func")
        assert hasattr(flash_attn_2, "flash_attn_varlen_func")
        assert hasattr(flash_attn_2, "flash_attn_with_kvcache")
        return flash_attn_2
    except Exception:
        return None


_fa2 = _load_flash_attention_2()
HAS_FA2 = _fa2 is not None

# Override for testing: set to 'fa3', 'fa2', 'sdpa', or None (auto)
_override_impl = None


def _resolve_impl():
    """Choose FA3, then FA2, then SDPA based on availability and dtype."""
    from nanochat.common import COMPUTE_DTYPE
    if _override_impl == 'fa3':
        assert HAS_FA3, "Cannot override to FA3: not available on this hardware"
        return 'fa3'
    if _override_impl == 'fa2':
        assert HAS_FA2, "Cannot override to FA2: not available in this environment"
        assert COMPUTE_DTYPE in (torch.float16, torch.bfloat16), \
            f"FA2 requires fp16 or bf16, got {COMPUTE_DTYPE}"
        return 'fa2'
    if _override_impl == 'sdpa':
        return 'sdpa'
    if HAS_FA3:
        # FA3 Hopper kernels only support bf16 and fp8; fp16/fp32 must use SDPA fallback
        if COMPUTE_DTYPE == torch.bfloat16:
            return 'fa3'
    if HAS_FA2 and COMPUTE_DTYPE in (torch.float16, torch.bfloat16):
        return 'fa2'
    return 'sdpa'


def _resolve_use_fa3():
    """Backward-compatible resolver used by existing callers and tests."""
    return _resolve_impl() == 'fa3'


def _resolve_use_fa2():
    return _resolve_impl() == 'fa2'

USE_FA3 = _resolve_use_fa3()
USE_FA2 = _resolve_use_fa2()


def _refresh_impl_flags():
    """Refresh cached implementation flags after changing the test override."""
    global USE_FA3, USE_FA2
    USE_FA3 = _resolve_use_fa3()
    USE_FA2 = _resolve_use_fa2()


# =============================================================================
# SDPA helpers
# =============================================================================
def _sdpa_attention(q, k, v, window_size, enable_gqa):
    """
    SDPA attention with sliding window support.
    q, k, v are (B, H, T, D) format.
    """
    Tq = q.size(2)
    Tk = k.size(2)
    window = window_size[0]

    # Full context, same length
    if (window < 0 or window >= Tq) and Tq == Tk:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)

    # Single token generation
    if Tq == 1:
        if window >= 0 and window < Tk:
            # window is "left" tokens we need to include (window + 1) keys total
            start = max(0, Tk - (window + 1))
            k = k[:, :, start:, :]
            v = v[:, :, start:, :]
        return F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)

    # Need explicit mask for sliding window/chunk inference
    device = q.device
    # For chunk inference (Tq != Tk), is_causal is not aligned to cache position => build an explicit bool mask
    row_idx = (Tk - Tq) + torch.arange(Tq, device=device).unsqueeze(1)
    col_idx = torch.arange(Tk, device=device).unsqueeze(0)
    mask = col_idx <= row_idx

    # sliding window (left)
    if window >= 0 and window < Tk:
        mask = mask & ((row_idx - col_idx) <= window)

    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=enable_gqa)

# =============================================================================
# Public API: Same interface as FA3
# =============================================================================
def flash_attn_func(q, k, v, causal=False, window_size=(-1, -1)):
    """
    Flash Attention for training (no KV cache).

    Args:
        q, k, v: Tensors of shape (B, T, H, D)
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.

    Returns:
        Output tensor of shape (B, T, H, D)
    """
    if USE_FA3:
        return _fa3.flash_attn_func(q, k, v, causal=causal, window_size=window_size)
    if USE_FA2:
        return _fa2.flash_attn_func(q, k, v, causal=causal, window_size=window_size)

    # SDPA fallback: transpose (B, T, H, D) -> (B, H, T, D)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    enable_gqa = q.size(1) != k.size(1)
    y = _sdpa_attention(q, k, v, window_size, enable_gqa)
    return y.transpose(1, 2)  # back to (B, T, H, D)


def flash_attn_varlen_func(q, k, v, cu_seqlens, max_seqlen, segment_ids,
                           causal=False, window_size=(-1, -1)):
    """
    Variable-length self-attention over a fixed-shape packed batch.

    Args:
        q, k, v: Tensors of shape (B, T, H, D).
        cu_seqlens: Fixed-size int32 cumulative offsets. Unused entries repeat B*T.
        max_seqlen: Static upper bound for every segment (normally T).
        segment_ids: Contiguous segment id per token, shape (B, T). This is used
            by the SDPA fallback and deliberately has a data-independent shape.
    """
    B, T = segment_ids.shape
    assert q.shape[:2] == (B, T)
    assert k.shape[:2] == (B, T)
    assert v.shape[:2] == (B, T)

    if USE_FA3:
        q_flat = q.reshape(B * T, q.size(2), q.size(3))
        k_flat = k.reshape(B * T, k.size(2), k.size(3))
        v_flat = v.reshape(B * T, v.size(2), v.size(3))
        # Dynamo cannot trace autograd.Function.apply when the exact same tensor
        # object is supplied for both Q and K metadata inputs.
        cu_seqlens_k = cu_seqlens.clone()
        y = _fa3.flash_attn_varlen_func(
            q_flat, k_flat, v_flat,
            cu_seqlens, cu_seqlens_k,
            max_seqlen, max_seqlen,
            causal=causal, window_size=window_size,
        )
        return y.reshape(B, T, q.size(2), q.size(3))
    if USE_FA2:
        q_flat = q.reshape(B * T, q.size(2), q.size(3))
        k_flat = k.reshape(B * T, k.size(2), k.size(3))
        v_flat = v.reshape(B * T, v.size(2), v.size(3))
        cu_seqlens_k = cu_seqlens.clone()
        y = _fa2.flash_attn_varlen_func(
            q_flat, k_flat, v_flat,
            cu_seqlens, cu_seqlens_k,
            max_seqlen, max_seqlen,
            causal=causal, window_size=window_size,
        )
        return y.reshape(B, T, q.size(2), q.size(3))

    # SDPA fallback. The segment equality mask is block diagonal; causal/window
    # constraints are based on absolute positions, whose differences are the same
    # as segment-local positions.
    q_sdpa = q.transpose(1, 2)
    k_sdpa = k.transpose(1, 2)
    v_sdpa = v.transpose(1, 2)
    positions = torch.arange(T, device=q.device)
    query_pos = positions.view(T, 1)
    key_pos = positions.view(1, T)
    mask = segment_ids[:, :, None] == segment_ids[:, None, :]
    if causal:
        mask = mask & (key_pos <= query_pos)
    left_window, right_window = window_size
    if left_window >= 0:
        mask = mask & ((query_pos - key_pos) <= left_window)
    if right_window >= 0:
        mask = mask & ((key_pos - query_pos) <= right_window)
    mask = mask.unsqueeze(1)
    enable_gqa = q_sdpa.size(1) != k_sdpa.size(1)
    y = F.scaled_dot_product_attention(
        q_sdpa, k_sdpa, v_sdpa, attn_mask=mask, enable_gqa=enable_gqa
    )
    return y.transpose(1, 2)


def flash_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None, cache_seqlens=None,
                            causal=False, window_size=(-1, -1)):
    """
    Flash Attention with KV cache for inference.

    FA3 updates k_cache/v_cache in-place. Our SDPA fallback does the same.

    Args:
        q: Queries, shape (B, T_new, H, D)
        k_cache, v_cache: Pre-allocated cache tensors, shape (B, T_max, H_kv, D)
        k, v: New keys/values to insert, shape (B, T_new, H_kv, D)
        cache_seqlens: Current position in cache, shape (B,) int32
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.

    Returns:
        Output tensor of shape (B, T_new, H, D)
    """
    if USE_FA3:
        return _fa3.flash_attn_with_kvcache(
            q, k_cache, v_cache, k=k, v=v, cache_seqlens=cache_seqlens,
            causal=causal, window_size=window_size
        )
    if USE_FA2:
        return _fa2.flash_attn_with_kvcache(
            q, k_cache, v_cache, k=k, v=v, cache_seqlens=cache_seqlens,
            causal=causal, window_size=window_size
        )

    # SDPA fallback: manually manage KV cache
    B, T_new, H, D = q.shape
    pos = cache_seqlens[0].item()  # assume uniform position across batch

    # Insert new k, v into cache (in-place, matching FA3 behavior)
    if k is not None and v is not None:
        k_cache[:, pos:pos+T_new, :, :] = k
        v_cache[:, pos:pos+T_new, :, :] = v

    # Get full cache up to current position + new tokens
    end_pos = pos + T_new
    k_full = k_cache[:, :end_pos, :, :]
    v_full = v_cache[:, :end_pos, :, :]

    # Transpose to SDPA layout: (B, T, H, D) -> (B, H, T, D)
    q_sdpa = q.transpose(1, 2)
    k_sdpa = k_full.transpose(1, 2)
    v_sdpa = v_full.transpose(1, 2)

    enable_gqa = q_sdpa.size(1) != k_sdpa.size(1)
    y_sdpa = _sdpa_attention(q_sdpa, k_sdpa, v_sdpa, window_size, enable_gqa)

    return y_sdpa.transpose(1, 2)  # back to (B, T, H, D)


# =============================================================================
# Export: flash_attn module interface (drop-in replacement for FA3)
# =============================================================================
from types import SimpleNamespace
flash_attn = SimpleNamespace(
    flash_attn_func=flash_attn_func,
    flash_attn_varlen_func=flash_attn_varlen_func,
    flash_attn_with_kvcache=flash_attn_with_kvcache,
)
