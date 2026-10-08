"""DISM plus FlashAttention sliding-window attention, with one shared V path."""
import torch
from .module import DismAttention
from .kernels.linear_rmsnorm_rope import FusedLinearRMSNormRoPE


@torch.compiler.allow_in_graph
def _compiled_varlen_swa(q, k, v, boundaries, maximum, window):
    """Keep repeated FA autograd wrappers out of Dynamo's nested tracers.

    AOTAutograd still traces the registered FlashAttention forward/backward ops.
    No tensor data is inspected here and no eager graph break is introduced.
    """
    from flash_attn import flash_attn_varlen_func
    return flash_attn_varlen_func(q, k, v, boundaries, boundaries, maximum, maximum,
                                 causal=True, window_size=(window-1, 0), dropout_p=0.,
                                 softmax_scale=q.shape[-1] ** -0.5)


class DismSwaAttention(DismAttention):
    """Independent hybrid variant; inherits FLA dispatch and DISM parameters.

    SWA Q/K are separate Linear+per-head RMSNorm+RoPE branches. Their head
    dimension equals value_dim (32/64), as required by FlashAttention 2.
    window_size counts the current token: default128 means [i-127, i].
    Raw outputs are added before the single shared output norm/gate/projection.
    """
    _decode_extra_states = 1

    def __init__(self, width, heads, *, window_size=128, **kwargs):
        super().__init__(width, heads, **kwargs)
        if isinstance(window_size, bool) or not isinstance(window_size, int) or window_size < 1:
            raise ValueError("window_size must be a positive integer")
        self.window_size = window_size
        self.swa_q_proj = FusedLinearRMSNormRoPE(width, heads, self.value_dim, eps=self.qknorm_eps)
        self.swa_k_proj = FusedLinearRMSNormRoPE(width, heads, self.value_dim, eps=self.qknorm_eps)
        self.swa_q_proj.rms_weight._no_weight_decay = True
        self.swa_k_proj.rms_weight._no_weight_decay = True

    def _swa_cuda(self, x, v, cu_seqlens=None, max_seqlen=None):
        # Lazy import: the plain DISM module does not require FlashAttention.
        from flash_attn import flash_attn_func, flash_attn_varlen_func
        length = x.shape[1] if cu_seqlens is None else max_seqlen
        cos, sin = self._rotary_tables(length, x.device, dimension=self.value_dim)
        q = self.swa_q_proj(x.contiguous(), cos, sin, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        k = self.swa_k_proj(x.contiguous(), cos, sin, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        options = dict(causal=True, window_size=(self.window_size-1, 0), dropout_p=0.0,
                       softmax_scale=self.value_dim ** -0.5)
        if cu_seqlens is None:
            return flash_attn_func(q, k, v, **options)
        if torch.compiler.is_compiling():
            return _compiled_varlen_swa(q.squeeze(0), k.squeeze(0), v.squeeze(0),
                                        cu_seqlens, max_seqlen, self.window_size).unsqueeze(0)
        return flash_attn_varlen_func(q.squeeze(0), k.squeeze(0), v.squeeze(0),
                                     cu_seqlens, cu_seqlens,
                                     max_seqlen, max_seqlen,
                                     **options).unsqueeze(0)

    def _combine_cuda(self, output, x, v, cu_seqlens, max_seqlen):
        return output + self._swa_cuda(x, v, cu_seqlens, max_seqlen)

    def _combine_torch(self, output, x, cache, previous_extra):
        from .decoding import _readout
        offset = cache.length - x.shape[1]
        cos, sin = self._rotary_tables(x.shape[1], x.device, offset=offset,
                                       dtype=x.dtype, dimension=self.value_dim)
        q = _readout(x, self.swa_q_proj, cos, sin)
        new_k = _readout(x, self.swa_k_proj, cos, sin)
        old_k = previous_extra[0] if previous_extra else new_k[:, :0]
        if old_k.shape != (x.shape[0], min(offset, self.window_size-1), self.heads, self.value_dim):
            raise ValueError("SWA cache shape/window_size changed")
        k = torch.cat((old_k, new_k), dim=1)
        # Reuse DISM's cached V instead of retaining another copy of V history.
        v = cache.v[:, -k.shape[1]:]
        score = torch.einsum('bqhd,bkhd->bhqk', q, k) * self.value_dim ** -0.5
        qi = torch.arange(offset, cache.length, device=x.device)[:, None]
        ki = torch.arange(cache.length-k.shape[1], cache.length, device=x.device)[None, :]
        valid = (ki <= qi) & (ki > qi-self.window_size)
        probability = score.masked_fill(~valid, -torch.inf).softmax(-1)
        local = torch.einsum('bhqk,bkhd->bqhd', probability, v)
        keep = self.window_size-1
        updated = k[:, -keep:].clone() if keep else k[:, :0].clone()
        return output + local, (updated,)
