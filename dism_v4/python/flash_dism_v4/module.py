"""Training/evaluation adapter retaining the v3 GDN and output-gate scaffold."""
import torch
from torch.nn import functional as F

from flash_dism.hybrid_gdn import DismGdnAttention
from flash_dism.module import value_residual_mix
from .blocked import dism_v4_blocked_from_embeddings


class V4DismGdnAttention(DismGdnAttention):
    """Full-candidate v4 readout, hard history, no SQ/SK or cached inference."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.sq_proj, self.sk_proj
        self.block_size = 64
        self.readout_precision = "fp32"

    @torch.compiler.disable
    def forward(self, hidden_states, attention_mask=None, past_key_values=None,
                use_cache=False, *, cu_seqlens=None, max_seqlen=None, hard=None,
                direction=None, tau_override=None, v_first=None,
                vocab_temperature=None, conv_seq_idx=None, direction_true=True,
                graph_layout=None, graph_binding=None, gdn_chunk_indices=None,
                _share_branch_boundaries=False):
        if use_cache or past_key_values is not None:
            raise NotImplementedError("v4 prototype has no cache/decode path")
        if graph_layout is not None or graph_binding is not None or gdn_chunk_indices is not None:
            raise ValueError("v4 prototype does not support planner/CUDA graph")
        if not direction_true or (direction is not None and not bool(direction.all())):
            raise ValueError("v4 requires direction=true")
        x = hidden_states
        if not x.is_cuda or x.shape[0] != 1 or cu_seqlens is None or hard is None:
            raise ValueError("v4 model adapter requires packed CUDA input and row-level hard flags")
        if attention_mask is not None:
            raise ValueError("use packed cu_seqlens instead of attention_mask")
        rows, heads = hard.shape
        total = x.shape[1]
        if heads != self.heads or total % rows:
            raise ValueError("hard flags must describe equal-length packed training rows")
        length = total // rows
        if conv_seq_idx is None:
            conv_seq_idx = torch.bucketize(
                torch.arange(total, device=x.device), cu_seqlens[1:-1], right=True)[None]

        def project(conv, channels, linear=None):
            options = {'seq_idx': conv_seq_idx} if getattr(conv, 'supports_seq_idx', False) else {}
            if linear is None:
                y = conv(x, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, **options)
            else:
                y = conv.depthwise_silu(linear, cu_seqlens=cu_seqlens, **options)
            return y.reshape(1, total, self.heads, channels)

        shared = bool(getattr(self, 'share_qk_linear', False))
        linear_q = self.q_conv.project_linear(x) if shared else None
        linear_k = self.k_conv.project_linear(x) if shared else None
        q = project(self.q_conv, self.head_dim, linear_q)
        k = project(self.k_conv, self.head_dim, linear_k)
        v = project(self.v_conv, self.value_dim)
        if self.value_residual and v_first is not None:
            v = value_residual_mix(v, v_first, self.v_residual_gate(x))
        tau = F.softplus(self.log_sel_tau.float()) if tau_override is None else tau_override
        tau = tau * self.tau_scale
        beta = self.vocab_temperature if vocab_temperature is None else vocab_temperature
        # Sampling stays per training row; documents only reset the history.
        output = dism_v4_blocked_from_embeddings(
            q.reshape(rows, length, heads, self.head_dim) * beta,
            k.reshape(rows, length, heads, self.head_dim) * beta,
            self.q_vocab, self.k_vocab,
            v.reshape(rows, length, heads, self.value_dim), tau,
            hard=hard, segment_ids=conv_seq_idx.reshape(rows, length),
            block_size=self.block_size, precision=self.readout_precision,
        ).reshape(1, total, heads, self.value_dim).to(v.dtype)
        output = self._combine_cuda(
            output, x, v, cu_seqlens, max_seqlen, v_first,
            linear_q, linear_k, conv_seq_idx)
        gate = self.g_proj_up(self.g_proj_down(x)).unflatten(-1, (heads, self.value_dim))
        output = self.o_proj(self.norm(output, gate).flatten(2))
        return output, None, None
