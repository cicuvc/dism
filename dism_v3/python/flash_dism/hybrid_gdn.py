"""DISM + gated delta-rule raw outputs, sharing Q/K input-linear parameters."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .module import DismAttention, value_residual_mix
from .kernels.conv1d import CausalShortConv1d

class DismGdnAttention(DismAttention):
    def __init__(self, width, heads, **kwargs):
        super().__init__(width, heads, **kwargs)
        self.gdn_q_conv = CausalShortConv1d(width, heads*self.head_dim, self.q_conv.kernel_size)
        self.gdn_k_conv = CausalShortConv1d(width, heads*self.head_dim, self.k_conv.kernel_size)
        self.gdn_v_conv = CausalShortConv1d(width, heads*self.value_dim, self.v_conv.kernel_size)
        self.gdn_a_proj = nn.Linear(width, heads, bias=False)
        self.gdn_b_proj = nn.Linear(width, heads, bias=False)
        self.gdn_A_log = nn.Parameter(torch.empty(heads, dtype=torch.float32))
        self.gdn_dt_bias = nn.Parameter(torch.empty(heads, dtype=torch.float32))
        self.gdn_A_log._no_weight_decay = True
        self.gdn_dt_bias._no_weight_decay = True
        if self.value_residual:
            self.gdn_v_residual_gate = nn.Linear(width, heads)
            self.gdn_v_residual_gate._value_residual_gate = True
        self.tie_branch_projections()
        self.init_gdn_parameters()

    def tie_branch_projections(self):
        # True Parameter aliasing; conv filters and both V paths remain independent.
        self.gdn_q_conv.weight = self.q_conv.weight
        self.gdn_k_conv.weight = self.k_conv.weight

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        self.tie_branch_projections()
        return result

    def load_state_dict(self, state_dict, strict=True, assign=False):
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self.tie_branch_projections()
        return result

    @torch.no_grad()
    def init_gdn_parameters(self):
        self.gdn_A_log.uniform_(0,16).clamp_min_(1e-6).log_()
        dt=torch.exp(torch.rand_like(self.gdn_dt_bias)*(math.log(.1)-math.log(.001))+math.log(.001))
        self.gdn_dt_bias.copy_(dt+torch.log(-torch.expm1(-dt)))

    @torch.compiler.disable
    def gdn_raw(self,x,cu_seqlens,max_seqlen,v_first=None):
        # Keep FLA metadata/custom autograd outside Dynamo (same policy as controls).
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        q=self.gdn_q_conv(x,cu_seqlens=cu_seqlens,max_seqlen=max_seqlen).unflatten(-1,(self.heads,self.head_dim))
        k=self.gdn_k_conv(x,cu_seqlens=cu_seqlens,max_seqlen=max_seqlen).unflatten(-1,(self.heads,self.head_dim))
        v=self.gdn_v_conv(x,cu_seqlens=cu_seqlens,max_seqlen=max_seqlen).unflatten(-1,(self.heads,self.value_dim))
        if self.value_residual and v_first is not None:
            v = value_residual_mix(v, v_first, self.gdn_v_residual_gate(x))
        beta=self.gdn_b_proj(x).sigmoid()
        g=-self.gdn_A_log.float().exp()*F.softplus(self.gdn_a_proj(x).float()+self.gdn_dt_bias.float())
        return chunk_gated_delta_rule(q=q,k=k,v=v,g=g,beta=beta,
            initial_state=None,output_final_state=False,cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=True)[0]

    def _combine_cuda(self,output,x,v,cu_seqlens,max_seqlen,v_first=None):
        return output+self.gdn_raw(x,cu_seqlens,max_seqlen,v_first)

    def forward(self,hidden_states,attention_mask=None,past_key_values=None,use_cache=False,**kwargs):
        if use_cache or past_key_values is not None:
            raise NotImplementedError('Hybrid GDN currently uses packed prefill/recomputed generation, without a cache')
        if not hidden_states.is_cuda or hidden_states.shape[1]%256:
            raise ValueError('Hybrid GDN requires CUDA and 256-aligned packed input')
        return super().forward(hidden_states,attention_mask=attention_mask,**kwargs)
