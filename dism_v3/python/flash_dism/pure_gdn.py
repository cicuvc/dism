"""GDN-only mixer using exactly the hybrid's GDN frontend and output path."""
import torch
from torch import nn
from torch.nn import functional as F
from .hybrid_gdn import DismGdnAttention
from .kernels.conv1d import CausalShortConv1d

class PureGdnAttention(nn.Module):
    is_pure_gdn=True
    gdn_raw=DismGdnAttention.gdn_raw
    init_gdn_parameters=DismGdnAttention.init_gdn_parameters

    def __init__(self,width,heads,*,head_dim=64,value_dim=64,conv_size=4,**kwargs):
        super().__init__()
        from fla.modules import FusedRMSNormGated
        self.width,self.heads,self.head_dim,self.value_dim=width,heads,head_dim,value_dim
        self.gdn_q_conv=CausalShortConv1d(width,heads*head_dim,conv_size)
        self.gdn_k_conv=CausalShortConv1d(width,heads*head_dim,conv_size)
        self.gdn_v_conv=CausalShortConv1d(width,heads*value_dim,conv_size)
        self.gdn_a_proj=nn.Linear(width,heads,bias=False)
        self.gdn_b_proj=nn.Linear(width,heads,bias=False)
        self.gdn_A_log=nn.Parameter(torch.empty(heads))
        self.gdn_dt_bias=nn.Parameter(torch.empty(heads))
        self.gdn_A_log._no_weight_decay=True;self.gdn_dt_bias._no_weight_decay=True
        self.g_proj_down=nn.Linear(width,max(1,width//8))
        self.g_proj_up=nn.Linear(max(1,width//8),heads*value_dim)
        self.norm=FusedRMSNormGated(value_dim,eps=1e-5)
        self.o_proj=nn.Linear(heads*value_dim,width)
        self.init_gdn_parameters()

    def forward(self,hidden_states,attention_mask=None,past_key_values=None,use_cache=False,**kwargs):
        if attention_mask is not None or past_key_values is not None or use_cache:
            raise NotImplementedError('Packed prefill only')
        x=hidden_states;b,n,_=x.shape
        out=self.gdn_raw(x,kwargs.get('cu_seqlens'),kwargs.get('max_seqlen'))
        gate=self.g_proj_up(self.g_proj_down(x)).reshape(b,n,self.heads,self.value_dim)
        if torch.compiler.is_compiling():
            values=out.float()*torch.rsqrt(out.float().square().mean(-1,keepdim=True)+self.norm.eps)
            out=(values*self.norm.weight.float()*F.silu(gate.float())).to(out.dtype).flatten(2)
        else:out=self.norm(out,gate).flatten(2)
        return self.o_proj(out),None,past_key_values
