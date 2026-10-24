from types import SimpleNamespace
import pytest
import torch
from nanochat.models.fla_transformer.yarn import yarn_parameters, YarnRotaryEmbedding
from fla.modules.rotary import RotaryEmbedding


@pytest.mark.parametrize('factor',[1.,2.,4.])
def test_yarn_matches_transformers_reference(factor):
    from transformers.modeling_rope_utils import _compute_yarn_parameters
    config=SimpleNamespace(hidden_size=64,num_attention_heads=1,head_dim=64,
        max_position_embeddings=int(2048*factor),standardize_rope_params=lambda:None,
        rope_parameters=dict(rope_theta=10000.,factor=factor,original_max_position_embeddings=2048,
                             beta_fast=32.,beta_slow=1.,truncate=True))
    ref,amp=_compute_yarn_parameters(config,torch.device('cpu'))
    actual,m=yarn_parameters(64,10000.,2048,factor)
    torch.testing.assert_close(actual,ref,rtol=1e-6,atol=1e-8)
    assert amp==m


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_factor_one_parity_and_packed_position_reset():
    native=RotaryEmbedding(64,device='cuda').eval()
    yarn=YarnRotaryEmbedding(native,2048,1.)
    q=torch.randn(1,512,2,64,device='cuda',dtype=torch.bfloat16)
    k=torch.randn_like(q);cu=torch.tensor([0,256,512],device='cuda',dtype=torch.int32)
    with torch.no_grad():
        a=native(q,k,cu_seqlens=cu,max_seqlen=512)
        b=yarn(q,k,cu_seqlens=cu,max_seqlen=512)
        for x,y in zip(a,b):torch.testing.assert_close(x,y,rtol=0,atol=0)
        y4=YarnRotaryEmbedding(native,2048,4.)
        both=y4(q,k,cu_seqlens=cu,max_seqlen=512)
        for start in [0,256]:
            alone=y4(q[:,start:start+256].contiguous(),k[:,start:start+256].contiguous(),
                     cu_seqlens=torch.tensor([0,256],device='cuda',dtype=torch.int32),max_seqlen=256)
            for x,y in zip(both,alone):torch.testing.assert_close(x[:,start:start+256],y,rtol=0,atol=0)
        y4._update_cos_sin_cache(8192,device=torch.device('cuda',0),dtype=torch.bfloat16)
        expected=(y4.inv_freq.float().cos()*y4.attention_factor).to(torch.bfloat16)
        torch.testing.assert_close(y4._cos_cached[1],expected,rtol=0,atol=0)
