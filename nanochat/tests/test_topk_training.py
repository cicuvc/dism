"""Opt-in identity and unchanged native soft-path regression."""
import pytest
import torch
from nanochat.models import ModelSpec, build_model
from nanochat.models.dism import DismConfig


def test_top4_identity_preserves_parameter_schema():
    config=dict(n_layer=2,n_embd=128,n_head=4,head_dim=32,value_dim=32,
                readout_dim=16,vocab_size=128,intermediate_size=256)
    legacy=ModelSpec('dism_gdn_hybrid',1,config)
    selected=ModelSpec('dism_gdn_hybrid',1,{**config,'hard_query_topk':4})
    with torch.device('meta'):
        a,b=build_model(legacy),build_model(selected)
    assert a.model_spec.fingerprint()==legacy.fingerprint()
    assert b.model_spec.fingerprint()!=a.model_spec.fingerprint()
    assert set(a.state_dict())==set(b.state_dict())
    assert [(n,p.shape,p.dtype,getattr(p,'_no_weight_decay',False)) for n,p in a.named_parameters()]==[
        (n,p.shape,p.dtype,getattr(p,'_no_weight_decay',False)) for n,p in b.named_parameters()]
    assert b.layers[1].attn.q_conv.weight is b.layers[1].attn.gdn_q_conv.weight
    for value in (0,2,True,4.0):
        with pytest.raises(ValueError,match='hard_query_topk'):DismConfig(**config,hard_query_topk=value)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA')
def test_top4_all_soft_preserves_native(monkeypatch):
    monkeypatch.setenv('DISM_HARD_PLANNER','0')
    from flash_dism.compiler import voc_forward
    from flash_dism.topk_training import vocabulary
    torch.manual_seed(76)
    b,n,h,d,r,dv,c=2,256,2,32,16,32,32
    q=torch.randn(b,n,h,d,device='cuda',dtype=torch.bfloat16)*.2
    k=torch.randn_like(q)*.2
    sq=torch.randn(b,n,h,r,device='cuda',dtype=torch.bfloat16)*.2;sk=torch.randn_like(sq)*.2
    v=torch.randn(b,n,h,dv,device='cuda',dtype=torch.bfloat16)
    eq=torch.randn(h,c,d,device='cuda')*.2;ek=torch.randn_like(eq)*.2
    tau=torch.tensor([0.,.1],device='cuda')
    args=[t.requires_grad_() for t in (q,k,sq,sk,v,eq,ek,tau)]
    direction=torch.ones(b,h,device='cuda',dtype=torch.bool);hard=torch.zeros_like(direction)
    segments=torch.zeros(b,n,device='cuda',dtype=torch.int32)
    a=voc_forward(*args,1.,direction,hard,None,None,direction_true=True)[0]
    z=vocabulary(*args,direction=direction,hard=hard,segments=segments)
    torch.testing.assert_close(z,a,atol=0,rtol=0)
    g=torch.randn_like(a)
    ga=torch.autograd.grad(a,args,g);gz=torch.autograd.grad(z,args,g)
    for x,y in zip(ga,gz):torch.testing.assert_close(x,y,atol=1e-5,rtol=1e-4)
