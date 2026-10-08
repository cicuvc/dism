"""Learned gate parameterization, hardening, compiler and cache contracts."""
import pytest
import torch
from torch.nn import functional as F
from flash_dism import (DismAttention,DismSwaAttention,DismV4Attention,
                        DismV4SwaAttention,DismConfig,DismForCausalLM)
from gradient_acceptance import assert_gradient

torch.set_num_threads(1)


def layer(cls=DismV4Attention,**kwargs):
    return cls(32,2,head_dim=32,value_dim=32,readout_dim=16,vocab_size=32,layer_idx=0,**kwargs)


def test_gate_sign_and_gradient():
    model=layer().train()
    with torch.no_grad():
        model.delta_proj.weight.zero_();model.delta_proj.bias.zero_()
        model.delta_proj.weight[0,0]=1;model.delta_proj.weight[1,1]=1
    x=torch.zeros(1,4,32,requires_grad=True)
    with torch.no_grad():x[0,:,:2]=torch.tensor([[-2.,3.],[0.,-1.],[2.,0.],[-3.,4.]])
    mask=torch.tensor([[[True,True,False,False],[False,True,True,False]]])
    result=model.gate_delta(x,delta_hard=mask)
    z=x[:,:,:2].transpose(1,2)
    expected=torch.where(mask,torch.where(z>=0,0.,1e6),torch.log1p(torch.exp(-z)))
    torch.testing.assert_close(result,expected)
    result.sum().backward()
    expected_dx=torch.where(mask,0.,-torch.sigmoid(-z)).transpose(1,2)
    torch.testing.assert_close(x.grad[:,:,:2],expected_dx)
    assert result.dtype==torch.float32 and result.is_contiguous()


@pytest.mark.parametrize('prob',[0.,.35,1.])
def test_probability_and_seed(prob):
    model=layer().train();x=torch.randn(2,1024,32)
    a=model.gate_delta(x,prob,hard_seed=144)
    b=model.gate_delta(x,prob,hard_seed=144)
    torch.testing.assert_close(a,b,atol=0,rtol=0)
    mask=(a==0)|(a==1e6)
    assert abs(mask.float().mean().item()-prob)<.04
    if 0<prob<1:
        assert not torch.equal(a,model.gate_delta(x,prob,hard_seed=145))
    model.eval();actual=model.gate_delta(x)
    assert ((actual==0)|(actual==1e6)).all()
    model.train();torch.testing.assert_close(model.gate_delta(x),-F.logsigmoid(model.delta_proj(x).float()).transpose(1,2))


@pytest.mark.parametrize('cls',[DismV4Attention,DismV4SwaAttention])
@pytest.mark.parametrize('packed',[False,True])
@pytest.mark.parametrize('prob',[0.,.5,1.])
def test_cuda_layers(cls,packed,prob):
    torch.manual_seed(921)
    model=layer(cls).cuda().train()
    base=layer(DismSwaAttention if cls is DismV4SwaAttention else DismAttention).cuda().train()
    base.load_state_dict({k:v for k,v in model.state_dict().items() if not k.startswith('delta_proj.')})
    x=torch.randn(1,512,32,device='cuda')
    mask=torch.rand(1,2,512,device='cuda')<prob
    opts=dict(hard_prob=prob,hard_seed=712,direction=torch.tensor([[True,False]],device='cuda'))
    if packed:opts.update(cu_seqlens=torch.tensor([0,256,512],device='cuda',dtype=torch.int32),max_seqlen=256)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        actual=model(x,delta_hard=mask,**opts)[0]
        z=F.linear(x,model.delta_proj.weight,model.delta_proj.bias).float().transpose(1,2)
        delta=torch.where(mask,torch.where(z>=0,0.,1e6),-F.logsigmoid(z)).contiguous()
        expected=base(x,gate_delta=delta,**opts)[0]
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    actual.float().square().mean().backward()
    assert torch.isfinite(model.delta_proj.weight.grad).all()
    assert (model.delta_proj.weight.grad.abs().sum()>0).item()==(prob<1)


@pytest.mark.parametrize('cls',[DismV4Attention,DismV4SwaAttention])
@pytest.mark.parametrize('packed',[False,True])
def test_compile_and_optimizer(cls,packed):
    torch.cuda.set_per_process_memory_fraction(.08)
    torch._inductor.config.compile_threads=1
    torch._dynamo.config.capture_dynamic_output_shape_ops=True
    torch.fx.experimental._config.use_duck_shape=False
    torch.manual_seed(922);model=layer(cls).cuda().train()
    compiled=torch.compile(model,fullgraph=True,dynamic=True)
    x=torch.randn(1,512,32,device='cuda')
    opts=dict(hard_prob=.5,hard_seed=714,direction=torch.tensor([[True,False]],device='cuda'))
    if packed:opts.update(cu_seqlens=torch.tensor([0,256,512],device='cuda',dtype=torch.int32),max_seqlen=256)
    with torch.autocast('cuda',dtype=torch.bfloat16):expected=model(x,**opts)[0]
    expected.float().square().mean().backward();grad=[p.grad.clone() for p in model.parameters()]
    model.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):actual=compiled(x,**opts)[0]
    actual.float().square().mean().backward()
    torch.testing.assert_close(actual,expected,atol=.005,rtol=.03)
    for p,g in zip(model.parameters(),grad):assert_gradient(p.grad,g,'parameter')
    previous=model.delta_proj.weight.detach().clone()
    torch.optim.AdamW(model.parameters(),lr=.001).step()
    assert not torch.equal(previous,model.delta_proj.weight)


@pytest.mark.parametrize('kind',['dism_v4','hybrid_v4'])
def test_model_cache_and_serialization(kind,tmp_path):
    torch.manual_seed(923)
    config=DismConfig(vocab_size=32,hidden_size=32,num_hidden_layers=1,num_heads=2,
        head_dim=32,value_dim=32,readout_dim=16,qk_vocab_size=32,intermediate_size=64,
        attention_type=kind,bos_token_id=1,eos_token_id=2)
    model=DismForCausalLM(config).eval()
    x=torch.randint(0,32,(1,9));opts=dict(direction=torch.tensor([[True,False]]))
    with torch.no_grad():
        whole=model(x,use_cache=False,**opts).logits
        first=model(x[:,:4],**opts)
        last=model(x[:,4:],past_key_values=first.past_key_values,**opts)
        torch.testing.assert_close(torch.cat([first.logits,last.logits],1),whole,atol=2e-5,rtol=2e-4)
    model.save_pretrained(tmp_path);restored=DismForCausalLM.from_pretrained(tmp_path).eval()
    with torch.no_grad():torch.testing.assert_close(restored(x,use_cache=False,**opts).logits,whole,atol=0,rtol=0)
