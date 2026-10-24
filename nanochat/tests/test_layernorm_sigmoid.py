import pytest
import torch
from nanochat.models import ModelSpec, build_model
from nanochat.models.dism import DismConfig
from nanochat.models.dism.output_gate import LayerNormRawGate, SigmoidInputLinear
import test_cuda_graph as graph_tests


def test_config_and_identity():
    with pytest.raises(ValueError,match='mixer_gate_scale'):
        DismConfig(mixer_output='layernorm_sigmoid')
    with pytest.raises(ValueError,match='mixer_gate_scale'):
        DismConfig(mixer_output='layernorm_sigmoid',mixer_gate_scales=[float('nan')]*12)
    config=dict(n_layer=2,n_embd=128,n_head=4,head_dim=32,value_dim=32,readout_dim=16,intermediate_size=256,vocab_size=128)
    old=ModelSpec('dism_gdn_hybrid',1,config)
    new=ModelSpec('dism_gdn_hybrid',1,dict(config,mixer_output='layernorm_sigmoid',mixer_gate_scales=[.53,.55]))
    with torch.device('meta'):
        a,b=build_model(old),build_model(new)
    assert old.fingerprint()!=new.fingerprint()
    assert list(a.state_dict())==list(b.state_dict())
    assert [n for n,_ in a.named_parameters()]==[n for n,_ in b.named_parameters()]


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_initialization_and_gate_math():
    config=dict(n_layer=2,n_embd=128,n_head=4,head_dim=32,value_dim=32,readout_dim=16,intermediate_size=256,vocab_size=128)
    models=[]
    for extra in ({},dict(mixer_output='layernorm_sigmoid',mixer_gate_scales=[.53,.55])):
        torch.manual_seed(42)
        with torch.device('meta'):m=build_model(ModelSpec('dism_gdn_hybrid',1,dict(config,**extra)))
        m.to_empty(device='cuda');m.init_weights();models.append(m)
    for (name,a),(_,b) in zip(models[0].state_dict().items(),models[1].state_dict().items()):
        torch.testing.assert_close(a,b,rtol=0,atol=0,msg=name)
    attn=models[1].layers[1].attn
    assert isinstance(attn.g_proj_up,SigmoidInputLinear)
    assert isinstance(attn.norm,LayerNormRawGate)
    x=torch.randn(1,256,128,device='cuda');out=torch.randn(1,256,4,32,device='cuda')
    with torch.autocast('cuda',dtype=torch.bfloat16):
        low=attn.g_proj_down(x)
        gate=attn.g_proj_up(low).unflatten(-1,(4,32))
        expected_gate=torch.nn.functional.linear(low.sigmoid(),attn.g_proj_up.weight,attn.g_proj_up.bias).unflatten(-1,(4,32))
        torch.testing.assert_close(gate,expected_gate,rtol=0,atol=0)
        ref=torch.nn.functional.layer_norm(out,(32,),attn.norm.weight,None,attn.norm.eps)*gate.float()*.55
        torch.testing.assert_close(attn.norm(out,gate),ref,rtol=1e-5,atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_planner_graph_and_resume(monkeypatch):
    graph_tests.test_packed_replay_matches_training(monkeypatch,True,False,True,'layernorm_sigmoid')
    graph_tests.test_capture_preserves_resumed_state_and_existing_gradients(monkeypatch,True,False,'layernorm_sigmoid')
