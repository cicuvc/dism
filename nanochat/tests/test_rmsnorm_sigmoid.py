import torch
from nanochat.models import ModelSpec, build_model
from nanochat.models.dism.output_gate import RMSNormRawGate, SigmoidInputLinear
import test_cuda_graph as graph_tests


def test_rmsnorm_sigmoid_math_and_all_layers():
    spec=ModelSpec('dism_gdn_hybrid',1,dict(n_layer=2,n_embd=128,n_head=4,head_dim=32,value_dim=32,readout_dim=16,intermediate_size=256,vocab_size=128,mixer_output='rmsnorm_sigmoid',mixer_gate_scales=[.53,.55]))
    model=build_model(spec);model.init_weights()
    for layer in model.layers:
        assert isinstance(layer.attn.norm,RMSNormRawGate)
        assert isinstance(layer.attn.g_proj_up,SigmoidInputLinear)
        x=torch.randn(2,3,4,32)+2;g=torch.randn_like(x)
        ref=torch.nn.functional.rms_norm(x,(32,),layer.attn.norm.weight,layer.attn.norm.eps)*g*layer.attn.norm.scale
        torch.testing.assert_close(layer.attn.norm(x,g),ref)
    model.set_training_step(0);assert model.hard_probability.item()==0
    model.set_training_step(model.config.anneal_steps);assert abs(model.hard_probability.item()-.95)<1e-6


def test_rmsnorm_sigmoid_graph_resume(monkeypatch):
    graph_tests.test_packed_replay_matches_training(monkeypatch,True,False,True,'rmsnorm_sigmoid')
    graph_tests.test_capture_preserves_resumed_state_and_existing_gradients(monkeypatch,True,False,'rmsnorm_sigmoid')
