import pytest
import torch
from nanochat.models import build_model, ModelSpec


def tiny():
    return build_model(ModelSpec('dism_v4_hybrid',1,dict(sequence_len=512,vocab_size=128,
        n_layer=1,n_head=2,n_embd=32,intermediate_size=64,head_dim=32,value_dim=32,
        readout_dim=16,qk_vocab_size=32,anneal_steps=101,hard_prob_max=.95))).cuda().train()


def test_annealed_gate_masks():
    model=tiny();model.init_weights()
    x=torch.randint(128,(2,512),device='cuda')
    cu=torch.tensor([0,512,1024],device='cuda',dtype=torch.int32)
    captured=[]
    handle=model.layers[0].attn.register_forward_pre_hook(
        lambda mod,args,kw:captured.append((kw['hard'].clone(),kw['delta_hard'].clone())),with_kwargs=True)
    for step,p in ((0,0.),(50,.475),(100,.95)):
        model.set_training_step(step)
        with torch.no_grad():model(x,cu_seqlens=cu)
        hard,gate=captured[-1]
        for mask in (hard,gate):assert abs(mask.float().mean().item()-p)<.06
        if step:assert not torch.equal(hard,gate)
    model.eval()
    with torch.no_grad():model(x,cu_seqlens=cu)
    assert all(mask.all() for mask in captured[-1])
    handle.remove()


def test_annealing_without_recompile():
    from torch._dynamo.testing import CompileCounterWithBackend
    torch.cuda.set_per_process_memory_fraction(.08)
    torch._inductor.config.compile_threads=1
    torch._dynamo.config.capture_dynamic_output_shape_ops=True
    torch.fx.experimental._config.use_duck_shape=False
    model=tiny();model.init_weights()
    counter=CompileCounterWithBackend('inductor')
    compiled=torch.compile(model,backend=counter,fullgraph=True,dynamic=True)
    x=torch.randint(128,(1,512),device='cuda');y=torch.randint(128,(1,512),device='cuda')
    cu=torch.tensor([0,256,512],device='cuda',dtype=torch.int32)
    for index,step in enumerate((0,50,100)):
        model.set_training_step(step);model.zero_grad(set_to_none=True)
        with torch._dynamo.config.patch(error_on_recompile=index>0):
            loss=compiled(x,y,cu_seqlens=cu);loss.backward()
        g=model.layers[0].attn.delta_proj.weight.grad
        assert torch.isfinite(loss) and torch.isfinite(g).all() and g.abs().sum()>0
        if not index:frames=counter.frame_count;assert frames>0
        assert counter.frame_count==frames
    # Diagnostic full-hard endpoint: prove the compiled graph reads the updated
    # probability tensor instead of specializing it to the first soft step.
    model.hard_probability.fill_(1.)
    model.zero_grad(set_to_none=True)
    with torch._dynamo.config.patch(error_on_recompile=True):
        loss=compiled(x,y,cu_seqlens=cu);loss.backward()
    assert torch.count_nonzero(model.layers[0].attn.delta_proj.weight.grad)==0
    assert counter.frame_count==frames
