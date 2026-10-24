"""Packed boundaries, upstream attention parity and full training replay."""
import copy
import pytest
import torch
from nanochat.models import ModelSpec, build_model, PretrainOptimizerConfig
from nanochat.attention_cuda_graph import PackedAttentionTrainingGraph


def make_spec(ce=True):
    return ModelSpec('fla_transformer',1,dict(n_layer=2,n_head=4,n_embd=128,
        intermediate_size=256,vocab_size=128,sequence_len=512,mxfp8_ce=ce))


def test_parameter_budget():
    with torch.device('meta'):
        model=build_model(ModelSpec('fla_transformer',1,{}))
    count=sum(p.numel() for p in model.parameters())
    assert abs(count-123473088)/123473088 < .0002


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_upstream_attention_and_document_isolation():
    from fla.layers.attn import Attention
    model=build_model(make_spec()).cuda();model.init_weights()
    attn=model.layers[0].attn
    x=torch.randn(1,512,128,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    cu=torch.tensor([0,256,512],device='cuda',dtype=torch.int32)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        ours=attn(x,cu_seqlens=cu,max_seqlen=512)
        upstream=Attention.forward(attn,x,cu_seqlens=cu)[0]
    torch.testing.assert_close(ours,upstream,atol=1e-3,rtol=1e-3)
    dy=torch.randn_like(ours)
    ga=torch.autograd.grad(ours,x,dy,retain_graph=True)[0]
    gb=torch.autograd.grad(upstream,x,dy)[0]
    torch.testing.assert_close(ga,gb,atol=1e-3,rtol=1e-3)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        changed=x.detach().clone();changed[:,:256].normal_()
        out=attn(changed,cu_seqlens=cu,max_seqlen=512)
        torch.testing.assert_close(out[:,256:],ours[:,256:],atol=0,rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('ce',[False,True])
def test_graph_boundaries_updates_and_eval(ce):
    torch.manual_seed(13)
    model=build_model(make_spec(ce)).cuda().train();model.init_weights()
    ref=build_model(make_spec(ce)).cuda().train();ref.load_state_dict(copy.deepcopy(model.state_dict()))
    ids=torch.randint(128,(2,512),device='cuda');labels=torch.randint(128,ids.shape,device='cuda')
    cu=torch.tensor([0,512,1024],device='cuda',dtype=torch.int32)
    graph=PackedAttentionTrainingGraph(model,ids,labels,cu,microbatches=2)
    cfg=PretrainOptimizerConfig(.001,.001,.001,.001,.01)
    opt=model.setup_pretraining_optimizer(cfg);refopt=ref.setup_pretraining_optimizer(cfg)
    try:
        for step in range(2):
            model.zero_grad(set_to_none=step==0);ref.zero_grad(set_to_none=step==0)
            for micro,bounds in enumerate(([0,256,512,1024],[0,0,512,768,1024])):
                cu=torch.tensor(bounds,device='cuda',dtype=torch.int32)
                labels.random_(128)
                if step==1 and micro==0:labels.fill_(-1)
                got,count=graph.forward_backward(ids,labels,cu)
                want,wcount=ref(ids,labels,cu_seqlens=torch.unique_consecutive(cu),loss_reduction='document_sum')
                (want/2).backward()
                torch.testing.assert_close(count,wcount,atol=0,rtol=0)
                torch.testing.assert_close(got,want,atol=.005,rtol=.005)
                del want,wcount
            for (name,a),(_,b) in zip(model.named_parameters(),ref.named_parameters()):
                assert (a.grad is None)==(b.grad is None),name
                if a.grad is not None:
                    assert torch.isfinite(a.grad).all(),name
                    torch.testing.assert_close(a.grad,b.grad,atol=.001,rtol=.03,msg=name)
            opt.step();refopt.step()
            model.eval()
            model.config.sequence_len=1024
            with torch.no_grad():
                model(torch.cat((ids,ids),1),torch.cat((labels,labels),1),
                    cu_seqlens=torch.tensor([0,1024,2048],device='cuda',dtype=torch.int32),loss_reduction='none')
            model.config.sequence_len=512
            model.train()
        # Restoring both model and optimizer must retain their complete schemas.
        restored=build_model(make_spec(ce)).cuda().train();restored.load_state_dict(copy.deepcopy(model.state_dict()))
        restored_opt=restored.setup_pretraining_optimizer(cfg)
        restored_opt.load_state_dict(copy.deepcopy(opt.state_dict()))
        assert set(restored.state_dict())==set(model.state_dict())
        assert len(restored_opt.state)==len(opt.state)
        model.zero_grad(set_to_none=True)
        restored.zero_grad(set_to_none=True)
        graph.forward_backward(ids,labels,cu)
        loss,_=restored(ids,labels,cu_seqlens=torch.unique_consecutive(cu),loss_reduction='document_sum')
        (loss/2).backward()
        del loss
        opt.step();restored_opt.step()
        for (name,a),(_,b) in zip(model.named_parameters(),restored.named_parameters()):
            torch.testing.assert_close(a,b,atol=2e-5,rtol=.01,msg=name)
    finally:graph.close()
