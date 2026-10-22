"""Actual CUDA Graph replay, including mutable packed metadata and RNG state."""
import copy
import pytest
import torch

from nanochat.models import ModelSpec, build_model, PretrainOptimizerConfig
from nanochat.cuda_graph import PackedTrainingGraph

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def model_pair(mxfp8):
    torch.manual_seed(981)
    spec = ModelSpec('dism_gdn_hybrid', 1, dict(
        sequence_len=512, vocab_size=128, n_layer=2, n_head=4,
        n_embd=128, intermediate_size=256, head_dim=32, value_dim=32,
        readout_dim=16, qk_vocab_size=32, anneal_steps=10,
        hard_prob_max=.95, mxfp8=mxfp8, mxfp8_projections=False,
        mxfp8_optimizer=mxfp8,
    ))
    a = build_model(spec).cuda().train()
    a.init_weights()
    b = build_model(spec).cuda().train()
    b.load_state_dict(copy.deepcopy(a.state_dict()))
    return a, b


@pytest.mark.parametrize('mxfp8', [False, True])
def test_packed_replay_matches_training(monkeypatch, mxfp8):
    monkeypatch.setenv('DISM_HARD_PLANNER', '0')
    model, reference = model_pair(mxfp8)
    ids = torch.randint(128, (2, 512), device='cuda')
    labels = torch.randint(128, (2, 512), device='cuda')
    initial = torch.tensor([0,512,1024], dtype=torch.int32, device='cuda')
    graph = PackedTrainingGraph(model, ids, labels, initial, microbatches=2)
    assert model.rng_counter.item() == reference.rng_counter.item() == 0
    configs = ([0,256,512,1024], [0,512,768,1024], [0,256,512,768,1024],
               [0,0,512,512,1024])
    optimizer_config = PretrainOptimizerConfig(.001,.001,.001,.001,.01)
    opt = model.setup_pretraining_optimizer(optimizer_config)
    ref_opt = reference.setup_pretraining_optimizer(optimizer_config)
    cu_storage = torch.empty(6, dtype=torch.int32, device='cuda')
    for step in range(2):
        model.set_training_step(3 + step*6)
        reference.set_training_step(3 + step*6)
        model.zero_grad(set_to_none=True)
        reference.zero_grad(set_to_none=True)
        for micro in range(2):
            bounds = configs[step*2+micro]
            cu = cu_storage[:len(bounds)]
            cu.copy_(torch.tensor(bounds,dtype=torch.int32,device='cuda'))
            ids.random_(128)
            labels.random_(128)
            labels[:, 256:512] = -1
            if step==1 and micro==0:
                labels.fill_(-1)  # No valid documents: no stale loss/gradient contribution.
            got, count = graph.forward_backward(ids, labels, cu)
            # Installed FLA's CPU index builder numbers only nonempty chunks
            # but leaves empty intervals in cu_seqlens. Compare with the
            # mathematically equivalent compact pack, not that NaN-producing
            # upstream path. Empty documents have zero valid CE count.
            reference_cu = torch.unique_consecutive(cu)
            want, expected_count = reference(ids, labels, cu_seqlens=reference_cu, loss_reduction='document_sum')
            (want/2).backward()
            torch.testing.assert_close(got, want, atol=.005, rtol=.02)
            torch.testing.assert_close(count, expected_count, atol=0, rtol=0)
            torch.testing.assert_close(model.rng_counter, reference.rng_counter, atol=0, rtol=0)
        for (name,p), (_,q) in zip(model.named_parameters(),reference.named_parameters()):
            assert (p.grad is None) == (q.grad is None), name
            if p.grad is not None:
                torch.testing.assert_close(p.grad, q.grad, atol=.001, rtol=.03, msg=name)
        opt.step()
        ref_opt.step()
        # Validation uses the original module and can rebind its cached MXFP8
        # plans; subsequent replay must retain its own referenced workspaces.
        model.eval()
        with torch.no_grad():
            model(ids, labels, cu_seqlens=torch.unique_consecutive(cu))
        model.train()
    assert set(model.state_dict()) == set(reference.state_dict())


def test_capture_preserves_resumed_state_and_existing_gradients(monkeypatch):
    monkeypatch.setenv('DISM_HARD_PLANNER', '0')
    model, restored = model_pair(True)
    ids = torch.randint(128,(1,512),device='cuda')
    labels = torch.randint(128,(1,512),device='cuda')
    bounds = torch.tensor([0,256,512],device='cuda',dtype=torch.int32)
    config = PretrainOptimizerConfig(.001,.001,.001,.001,.01)
    optimizer = model.setup_pretraining_optimizer(config)
    model.set_training_step(4)
    model(ids,labels,cu_seqlens=bounds).backward()
    optimizer.step()
    model.zero_grad(set_to_none=True)
    # A resumed model has nonzero RNG/annealing and AdamW state before capture.
    restored.load_state_dict(copy.deepcopy(model.state_dict()))
    restored_optimizer = restored.setup_pretraining_optimizer(config)
    restored_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    for a,b in zip(restored_optimizer.state.values(),optimizer.state.values()):
        assert a['step']==b['step']
        for name in ('exp_avg','exp_avg_sq'):
            assert a[name].dtype==b[name].dtype==torch.bfloat16
            torch.testing.assert_close(a[name],b[name],rtol=0,atol=0)
    model(ids,labels,cu_seqlens=bounds,loss_reduction='document_sum')[0].backward()
    restored(ids,labels,cu_seqlens=bounds,loss_reduction='document_sum')[0].backward()
    before = {n:p.grad.clone() for n,p in restored.named_parameters() if p.grad is not None}
    counter = restored.rng_counter.clone()
    graph = PackedTrainingGraph(restored,ids,labels,bounds)
    torch.testing.assert_close(restored.rng_counter,counter,rtol=0,atol=0)
    for n,p in restored.named_parameters():
        if n in before:
            torch.testing.assert_close(p.grad,before[n],rtol=0,atol=0)
    got,_ = graph.forward_backward(ids,labels,bounds)
    want,_ = model(ids,labels,cu_seqlens=bounds,loss_reduction='document_sum')
    want.backward()
    torch.testing.assert_close(got,want,rtol=.02,atol=.005)
    for (n,p),(_,q) in zip(restored.named_parameters(),model.named_parameters()):
        if p.grad is not None:
            torch.testing.assert_close(p.grad,q.grad,rtol=.03,atol=.001,msg=n)
    restored_optimizer.step()
    optimizer.step()
    for a,b in zip(restored_optimizer.state.values(),optimizer.state.values()):
        assert a.keys() == b.keys()
        assert a['step'] == b['step']


def test_graph_rejects_changed_invalid_boundaries(monkeypatch):
    monkeypatch.setenv('DISM_HARD_PLANNER', '0')
    import cu_flash_dism as native
    table = native.make_graph_layout(torch.tensor([0,512,1024],dtype=torch.int32),1024,512,512)
    before = table.clone()
    for bounds in ([0,255,512,1024], [0,768,1024], [0,256,768,1024], [0,512,512]):
        with pytest.raises(RuntimeError):
            native.update_graph_layout(table,torch.tensor(bounds,dtype=torch.int32))
        torch.testing.assert_close(table,before)
