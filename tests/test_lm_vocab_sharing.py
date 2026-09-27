"""Shared/group codebook contracts and true CUDA branch-gradient accumulation."""
from dataclasses import replace
import pytest
import torch
from torch.nn import functional as F

from dism_v2.lm_model import LMConfig, DismAttention, DecoderLM, parameter_groups, parameter_count, matched_swa_config


@pytest.mark.parametrize('groups', [1, 2, 4])
@pytest.mark.parametrize('tie', [False, True])
def test_parameters_state_and_mapping(groups, tie):
    cfg=LMConfig(dism_vocab_groups=groups,dism_tie_qk_vocab=tie)
    module=DismAttention(cfg)
    assert module.q_voc.shape==(groups,512,64)
    assert (module.k_voc is None)==tie
    q,k=module.expanded_vocabularies(torch.float32)
    for head in range(4):
        torch.testing.assert_close(q[head],module.q_voc[head//(4//groups)],atol=0,rtol=0)
        torch.testing.assert_close(k[head],(module.q_voc if tie else module.k_voc)[head//(4//groups)],atol=0,rtol=0)
    owner=[p for n,p in module.named_parameters() if n in ('q_voc','k_voc')]
    assert len(owner)==(1 if tie else 2)
    params=parameter_groups(module,.01)
    assert len({id(p) for g in params for p in g['params']})==sum(len(g['params']) for g in params)
    assert all(any(p is x for x in params[1]['params']) for p in owner)
    state=module.state_dict()
    assert ('k_voc' not in state)==tie
    restored=DismAttention(cfg)
    restored.load_state_dict(state,strict=True)
    assert restored.k_voc is None if tie else restored.k_voc is not None
    expected=49_678_876 -15*(8-(1 if tie else 2)*groups)*512*64
    assert parameter_count(cfg)==expected
    control=matched_swa_config(cfg)
    assert control.dism_vocab_groups is None and not control.dism_tie_qk_vocab


def test_default_and_invalid_config():
    cfg=LMConfig()
    assert parameter_count(cfg)==49_678_876
    with torch.device('meta'):
        old=DecoderLM(cfg)
        assert old.blocks[0].dism.q_voc.shape==(4,512,64)
        assert 'blocks.0.dism.k_voc' in old.state_dict()
    for group in (0,3,5,-1,True):
        with pytest.raises(ValueError):
            DismAttention(replace(cfg,dism_vocab_groups=group))
    with pytest.raises(ValueError):
        DecoderLM(replace(cfg,architecture='swa_only',dism_tie_qk_vocab=True))
    # No silent migration of independent Q/K checkpoints into a tied model.
    with pytest.raises(RuntimeError):
        DismAttention(replace(cfg,dism_tie_qk_vocab=True)).load_state_dict(DismAttention(cfg).state_dict())


@pytest.mark.parametrize('groups', [1,2,4])
@pytest.mark.parametrize('tie', [False,True])
def test_fp32_accumulation_after_separate_casts(groups,tie):
    module=DismAttention(LMConfig(qk_vocab=64,dism_vocab_groups=groups,dism_tie_qk_vocab=tie))
    q,k=module.expanded_vocabularies(torch.bfloat16)
    q.retain_grad();k.retain_grad()
    assert q is not k
    torch.manual_seed(741)
    (q.float()*torch.randn_like(q.float())+k.float()*torch.randn_like(k.float())).sum().backward()
    dq=q.grad.float().reshape(groups,4//groups,64,64).sum(1)
    dk=k.grad.float().reshape(groups,4//groups,64,64).sum(1)
    torch.testing.assert_close(module.q_voc.grad,dq+dk if tie else dq,rtol=0,atol=0)
    if not tie:torch.testing.assert_close(module.k_voc.grad,dk,rtol=0,atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('groups', [1,2,4])
@pytest.mark.parametrize('tie', [False,True])
@pytest.mark.parametrize('direction', ['q_from_k','k_from_q'])
@pytest.mark.parametrize('probability', [0.,.37,1.])
def test_cuda_branch_gradient_sum(groups,tie,direction,probability):
    from dism_v2.autograd import voc_dism
    torch.manual_seed(781)
    module=DismAttention(LMConfig(qk_vocab=64,dism_vocab_groups=groups,dism_tie_qk_vocab=tie)).cuda()
    qvoc,kvoc=module.expanded_vocabularies(torch.bfloat16)
    qvoc.retain_grad();kvoc.retain_grad()
    q,k,v=[torch.randn(1,4,129,64,device='cuda',dtype=torch.bfloat16,requires_grad=True) for _ in range(3)]
    tau=torch.full((4,),.3,device='cuda',requires_grad=True)
    out=voc_dism(q,k,v,tau,qvoc,kvoc,hard_prob=probability,direction=direction,
                 generator=torch.Generator(device='cuda').manual_seed(19),
                 embedding_backend='cuda',embedding_backward_backend='cuda')
    weight=torch.randn_like(out)
    (out.float()*weight.float()).mean().backward()
    dq=qvoc.grad.float().reshape(groups,4//groups,64,64).sum(1)
    dk=kvoc.grad.float().reshape(groups,4//groups,64,64).sum(1)
    torch.testing.assert_close(module.q_voc.grad,dq+dk if tie else dq,rtol=0,atol=1e-9)
    if not tie:torch.testing.assert_close(module.k_voc.grad,dk,rtol=0,atol=1e-9)
    for tensor in (q,k,v,tau,module.q_voc):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
    if probability<1:
        assert module.q_voc.grad.abs().sum()>0
    else:
        assert torch.count_nonzero(module.q_voc.grad)==0  # Hard argmax has no derivative.
    # Identical values, but all head/branch vocabulary parameters independent.
    uq=qvoc.detach().float().requires_grad_();uk=kvoc.detach().float().requires_grad_()
    replay=voc_dism(q.detach(),k.detach(),v.detach(),tau.detach(),uq.to(torch.bfloat16),uk.to(torch.bfloat16),
                   hard_prob=probability,direction=direction,
                   generator=torch.Generator(device='cuda').manual_seed(19),
                   embedding_backend='cuda',embedding_backward_backend='cuda')
    torch.testing.assert_close(out,replay,rtol=0,atol=0)
    (replay.float()*weight.float()).mean().backward()
    # Independent kernel launches may differ at BF16 rounding boundaries after atomics.
    torch.testing.assert_close(uq.grad,qvoc.grad.float(),rtol=.02,atol=2e-7)
    torch.testing.assert_close(uk.grad,kvoc.grad.float(),rtol=.02,atol=2e-7)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('architecture', ['hybrid','hybrid_shared'])
@pytest.mark.parametrize('groups', [1,2,4])
@pytest.mark.parametrize('probability', [.37,1.])
def test_model_cuda(architecture,groups,probability):
    torch.manual_seed(92)
    cfg=LMConfig(architecture=architecture,layers=1,vocab_size=128,ffn_hidden=128,
                 context=256,softcap=30.,dism_tie_qk_vocab=True,dism_vocab_groups=groups)
    model=DecoderLM(cfg).cuda()
    x=torch.randint(128,(1,129),device='cuda')
    target=torch.randint(128,x.shape,device='cuda')
    with torch.autocast('cuda',dtype=torch.bfloat16):
        h=model.forward_features(x,probability,torch.Generator(device='cuda').manual_seed(54))
        logits=model.lm_head(h).float()
    F.cross_entropy((30*(logits/30).tanh()).flatten(0,1),target.flatten()).backward()
    for name,p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(),name
    if probability<1:assert model.blocks[0].dism.q_voc.grad.abs().sum()>0
    assert model.embedding.weight is not model.lm_head.weight


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('architecture', ['hybrid','hybrid_shared'])
def test_grouped_model_matches_explicit_independent_copy(architecture):
    torch.manual_seed(155)
    cfg=LMConfig(architecture=architecture,layers=1,vocab_size=128,ffn_hidden=128,
                 context=256,softcap=30.,dism_tie_qk_vocab=True,dism_vocab_groups=2)
    shared=DecoderLM(cfg).cuda()
    independent=DecoderLM(replace(cfg,dism_tie_qk_vocab=False,dism_vocab_groups=None)).cuda()
    state=dict(shared.state_dict())
    vocab=state['blocks.0.dism.q_voc'].repeat_interleave(2,0)
    state['blocks.0.dism.q_voc']=vocab
    state['blocks.0.dism.k_voc']=vocab.clone()
    independent.load_state_dict(state,strict=True)
    x=torch.randint(128,(1,129),device='cuda')
    upstream=torch.randn(1,129,128,device='cuda')
    outputs=[]
    for model in (shared,independent):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            h=model.forward_features(x,.37,torch.Generator(device='cuda').manual_seed(832))
            y=model.lm_head(h)
        outputs.append(y.detach())
        (y.float()*upstream).mean().backward()
    torch.testing.assert_close(*outputs,atol=0,rtol=0)
    params=dict(independent.named_parameters())
    for name,p in shared.named_parameters():
        if name=='blocks.0.dism.q_voc':
            expected=(params[name].grad.reshape(2,2,512,64).sum(1)+
                      params['blocks.0.dism.k_voc'].grad.reshape(2,2,512,64).sum(1))
        else:expected=params[name].grad
        torch.testing.assert_close(p.grad,expected,atol=2e-7,rtol=.02,msg=name)
