import math
import pytest
import torch
from dism_v2.lm_model import LMConfig,DecoderLM,parameter_count,vocab_balance_loss


def test_uniform():
    q=torch.zeros(2,2,32,4,requires_grad=True)
    e=torch.randn(2,8,4,requires_grad=True)
    loss=vocab_balance_loss(q,q,e,e)
    torch.testing.assert_close(loss,torch.zeros_like(loss),atol=1e-7,rtol=0)
    loss.backward()
    assert torch.isfinite(q.grad).all()


def test_against_explicit_samples_and_gradients():
    torch.manual_seed(4)
    xs=[torch.randn(2,2,32,4,requires_grad=True) for _ in range(2)]
    es=[torch.randn(2,8,4,requires_grad=True) for _ in range(2)]
    actual=vocab_balance_loss(*xs,*es)
    refs=[]
    for z,e in zip(xs,es):
        sample=torch.stack([z[0,:,0::16],z[1,:,1::16]])
        logits=torch.einsum('bhsd,hvd->bhsv',sample,e)
        marginal=logits.softmax(-1).mean((0,2))
        refs.append((marginal*(marginal.log()+math.log(8))).sum(-1).mean())
    expected=torch.stack(refs).mean()
    torch.testing.assert_close(actual,expected)
    g0=torch.autograd.grad(actual,xs+es,retain_graph=True)
    g1=torch.autograd.grad(expected,xs+es)
    for a,b in zip(g0,g1):torch.testing.assert_close(a,b)
    assert 0<actual<math.log(8)


def test_same_initial_weights():
    cfg=dict(layers=1,heads=1,width=64,vocab_size=128,ffn_hidden=64,softcap=30.)
    torch.manual_seed(777);base=DecoderLM(LMConfig(**cfg))
    torch.manual_seed(777);variant=DecoderLM(LMConfig(**cfg,vocab_balance_weight=.01))
    for name,value in base.state_dict().items():
        torch.testing.assert_close(value,variant.state_dict()[name],atol=0,rtol=0)
    assert parameter_count(LMConfig(vocab_balance_weight=.01))==49678876


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('prob',[0.,.5,1.])
def test_model_gradients_and_eval_ce_only(prob):
    torch.manual_seed(777)
    model=DecoderLM(LMConfig(layers=1,heads=1,width=64,ffn_hidden=64,context=64,
                             softcap=30.,vocab_balance_weight=.01)).cuda()
    x=torch.randint(50257,(2,64),device='cuda')
    with torch.autocast('cuda',dtype=torch.bfloat16):
        loss=model(x,x,prob,torch.Generator(device='cuda').manual_seed(778))
    torch.testing.assert_close(loss,model.last_ce+.01*model.last_balance)
    assert model.last_balance>0
    loss.backward()
    for name,param in model.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all(),name
    # Full hard core gives no vocab gradient, but independent soft auxiliary does.
    assert model.blocks[0].dism.q_voc.grad.abs().sum()>0
    model.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        evaluation=model(x,x,prob,torch.Generator(device='cuda').manual_seed(778))
    assert model.last_balance==0
    torch.testing.assert_close(evaluation,model.last_ce,atol=0,rtol=0)
