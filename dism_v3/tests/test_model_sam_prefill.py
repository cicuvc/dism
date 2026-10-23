"""Explicit model SAM routing and packed-document isolation."""
import pytest
import torch
from flash_dism import DismAttention, VarlenLayout
from flash_dism.inference import HardDismPrefill
from flash_dism.inference.model_prefill import use_sam_prefill, vocabulary_prefill
from flash_dism.compiler import voc_forward

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('lengths', [None, (256, 256), (256, 0, 512)])
@pytest.mark.parametrize('precision', ['tf32x3', 'bf16'])
@torch.inference_mode()
def test_packed_core(lengths, precision):
    torch.manual_seed(773)
    b, n, h = (2, 256, 2) if lengths is None else (1, sum(lengths), 2)
    q,k,sq,sk,v = [(torch.randn(b,n,h,d,device='cuda')*.2).bfloat16() for d in (32,32,16,16,32)]
    eq,ek = [torch.randn(h,32,32,device='cuda') for _ in range(2)]
    tau=torch.tensor([0.,.35],device='cuda')
    hard=torch.ones(b,h,device='cuda',dtype=torch.bool)
    direction=torch.ones_like(hard)
    boundaries=None if lengths is None else torch.tensor([0,*torch.tensor(lengths).cumsum(0).tolist()],device='cuda',dtype=torch.int32)
    layout=None if boundaries is None else VarlenLayout.from_cu_seqlens(boundaries,n)
    args=(q,k,sq,sk,v,eq,ek,tau,.7)
    expected=voc_forward(*args,direction,hard,boundaries,direction_true=True)[0]
    with HardDismPrefill(workers=2,mma_precision=precision) as engine:
        actual=vocabulary_prefill(engine,*args,hard,layout=layout)
        # Existing quadratic hard-output criterion; no tolerance changes.
        torch.testing.assert_close(actual,expected,atol=.008,rtol=.025)
        if lengths is not None:
            offset=0
            for length in lengths:
                if length:
                    vectors=[x[:,offset:offset+length].contiguous() for x in (q,k,sq,sk,v)]
                    independent=vocabulary_prefill(engine,*vectors,eq,ek,tau,.7,hard,layout=None)
                    torch.testing.assert_close(actual[:,offset:offset+length],independent,atol=0,rtol=0)
                offset+=length
        with pytest.raises(ValueError,match='all-hard'):
            vocabulary_prefill(engine,*args,torch.zeros_like(hard),layout=layout)


@torch.inference_mode()
def test_model_context_restores_and_keeps_state():
    torch.manual_seed(933)
    model=DismAttention(64,2,head_dim=32,value_dim=32,readout_dim=16,vocab_size=32).cuda().eval()
    x=torch.randn(1,512,64,device='cuda')
    cu=torch.tensor([0,256,512],device='cuda',dtype=torch.int32)
    state={k:v.clone() for k,v in model.state_dict().items()}
    with HardDismPrefill(workers=2,mma_precision='tf32x3') as engine, torch.autocast('cuda',dtype=torch.bfloat16):
        expected=model(x,cu_seqlens=cu,direction_true=True)[0]
        with use_sam_prefill(model,engine):
            actual=model(x,cu_seqlens=cu,direction_true=True)[0]
            torch.testing.assert_close(actual,expected,atol=.008,rtol=.025)
            with pytest.raises(ValueError,match='already active'):
                with use_sam_prefill(model,engine): pass
            with pytest.raises(ValueError,match='fixed-true'):
                model(x,cu_seqlens=cu)
        assert not hasattr(model,'_sam_prefill_engine')
        torch.testing.assert_close(model(x,cu_seqlens=cu,direction_true=True)[0],expected,atol=0,rtol=0)
        with pytest.raises(RuntimeError,match='stop'):
            with use_sam_prefill(model,engine): raise RuntimeError('stop')
        assert not hasattr(model,'_sam_prefill_engine')
    for k,v in model.state_dict().items():
        torch.testing.assert_close(v,state[k],atol=0,rtol=0)


def test_context_rejects_training_and_gradients():
    model=DismAttention(64,2,head_dim=32,value_dim=32,readout_dim=16,vocab_size=32).cuda()
    with pytest.raises(ValueError,match='eval'):
        with use_sam_prefill(model,None): pass
    model.eval()
    with pytest.raises(ValueError,match='gradients'):
        with use_sam_prefill(model,None): pass
