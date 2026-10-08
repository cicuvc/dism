"""Vocabulary, compiler, module and checkpoint coverage for v4 attenuation."""
import io
import pytest
import torch
from flash_dism import voc_dism, DismSwaAttention
from flash_dism.reference.dism_v4_ref import dism_ref
from gradient_acceptance import assert_gradient
from test_v4_cuda import sample


def vocabulary_reference(q,k,sq,sk,v,eq,ek,tau,gate,direction,hard):
    eq,ek=eq.bfloat16().float(),ek.bfloat16().float()
    qs=torch.einsum('bnhd,hvd->bhnv',q.float(),eq)
    ks=torch.einsum('bnhd,hvd->bhnv',k.float(),ek)
    qfk=torch.einsum('bhnv,hvd->bnhd',ks.softmax(-1),eq).bfloat16()
    kfq=torch.einsum('bhnv,hvd->bnhd',qs.softmax(-1),ek).bfloat16()
    select=direction[:,None,:,None]
    return dism_ref(torch.where(select,q,kfq),torch.where(select,qfk,k),sq,sk,
        qs.logsumexp(-1).transpose(1,2),ks.logsumexp(-1).transpose(1,2),
        qs.argmax(-1),ks.argmax(-1),direction,hard,gate,v,tau)


@pytest.mark.parametrize('mode',['soft','mixed','hard'])
@pytest.mark.parametrize('packed',[False,True])
@pytest.mark.parametrize('direction_value',[False,True])
def test_vocabulary_gradients(mode,packed,direction_value):
    x,delta=sample(mode=mode)
    torch.manual_seed(163)
    args=x[:5]+[torch.randn(2,512,64)*.2,torch.randn(2,512,64)*.2,x[-1],delta]
    gpu=[a.cuda().requires_grad_() for a in args]
    ref=[a.clone().requires_grad_() for a in args]
    direction=torch.full_like(x[9],direction_value);hard=x[10]
    cu=torch.tensor([0,256],device='cuda',dtype=torch.int32) if packed else None
    actual=voc_dism(*gpu[:8],direction=direction.cuda(),hard=hard.cuda(),gate_delta=gpu[-1],cu_seqlens=cu)
    expected=vocabulary_reference(*ref,direction,hard)
    do=torch.randn_like(x[4])
    ga=torch.autograd.grad(actual,gpu,do.cuda());ge=torch.autograd.grad(expected,ref,do.float())
    torch.testing.assert_close(actual.cpu().float(),expected,atol=.003,rtol=.03)
    for name,a,b in zip(('q','k','sq','sk','v','eq','ek','tau','gate_delta'),ga,ge):assert_gradient(a.cpu(),b,name)


class GatedHybrid(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attention=DismSwaAttention(64,2,head_dim=64,value_dim=64,readout_dim=32,vocab_size=512)
        self.gate=torch.nn.Linear(64,2)
    def forward(self,x,hard,direction,cu):
        delta=torch.nn.functional.softplus(-self.gate(x).float()).transpose(1,2).contiguous()
        return self.attention(x,hard=hard,direction=direction,gate_delta=delta,
            cu_seqlens=cu,max_seqlen=512 if cu is not None else None)[0]


@pytest.mark.parametrize('packed',[False,True])
def test_compiled_module_optimizer_checkpoint(packed):
    torch.manual_seed(773)
    torch.cuda.set_per_process_memory_fraction(.08)
    torch._inductor.config.compile_threads=1
    torch._dynamo.config.capture_dynamic_output_shape_ops=True
    torch.fx.experimental._config.use_duck_shape=False
    model=GatedHybrid().cuda().train()
    compiled=torch.compile(model,fullgraph=True,dynamic=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    x=torch.randn(1,512,64,device='cuda')
    hard=torch.rand(1,2,512,device='cuda')<.5
    direction=torch.tensor([[True,False]],device='cuda')
    cu=torch.tensor([0,256,512],device='cuda',dtype=torch.int32) if packed else None
    with torch.autocast('cuda',dtype=torch.bfloat16):expected=model(x,hard,direction,cu)
    expected.float().square().mean().backward()
    eager=[p.grad.clone() for p in model.parameters()]
    model.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):actual=compiled(x,hard,direction,cu)
    actual.float().square().mean().backward()
    torch.testing.assert_close(actual,expected,atol=.005,rtol=.03)
    for p,g in zip(model.parameters(),eager):assert_gradient(p.grad,g,'module_parameter')
    assert model.gate.weight.grad.abs().sum()>0
    optimizer.step();model.zero_grad(set_to_none=True)
    checkpoint=io.BytesIO();torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict()},checkpoint)
    checkpoint.seek(0);saved=torch.load(checkpoint,weights_only=True)
    restored=GatedHybrid().cuda().train();restored.load_state_dict(saved['model'])
    restored_optimizer=torch.optim.AdamW(restored.parameters(),lr=.001);restored_optimizer.load_state_dict(saved['optimizer'])
    restored_compiled=torch.compile(restored,fullgraph=True,dynamic=True)
    for active,opt in ((compiled,optimizer),(restored_compiled,restored_optimizer)):
        with torch.autocast('cuda',dtype=torch.bfloat16):out=active(x,hard,direction,cu)
        loss=out.float().square().mean();assert torch.isfinite(loss)
        loss.backward();opt.step();opt.zero_grad(set_to_none=True)
    for a,b in zip(model.parameters(),restored.parameters()):torch.testing.assert_close(a,b,atol=1e-5,rtol=1e-4)


def test_compiled_dynamic_documents():
    from torch._dynamo.testing import CompileCounterWithBackend
    from flash_dism.kernels.dynamo_utils import mark_cu_seqlens_dynamic
    torch.manual_seed(774)
    torch.cuda.set_per_process_memory_fraction(.08)
    torch._inductor.config.compile_threads=1
    torch._dynamo.config.capture_dynamic_output_shape_ops=True
    torch.fx.experimental._config.use_duck_shape=False
    model=GatedHybrid().cuda().train()
    counter=CompileCounterWithBackend('inductor')
    compiled=torch.compile(model,backend=counter,fullgraph=True,dynamic=True)
    x=torch.randn(1,512,64,device='cuda')
    hard=torch.rand(1,2,512,device='cuda')<.5
    direction=torch.tensor([[True,False]],device='cuda')
    for i,bounds in enumerate(([0,256,512],[0,512],[0,0,256,256,512])):
        cu=torch.tensor(bounds,device='cuda',dtype=torch.int32)
        mark_cu_seqlens_dynamic(cu)
        model.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):expected=model(x,hard,direction,cu)
        expected.float().square().mean().backward()
        ref=[p.grad.clone() for p in model.parameters()]
        model.zero_grad(set_to_none=True)
        with torch._dynamo.config.patch(error_on_recompile=i>0):
            with torch.autocast('cuda',dtype=torch.bfloat16):out=compiled(x,hard,direction,cu)
            out.float().square().mean().backward()
        torch.testing.assert_close(out,expected,atol=.005,rtol=.03)
        for p,g in zip(model.parameters(),ref):assert_gradient(p.grad,g,'module_parameter')
        if not i:frames=counter.frame_count;assert frames>0
        assert counter.frame_count==frames


def test_module_padding_gate_mapping():
    from flash_dism import DismAttention
    torch.manual_seed(776)
    model=DismAttention(64,2,head_dim=64,value_dim=64,readout_dim=32,vocab_size=512).cuda().train()
    x=torch.randn(2,512,64,device='cuda')
    gate=torch.rand(2,2,512,device='cuda',requires_grad=True)
    mask=torch.ones(2,512,device='cuda',dtype=torch.bool);mask[0,:256]=False
    hard=torch.rand(2,2,512,device='cuda')<.5
    direction=torch.tensor([[True,False],[True,False]],device='cuda')
    with torch.autocast('cuda',dtype=torch.bfloat16):
        actual=model(x,attention_mask=mask,gate_delta=gate,hard=hard,direction=direction)[0]
        pieces=[]
        for b,start in ((0,256),(1,0)):
            pieces.append(model(x[b:b+1,start:],gate_delta=gate[b:b+1,:,start:],
                hard=hard[b:b+1,:,start:],direction=direction[b:b+1])[0])
        expected=torch.cat([torch.cat([torch.zeros_like(pieces[0]),pieces[0]],1),pieces[1]],0)
    torch.testing.assert_close(actual,expected,atol=.003,rtol=.03)
    ga=torch.autograd.grad(actual.float().square().sum(),gate,retain_graph=True)[0]
    ge=torch.autograd.grad(expected.float().square().sum(),gate)[0]
    assert_gradient(ga,ge,'gate_delta')
    assert torch.count_nonzero(ga[0,:,:256])==0
