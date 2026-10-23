"""Final RMSNorm/CE handoff: rounding, chunk layouts, ownership and replay."""
import pytest
import torch
import torch.nn.functional as F

from flash_mxfp8 import RMSNormMXFP8, MXFP8FusedLinearCrossEntropy
from flash_mxfp8.tk_silu_bwd import SiluBwdQuantizeLibrary

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason='SM120 required',
)


@pytest.mark.parametrize('m,k,chunk', [(1,128,128), (129,128,128), (137,192,128),
                                      (1025,768,1024), (2177,768,1024)])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('autocast', [False, True])
def test_norm_layout_rounding_and_gradients(m, k, chunk, dtype, autocast):
    torch.manual_seed(53)
    x = torch.randn(m,k,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    factor = (torch.randn(k,device='cuda')*.1+1).to(dtype).requires_grad_()
    norm = RMSNormMXFP8(chunk_size=chunk)
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=autocast):
        y, chunks = norm(x,factor)
        reference = F.rms_norm(x,(k,),factor,1e-6).bfloat16()
    # Same BF16 output tolerance as the existing fused-add RMSNorm regression.
    # Different FP32 reduction order can flip a few BF16 rounding boundaries.
    torch.testing.assert_close(y,reference,rtol=.01,atol=.008)
    for i, actual in enumerate(chunks):
        expected = SiluBwdQuantizeLibrary().plan_direct(y.detach()[i*chunk:(i+1)*chunk])()
        for a,b in zip(actual,expected):
            assert a.is_contiguous()
            torch.testing.assert_close(a.view(torch.uint8),b.view(torch.uint8),rtol=0,atol=0)
        assert torch.count_nonzero(actual[2][min(chunk,m-i*chunk):].float()) == 0
    dy = torch.randn_like(y)
    grads = torch.autograd.grad(y,(x,factor),dy,retain_graph=True)
    refs = torch.autograd.grad(reference,(x,factor),dy)
    for a,b in zip(grads,refs):
        relative = (a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-30)
        assert relative < .001
    saved = [y.clone(),*[q.clone() for row in chunks for q in row]]
    norm(x.detach()*2,factor.detach()+.2)
    for a,b in zip([y,*[q for row in chunks for q in row]],saved):
        torch.testing.assert_close(a.view(torch.uint8),b.view(torch.uint8),rtol=0,atol=0)
    again = torch.autograd.grad(y,(x,factor),dy)
    for a,b in zip(grads,again):
        torch.testing.assert_close(a,b,rtol=0,atol=0)


@pytest.mark.parametrize('balance', [None,'token','document','document_sum'])
@pytest.mark.parametrize('all_ignored', [False,True])
def test_prequantized_ce_matches_direct_and_rebinds(balance,all_ignored):
    torch.manual_seed(938)
    norm=RMSNormMXFP8(chunk_size=128)
    criterion=MXFP8FusedLinearCrossEntropy(chunk_size=128,balance=balance,softcap=.7)
    x=torch.randn(385,128,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    factor=torch.randn(128,device='cuda',requires_grad=True)
    w=(torch.randn(256,128,device='cuda')*.05).requires_grad_()
    target=torch.randint(256,(385,),device='cuda');target[::7]=-100
    if all_ignored:target.fill_(-100)
    cu=torch.tensor([0,0,1,31,320,384,385,385],device='cuda',dtype=torch.int32)
    for index in range(2):
        y,chunks=norm(x,factor)
        def loss(value):return value[0] if balance=='document_sum' else value
        a=criterion(y,target,w,cu_seqlens=cu,quantized_input=chunks)
        b=criterion(y,target,w,cu_seqlens=cu)
        torch.testing.assert_close(loss(a),loss(b),rtol=0,atol=0)
        if balance=='document_sum':torch.testing.assert_close(a[1],b[1],rtol=0,atol=0)
        ga=torch.autograd.grad(loss(a)*-.5,(x,factor,w),retain_graph=True)
        gb=torch.autograd.grad(loss(b)*-.5,(x,factor,w))
        for u,v in zip(ga,gb):torch.testing.assert_close(u,v,rtol=0,atol=0)
        # New buffers with the same shape must rebind both GEMM and wgrad.
        with torch.no_grad():x.add_(.1);factor.mul_(.9);w.add_(.001)
    assert all(state.xq is None for key,state in criterion._states.items() if key[-1])


def test_prequantized_validation():
    x=torch.randn(129,128,device='cuda',dtype=torch.bfloat16)
    y,chunks=RMSNormMXFP8(chunk_size=128)(x,torch.ones(128,device='cuda'))
    w=torch.randn(256,128,device='cuda');target=torch.zeros(129,device='cuda',dtype=torch.int64)
    ce=MXFP8FusedLinearCrossEntropy(chunk_size=128)
    with pytest.raises(ValueError,match='one four-tensor'):ce(y,target,w,quantized_input=chunks[:-1])
    bad=list(chunks);bad[0]=(bad[0][0],bad[0][1].T,*bad[0][2:])
    with pytest.raises(ValueError,match='prequantized'):ce(y,target,w,quantized_input=bad)
    with pytest.raises(ValueError,match='chunk_size'):RMSNormMXFP8(chunk_size=32)


def test_norm_ce_graph_mutation_and_multiple_forwards():
    torch.manual_seed(94)
    x=torch.randn(257,128,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    factor=torch.ones(128,device='cuda',requires_grad=True)
    w=(torch.randn(256,128,device='cuda')*.05).requires_grad_()
    targets=torch.randint(256,(257,),device='cuda')
    cu=torch.tensor([0,128,257],device='cuda',dtype=torch.int32)
    scalar=torch.tensor(.5,device='cuda')
    norm=RMSNormMXFP8(chunk_size=128)
    ce=MXFP8FusedLinearCrossEntropy(chunk_size=128,balance='document_sum',softcap=30.)
    def step():
        for p in (x,factor,w):p.grad=None
        y,q=norm(x,factor)
        a,count=ce(y,targets,w,cu_seqlens=cu,quantized_input=q)
        # Reuse before backward must not overwrite inverse or saved CE gradients.
        z,q2=norm(x+.1,factor)
        b,_=ce(z,targets,w,cu_seqlens=cu,quantized_input=q2)
        value=(a+b)*scalar
        value.backward()
        return value,count
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):step()
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):result,count=step()
    for i in range(3):
        with torch.no_grad():
            x.add_(.03);factor.mul_(.99);w.add_(.0001)
            targets.copy_((targets+1)%256);cu[1]=64+i*16;scalar.fill_(-.5 if i%2 else .25)
        graph.replay();torch.cuda.synchronize()
        actual=result.clone();actual_count=count.clone()
        grads=[p.grad.clone() for p in (x,factor,w)]
        # Keep graph-owned grad buffers; eager comparison must not rebind them.
        old=[p.grad for p in (x,factor,w)]
        expected,n=step()
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        torch.testing.assert_close(actual_count,n,rtol=0,atol=0)
        for p,a,g in zip((x,factor,w),grads,old):
            torch.testing.assert_close(a,p.grad,rtol=0,atol=0)
            p.grad=g
