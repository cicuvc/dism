"""Production atomic backward: independent oracle and legacy comparisons."""
import pytest
import torch

from flash_dism.emb_kernel import emb_fwd_wrapper, emb_bwd_wrapper, _emb_bwd_legacy
from flash_dism.kernels.interp_bwd_split import emb_bwd_split
from gradient_acceptance import assert_gradient


@pytest.mark.parametrize('d,vocab,n,strided', [(32,37,256,False), (64,512,768,False),
                                             (64,37,256,True), (32,512,256,True)])
@pytest.mark.parametrize('beta,tau_value', [(1.,None), (.7,.5), (1.7,2.)])
@pytest.mark.parametrize('autotune,splits', [(False, 1), (False, 16), (True, 16)])
def test_split_oracle(d, vocab, n, strided, beta, tau_value, splits, autotune):
    torch.manual_seed(275)
    b, h = 2, 3
    def rand(shape):
        return (torch.randn(shape, device='cuda') * .25).bfloat16()
    q = rand((b,h,n,d)).transpose(1,2) if strided else rand((b,n,h,d))
    k = rand((b,n*2,h,d))[:,::2] if strided else rand((b,n,h,d))
    eq, ek = rand((h,vocab*2,d))[:,::2], rand((h,vocab,d))
    tau = None if tau_value is None else torch.full((h,),tau_value,device='cuda')
    saved = emb_fwd_wrapper(q,k,eq,ek,beta,tau)
    doq, dok = rand((b,h,n,d)).transpose(1,2), rand((b,n,h,d))
    dlq, dlk = [torch.randn((b,h,n),device='cuda') for _ in range(2)]
    args = (q,k,eq,ek,*saved[:4],doq,dok,dlq,dlk,beta,tau)
    old = _emb_bwd_legacy(*args)
    actual = emb_bwd_wrapper(*args) if autotune else emb_bwd_split(*args,splits=splits,autotune=False)
    xq,xk,qe,ke = [t.double().detach().requires_grad_() for t in (q,k,eq,ek)]
    sq = torch.einsum('bnhd,hvd->bhnv',xq,qe)*beta
    sk = torch.einsum('bnhd,hvd->bhnv',xk,ke)*beta
    oq = torch.einsum('bhnv,hvd->bnhd',sk.softmax(-1),qe)
    ok = torch.einsum('bhnv,hvd->bnhd',sq.softmax(-1),ke)
    loss = (oq*doq.double()).sum()+(ok*dok.double()).sum()
    loss += (sk.logsumexp(-1)*dlq.double()).sum()+(sq.logsumexp(-1)*dlk.double()).sum()
    reference = torch.autograd.grad(loss,(xq,xk,qe,ke))
    for name,new,baseline,oracle in zip(('q','k','eq','ek'),actual,old,reference):
        assert_gradient(baseline,oracle,name)
        assert_gradient(new,oracle,name)
        # Splitting only changes FP32 reduction grouping, not BF16 math.
        relative = (new.double()-baseline.double()).norm()/baseline.double().norm().clamp_min(1e-12)
        assert relative < 1e-4, (name,relative.item())


@pytest.mark.parametrize('direction_value',[False,True])
@pytest.mark.parametrize('probability',[0.,.5,1.])
def test_compiled_operator_replacement(monkeypatch,direction_value,probability):
    import flash_dism.compiler as compiler
    from flash_dism.varlen import VarlenLayout
    torch.manual_seed(173)
    n,h,d,r = 768,2,64,32
    def rand(c):
        return (torch.randn(1,n,h,c,device='cuda')*.25).bfloat16()
    q,k,sq,sk,v = [rand(c) for c in (d,d,r,r,d)]
    eq,ek = [torch.randn(h,512,d,device='cuda')*.25 for _ in range(2)]
    tau = torch.full((h,),.5,device='cuda')
    hard = torch.rand(1,h,device='cuda') < probability
    direction = torch.full((1,h),direction_value,device='cuda',dtype=torch.bool)
    boundaries = torch.tensor([0,256,768],device='cuda',dtype=torch.int32)
    layout = VarlenLayout.from_cu_seqlens(boundaries,n).table.flatten()
    inputs = [q,k,sq,sk,v,eq,ek,tau,torch.tensor(1.7),direction,hard]
    saved = compiler.voc_forward(*inputs,boundaries,layout)
    do = torch.randn_like(v)
    # Reuse one core result: its cross-CTA FP32 atomics are not bitwise
    # deterministic. Isolate interpolation instead of relaxing its checks.
    core_backward = compiler.cu.core_backward
    captured = []
    def capture(*args):
        result = core_backward(*args)
        captured.append(result)
        return result
    monkeypatch.setattr(compiler.cu,'core_backward',capture)
    monkeypatch.setattr(compiler,'emb_bwd_wrapper',_emb_bwd_legacy)
    expected = compiler.voc_backward(inputs,saved,boundaries,do,layout)
    monkeypatch.setattr(compiler.cu,'core_backward',lambda *args: captured[0])
    monkeypatch.setattr(compiler,'emb_bwd_wrapper',emb_bwd_wrapper)
    actual = compiler.voc_backward(inputs,saved,boundaries,do,layout)
    for name,new,old in zip(('q','k','sq','sk','v','eq','ek','rtau'),actual,expected):
        assert_gradient(new,old,name)
    for index in (2,3,4,7):
        torch.testing.assert_close(actual[index],expected[index],atol=0,rtol=0)


def test_optional_lse_and_empty():
    for n in (0,65,256):
        q = torch.zeros((1,n,2,32),device='cuda',dtype=torch.bfloat16)
        table = torch.zeros((2,37,32),device='cuda',dtype=torch.bfloat16)
        scalar = torch.full((1,2,n),torch.log(torch.tensor(37.)).item(),device='cuda')
        if n != 256:
            with pytest.raises(RuntimeError,match='256-token aligned'):
                emb_bwd_split(q,q,table,table,q,q,scalar,scalar,q,q,None,None)
            continue
        got = emb_bwd_split(q,q,table,table,q,q,scalar,scalar,q,q,None,None)
        assert all(torch.count_nonzero(t).item()==0 for t in got)


def test_inductor_autograd(monkeypatch):
    import flash_dism.compiler as compiler
    from flash_dism.varlen import VarlenLayout
    torch.manual_seed(276)
    n,h=512,2
    values=[(torch.randn(1,n,h,d,device='cuda')*.25).bfloat16().requires_grad_()
            for d in (64,64,32,32,64)]
    values += [(torch.randn(h,512,64,device='cuda')*.25).requires_grad_() for _ in range(2)]
    values += [torch.full((h,),.5,device='cuda',requires_grad=True)]
    boundaries=torch.tensor([0,256,512],device='cuda',dtype=torch.int32)
    layout=VarlenLayout.from_cu_seqlens(boundaries,n).table.flatten()
    direction=torch.ones(1,h,device='cuda',dtype=torch.bool)
    hard=torch.rand(1,h,device='cuda')<.5
    beta=torch.tensor(1.)
    def run(*inputs):
        return compiler.voc_forward(*inputs,beta,direction,hard,boundaries,layout)[0]
    output=run(*values)
    dout=torch.randn_like(output)
    monkeypatch.setattr(compiler,'emb_bwd_wrapper',_emb_bwd_legacy)
    expected=torch.autograd.grad(output,values,dout)
    monkeypatch.setattr(compiler,'emb_bwd_wrapper',emb_bwd_wrapper)
    with torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True), \
         torch._inductor.config.patch(compile_threads=1):
        compiled=torch.compile(run,fullgraph=True,dynamic=True)
        actual_output=compiled(*values)
        actual=torch.autograd.grad(actual_output,values,dout)
    torch.testing.assert_close(actual_output,output,atol=0,rtol=0)
    for name,new,old in zip(('q','k','sq','sk','v','eq','ek','rtau'),actual,expected):
        assert_gradient(new,old,name)


def test_autotune_cold_and_warm_preserve_both_contributions(monkeypatch):
    from flash_dism.kernels.interp_bwd_split import _dq, _dkv
    # Separate Q/K layouts force TWO first-call cache misses. A reset_to_zero
    # tuning hook would incorrectly delete the first attention's gradients.
    for tuner in (_dq, _dkv):
        monkeypatch.setattr(tuner, 'cache', {})
    calls = []
    benchmark = _dkv.do_bench
    def counted(fn, quantiles):
        calls.append(1)
        return benchmark(fn, quantiles)
    monkeypatch.setattr(_dkv, '_do_bench', counted)
    monkeypatch.setattr(_dkv, 'do_bench', counted)
    torch.manual_seed(981)
    b, n, h, d, vocab = 1, 256, 2, 64, 37
    def rand(shape):
        return (torch.randn(shape, device='cuda') * .25).bfloat16()
    q = rand((b, h, n, d)).transpose(1, 2)
    k = rand((b, n, h, d))
    eq, ek = [rand((h, vocab, d)) for _ in range(2)]
    saved = emb_fwd_wrapper(q, k, eq, ek)
    doq, dok = [rand((b, n, h, d)) for _ in range(2)]
    dlq, dlk = [torch.randn(b, h, n, device='cuda') for _ in range(2)]
    args = (q, k, eq, ek, *saved[:4], doq, dok, dlq, dlk)
    expected = _emb_bwd_legacy(*args)
    cold = emb_bwd_wrapper(*args)
    assert len(_dkv.cache) == 2
    trials = len(calls)
    assert trials > 2
    # Change data without changing the tuning key, checking cache hits don't
    # reuse old gradients or accumulate across separate wrapper invocations.
    doq.mul_(.5)
    dlk.mul_(2.)
    warm_expected = _emb_bwd_legacy(*args)
    warm = emb_bwd_wrapper(*args)
    assert len(calls) == trials
    for result, reference in ((cold, expected), (warm, warm_expected)):
        for new, old in zip(result, reference):
            relative = (new.double() - old.double()).norm() / old.double().norm().clamp_min(1e-12)
            assert relative < 1e-4


def test_production_backward_cuda_graph_after_warmup():
    torch.manual_seed(982)
    q, k = [(torch.randn(1, 256, 1, 32, device='cuda') * .25).bfloat16() for _ in range(2)]
    eq, ek = [(torch.randn(1, 37, 32, device='cuda') * .25).bfloat16() for _ in range(2)]
    saved = emb_fwd_wrapper(q, k, eq, ek)
    doq, dok = [torch.randn_like(q) for _ in range(2)]
    args = (q, k, eq, ek, *saved[:4], doq, dok, None, None)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            emb_bwd_wrapper(*args)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = emb_bwd_wrapper(*args)
    for _ in range(2):
        doq.mul_(.5)
        graph.replay()
        expected = _emb_bwd_legacy(*args)
        for new, old in zip(actual, expected):
            relative = (new.double() - old.double()).norm() / old.double().norm().clamp_min(1e-12)
            assert relative < 1e-4
