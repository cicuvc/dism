"""Opt-in interpolation backward: independent oracle and production comparisons."""
from functools import partial

import pytest
import torch

from flash_dism.emb_kernel import emb_fwd_wrapper, emb_bwd_wrapper
from flash_dism.kernels.interp_bwd_split import emb_bwd_split
from gradient_acceptance import assert_gradient


@pytest.mark.parametrize('d,vocab,n,strided', [(32,37,256,False), (64,512,768,False),
                                             (64,37,256,True), (32,512,256,True)])
@pytest.mark.parametrize('beta,tau_value', [(1.,None), (.7,.5), (1.7,2.)])
@pytest.mark.parametrize('reduction,splits',[('partial',1),('partial',16),
                                            ('atomic',1),('atomic',16),('register',1)])
def test_split_oracle(d, vocab, n, strided, beta, tau_value, splits, reduction):
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
    old = emb_bwd_wrapper(*args)
    actual = emb_bwd_split(*args,splits=splits,reduction=reduction)
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
@pytest.mark.parametrize('reduction',['partial','atomic','register'])
def test_compiled_operator_replacement(monkeypatch,direction_value,probability,reduction):
    import flash_dism.compiler as compiler
    from flash_dism.varlen import VarlenLayout
    torch.manual_seed(173)
    n,h,d,r = 768,2,64,32
    def rand(c):
        return (torch.randn(1,n,h,c,device='cuda')*.25).bfloat16()
    q,k,sq,sk,v = [rand(c) for c in (d,d,r,r,d)]
    eq,ek = [torch.randn(h,512,d,device='cuda')*.25 for _ in range(2)]
    tau = torch.full((h,),.5,device='cuda')
    hard = (torch.rand(1,h,3,1,device='cuda') < probability).expand(1,h,3,256).reshape(1,h,n).contiguous()
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
    expected = compiler.voc_backward(inputs,saved,boundaries,do,layout)
    monkeypatch.setattr(compiler.cu,'core_backward',lambda *args: captured[0])
    monkeypatch.setattr(compiler,'emb_bwd_wrapper',partial(emb_bwd_split,reduction=reduction,
                                                        splits=1 if reduction=='register' else 16))
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


@pytest.mark.parametrize('reduction',['partial','atomic','register'])
def test_inductor_autograd(monkeypatch,reduction):
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
    hard=(torch.rand(1,h,2,1,device='cuda')<.5).expand(1,h,2,256).reshape(1,h,n).contiguous()
    beta=torch.tensor(1.)
    def run(*inputs):
        return compiler.voc_forward(*inputs,beta,direction,hard,boundaries,layout)[0]
    output=run(*values)
    dout=torch.randn_like(output)
    expected=torch.autograd.grad(output,values,dout)
    monkeypatch.setattr(compiler,'emb_bwd_wrapper',partial(emb_bwd_split,reduction=reduction,
                                                        splits=1 if reduction=='register' else 16))
    with torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True), \
         torch._inductor.config.patch(compile_threads=1):
        compiled=torch.compile(run,fullgraph=True,dynamic=True)
        actual_output=compiled(*values)
        actual=torch.autograd.grad(actual_output,values,dout)
    torch.testing.assert_close(actual_output,output,atol=0,rtol=0)
    for name,new,old in zip(('q','k','sq','sk','v','eq','ek','rtau'),actual,expected):
        assert_gradient(new,old,name)
