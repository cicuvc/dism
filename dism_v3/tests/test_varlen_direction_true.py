"""Fixed-true specialization versus the retained runtime-direction kernels."""
import itertools
import pytest
import torch
import cu_flash_dism as cu
from flash_dism.varlen import VarlenLayout, forward_varlen, backward_varlen


def inputs(r, d, dv, lengths, mode):
    torch.manual_seed(812)
    n, h = sum(lengths), 2
    bounds = torch.tensor([0, *itertools.accumulate(lengths)], dtype=torch.int32)
    layout = VarlenLayout.from_cu_seqlens(bounds, n)
    vectors = [(torch.randn(1,n,h,c,device='cuda')*.2).bfloat16()
               for c in (d,d,r,r,dv)]
    lse = [torch.randn(1,n,h,device='cuda')*.2+5 for _ in range(2)]
    labels = [torch.randint(32,(1,h,n),device='cuda') for _ in range(2)]
    direction = torch.ones(1,h,device='cuda',dtype=torch.bool)
    hard = torch.tensor([[mode != 'soft', mode == 'hard']],device='cuda')
    return layout, (*vectors,*lse,*labels,direction,hard,torch.ones(h,device='cuda'))


@pytest.mark.parametrize('r,d,dv', list(itertools.product((16,32),(32,64),(32,64))))
@pytest.mark.parametrize('mode', ['soft','mixed','hard'])
def test_true_stages(r,d,dv,mode):
    layout, x = inputs(r,d,dv,[256,0,512],mode)
    old, norm, state = forward_varlen(*x,layout=layout,save_state=True)
    new, nn, ns = forward_varlen(*x,layout=layout,save_state=True,direction_true=True)
    torch.testing.assert_close(new,old,rtol=0,atol=0)
    for a,b in [(nn,norm),(ns['vertical'],state['vertical']),
                (ns['operands'][12],state['operands'][12])]:
        torch.testing.assert_close(a,b,rtol=2e-6,atol=2e-5)
    do = torch.randn_like(new)
    expected = backward_varlen(state,do)
    actual = backward_varlen(ns,do)
    for k in expected:
        torch.testing.assert_close(actual[k],expected[k],rtol=2e-5,atol=2e-6)
    assert torch.count_nonzero(actual['k_lse']) == 0


def test_true_ignores_unused_metadata_and_empty():
    for lengths in ([0,0], [256,512]):
        layout, x = inputs(32,64,64,lengths,'soft')
        old,_,state = forward_varlen(*x,layout=layout,save_state=True,direction_true=True)
        # Deliberately poison unused K LSE; true kernels must never consume it.
        poisoned = list(x)
        poisoned[6] = torch.full_like(x[6],float('nan'))
        new,_,ns = forward_varlen(*poisoned,layout=layout,save_state=True,direction_true=True)
        torch.testing.assert_close(new,old,rtol=0,atol=0)
        do = torch.randn_like(old)
        eg, ag = backward_varlen(state,do), backward_varlen(ns,do)
        for k in eg:
            torch.testing.assert_close(ag[k],eg[k],rtol=2e-5,atol=2e-6)


def test_true_kernel_does_not_read_direction():
    layout, x = inputs(32,64,64,[256,512],'soft')
    expected,_,state = forward_varlen(*x,layout=layout,save_state=True,direction_true=True)
    operands = list(state['operands'])
    # Native internal contract: the host specialization overrides device flags.
    operands[10] = torch.zeros_like(operands[10])
    a,b = cu.varlen_summary(operands[:12],layout.table,True)
    operands[12] = cu.varlen_chunk(a,b,layout.table,2)
    actual,norm,vertical = cu.varlen_forward(operands,layout.table,True,False,True)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    do = torch.randn_like(actual)
    eg = backward_varlen(state,do)
    ag,_ = cu.core_backward(operands,vertical,actual,norm,do,layout.table,0,False,False,True)
    for k in eg:
        torch.testing.assert_close(ag[k],eg[k],rtol=2e-5,atol=2e-6)


def test_training_rows_on_nondefault_stream():
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        layout, x = inputs(32,64,64,[256,256,256,256],'soft')
        x = list(x)
        x[10] = torch.tensor([[True,False],[False,True]],device='cuda')
        old,_,state = forward_varlen(*x,layout=layout,save_state=True)
        new,_,ns = forward_varlen(*x,layout=layout,save_state=True,direction_true=True)
        torch.testing.assert_close(new,old,rtol=0,atol=0)
        do = torch.randn_like(old)
        eg, ag = backward_varlen(state,do), backward_varlen(ns,do)
        for k in eg:
            torch.testing.assert_close(ag[k],eg[k],rtol=2e-5,atol=2e-6)
    stream.synchronize()
