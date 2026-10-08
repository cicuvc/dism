import pytest
import cu_flash_dism as _precision_backend
import torch
import cu_flash_dism as cu
from flash_dism.backward import backward_core
from test_backward_summary import prepare,pytestmark
from test_forward import inputs_for
from flash_dism.forward import forward_core


@pytest.mark.parametrize('n',[31,32,127,128,255,256,511,1025])
@pytest.mark.skipif(not _precision_backend.fp32_enabled(),
                    reason='requires optional FP32 validation instances')
def test_repeated_tasks_and_cta_counts(n):
    _,state,do,_,_=prepare(n)
    baseline=backward_core(state,do,ctas=1,fp32_output=True)
    for count in (2,0,1):
        actual=backward_core(state,do,ctas=count,fp32_output=True)
        for name,value in actual.items():
            torch.testing.assert_close(value,baseline[name],atol=2e-4,rtol=2e-4,msg=name)


@pytest.mark.parametrize('n',[256,512,768])
def test_unmatched_hard_zero_gradients(n):
    _,state,do,_,_=prepare(n,'hard')
    state['operands'][7].fill_(1)
    state['operands'][8].fill_(2)
    # Regenerate forward after changing labels, including all saved checkpoints.
    q,k,sq,sk,v,lq,lk,iq,ik,hard,direction,tau,horizontal=state['operands']
    sa,sb=cu.summarization(q,k,lq,lk,iq,ik,hard,direction,tau,n,1)
    horizontal=cu.chunk_scan(sa,sb,n)
    operands=(q,k,sq,sk,v,lq,lk,iq,ik,hard,direction,tau,horizontal)
    state['operands']=operands
    state['vertical'].fill_(-1e6)
    state['output'],state['lse2']=cu.forward_output(*operands,n,1,state['vertical'])
    result=backward_core(state,do,ctas=1)
    for name,x in result.items():
        assert torch.count_nonzero(x)==0,name


def test_nondefault_stream():
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _,state,do,_,_=prepare(512)
        result=backward_core(state,do,ctas=1)
    torch.cuda.current_stream().wait_stream(stream)
    for value in result.values():
        assert torch.isfinite(value).all()


@pytest.mark.skipif(not _precision_backend.fp32_enabled(),
                    reason='requires optional FP32 validation instances')
def test_zero_soft_query_gradient_structure():
    inputs=list(inputs_for(129,'soft'))
    inputs[2].zero_()
    out,_,state=forward_core(*inputs,save_state=True)
    assert torch.count_nonzero(out)==0
    do=torch.randn_like(out).bfloat16()
    gradients=backward_core(state,do,fp32_output=True)
    for name in ('q_vec','k_vec','sk_vec','v','q_lse','k_lse','rtau'):
        assert torch.count_nonzero(gradients[name])==0,name
    assert gradients['sq_vec'].norm()>0


@pytest.mark.skipif(not _precision_backend.fp32_enabled(),
                    reason='requires optional FP32 validation instances')
def test_soft_lse_tau_chain_rule():
    # Raw LSE and tau are independent API inputs: absorbing tau must not count
    # its soft-branch gradient twice in the Python custom autograd wrapper.
    _,state,do,_,_=prepare(129,'soft')
    g=backward_core(state,do,fp32_output=True)
    dlse=(g['q_lse'].double()+g['k_lse'].double()).sum((0,1))
    torch.testing.assert_close(g['rtau'].double(),-dlse,atol=1e-4,rtol=1e-4)
