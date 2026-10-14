"""Hard forward vs the maintained v3 oracle, without tolerance changes."""
from pathlib import Path
import subprocess

import numpy as np
import pytest
import torch

from flash_dism.hard_forward import forward_hard, hard_readout, hard_log_weights
from flash_dism.reference.dism_v3_ref import dism_ref
from flash_dism.varlen import VarlenLayout
from test_hard_summary import labels


def fixture(n=512,pattern='boundary',b=2,h=3,r=32,dv=64):
    torch.manual_seed(417)
    sq,sk=[(torch.nn.functional.silu(torch.randn(b,n,h,r,device='cuda'))*.4).bfloat16()
           for _ in range(2)]
    v=torch.randn(b,n,h,dv,device='cuda').bfloat16()
    iq,ik=labels(n,pattern,b,h)
    hard=torch.ones((b,h),device='cuda',dtype=torch.bool)
    if b>1 and h>1:
        hard[0,1]=False
        hard[1,0]=False
    tau=torch.linspace(-.5,.5,h,device='cuda')
    return sq,sk,v,iq,ik,hard,tau


def oracle(x,direction=True):
    sq,sk,v,iq,ik,hard,tau=x
    b,n,h,_=sq.shape
    q=torch.zeros(b,n,h,32,device='cuda',dtype=torch.float64)
    lse=torch.zeros(b,n,h,device='cuda',dtype=torch.float64)
    flags=torch.ones_like(iq,dtype=torch.bool)
    output=dism_ref(q,q,sq.double(),sk.double(),lse,lse,iq,ik,
                    torch.full((b,h),direction,device='cuda'),flags,v.double(),tau.double())
    return output*(hard[:,None,:,None])


def normalizer_oracle(x):
    _,_,_,iq,ik,hard,tau=x
    q,k=iq.cpu().numpy(),ik.cpu().numpy()
    flags,t=hard.cpu().numpy(),tau.double().cpu().numpy()
    b,h,n=q.shape
    previous=np.full((b,h,n),-np.inf)
    result=np.empty_like(previous)
    for i in range(n):
        shifted=np.concatenate((np.full((b,h,1),-np.inf),previous[:,:,:-1]),axis=-1)
        previous=np.where(q[:,:,i,None]==k,t[None,:,None]+np.logaddexp(0,shifted),-np.inf)
        result[:,:,i]=np.logaddexp(0,np.logaddexp.reduce(previous[:,:,:i+1],axis=-1))/np.log(2.)
    return torch.from_numpy(np.where(flags[:,:,None],result,0.)).cuda()


def assert_output(actual,expected):
    torch.testing.assert_close(actual.float(),expected.float(),atol=.008,rtol=.025)
    error=(actual.double()-expected.double()).norm()/expected.double().norm().clamp_min(1e-8)
    assert error<.01,error.item()


@pytest.mark.parametrize('n',[256,2048])
def test_log_weight_table(n):
    tau=torch.tensor([-4.,-.5,-1e-6,-1e-12,0.,1e-12,1e-6,.5,4.],device='cuda')
    actual=hard_log_weights(tau,n)
    length=torch.arange(1,n+1,device='cuda',dtype=torch.float64)
    t=tau.double()[:,None]
    a=t.abs()
    expected=(torch.where(t>0,length*t,t)+torch.log(-torch.expm1(-length*a))-
              torch.log(-torch.expm1(-a)))/np.log(2.)
    expected[tau==0]=length.log2()
    assert (actual[:,0]==-1e6).all()
    torch.testing.assert_close(actual[:,1:].double(),expected,atol=2e-5,rtol=2e-6)


@pytest.mark.parametrize('pattern',['random','equal','disjoint','periodic','boundary'])
@pytest.mark.parametrize('n',[256,512])
def test_forward_oracle(n,pattern):
    x=fixture(n,pattern)
    output,norm=forward_hard(*x,ctas=1)
    assert_output(output,oracle(x))
    torch.testing.assert_close(norm.double(),normalizer_oracle(x),atol=2e-5,rtol=2e-6)
    assert output.dtype==torch.bfloat16 and norm.dtype==torch.float32


@pytest.mark.parametrize('tau',[-1.,-1e-6,0.,1e-6,.1,.5,4.])
def test_long_all_match(tau):
    x=list(fixture(1024,'equal',b=1,h=1))
    x[-1].fill_(tau)
    output,norm=forward_hard(*x,ctas=1)
    assert_output(output,oracle(x,False))
    torch.testing.assert_close(norm.double(),normalizer_oracle(x),atol=2e-5,rtol=2e-6)


@pytest.mark.parametrize('r,dv',[(16,32),(16,64),(32,32),(32,64)])
def test_readout_dimensions(r,dv):
    x=fixture(512,'periodic',r=r,dv=dv)
    actual,norm=forward_hard(*x,ctas=2)
    assert_output(actual,oracle(x))
    torch.testing.assert_close(norm.double(),normalizer_oracle(x),atol=2e-5,rtol=2e-6)


def test_signed_readout_fallback_and_skip():
    x=list(fixture())
    output,norm=forward_hard(*x,ctas=1)
    x[0]=-x[0]
    opposite,opposite_norm=forward_hard(*x,ctas=1)
    torch.testing.assert_close(opposite,-output,rtol=0,atol=0)
    torch.testing.assert_close(opposite_norm,norm,rtol=0,atol=0)
    x[0].zero_()
    zero,zero_norm=forward_hard(*x,ctas=1)
    assert not zero.count_nonzero()
    torch.testing.assert_close(zero_norm,norm,rtol=0,atol=0)
    x[3].fill_(1)
    x[4].fill_(2)
    zero,zero_norm=forward_hard(*x,ctas=1)
    assert not zero.count_nonzero() and not zero_norm.count_nonzero()
    x[5].fill_(False)
    for item in x[:3]: item.fill_(float('nan'))
    zero,zero_norm=forward_hard(*x,ctas=1)
    assert not zero.count_nonzero() and not zero_norm.count_nonzero()


def packed_fixture(pattern='boundary'):
    x=list(fixture(2048,pattern,b=1,h=2))
    x[5]=torch.tensor([[True,False],[False,True]],device='cuda')
    layout=VarlenLayout.from_cu_seqlens(torch.tensor([0,0,256,512,1024,1024,2048],dtype=torch.int32),2048)
    return x,layout


def doc_slice(x,start,end):
    return tuple([item[:,start:end].contiguous() for item in x[:3]]+
                 [item[:,:,start:end].contiguous() for item in x[3:5]]+
                 [x[5][start//1024:start//1024+1].contiguous(),x[6]])


@pytest.mark.parametrize('pattern',['equal','periodic','boundary'])
def test_packed_matches_fixed_and_oracle(pattern):
    x,layout=packed_fixture(pattern)
    actual,norm=forward_hard(*x,layout=layout,ctas=1)
    for start,end in ((0,256),(256,512),(512,1024),(1024,2048)):
        local=doc_slice(x,start,end)
        expected,expected_norm=forward_hard(*local,ctas=1)
        torch.testing.assert_close(actual[:,start:end],expected,atol=0,rtol=0)
        torch.testing.assert_close(norm[:,:,start:end],expected_norm,atol=2e-5,rtol=2e-6)
        assert_output(actual[:,start:end],oracle(local))
    x[2][:,:256].add_(5)
    changed,changed_norm=forward_hard(*x,layout=layout,ctas=1)
    torch.testing.assert_close(changed[:,256:],actual[:,256:],rtol=0,atol=0)
    torch.testing.assert_close(changed_norm,norm,rtol=0,atol=0)


@pytest.mark.parametrize('packed',[False,True])
def test_runtime_handoffs(packed):
    if packed: x,layout=packed_fixture()
    else: x,layout=fixture(),None
    expected,norm=forward_hard(*x,layout=layout,ctas=1)
    # Sanitizer target: exact replay, mixed tasks, both pipeline/mail phases.
    assert torch.isfinite(expected).all() and torch.isfinite(norm).all()
    for ctas in (2,0,1):
        actual,actual_norm=forward_hard(*x,layout=layout,ctas=ctas)
        torch.testing.assert_close(actual,expected,atol=0,rtol=0)
        torch.testing.assert_close(actual_norm,norm,atol=0,rtol=0)


def test_causality_graph_and_tau_refresh():
    x=list(fixture(512,'equal'))
    original,_=forward_hard(*x)
    x[2][:,127:]=20
    changed,_=forward_hard(*x)
    torch.testing.assert_close(original[:,:127],changed[:,:127],atol=0,rtol=0)
    torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured,norm=forward_hard(*x,ctas=1)
    for tau in (0.,.1,-.1):
        x[-1].fill_(tau)
        graph.replay()
        assert_output(captured,oracle(x))
        torch.testing.assert_close(norm.double(),normalizer_oracle(x),atol=2e-5,rtol=2e-6)


def test_validation():
    x=list(fixture(256))
    _,_,state=forward_hard(*x,save_state=True)
    with pytest.raises(RuntimeError,match='too short'):
        hard_readout(*x[:6],state['boundary'],state['log_weights'][:,:128].contiguous())
    with pytest.raises(RuntimeError,match='boundary'):
        hard_readout(*x[:6],state['boundary'].float(),state['log_weights'])
    x[5]=x[5][:,:,None].expand(2,3,256)
    with pytest.raises(RuntimeError,match='token-level'):
        forward_hard(*x)


def test_packed_metadata_cache_no_repeated_h2d():
    x,layout=packed_fixture()
    for _ in range(4): forward_hard(*x,layout=layout)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(8): forward_hard(*x,layout=layout)
        torch.cuda.synchronize()
    copies=[e.name for e in prof.events()
            if e.device_type==torch.autograd.DeviceType.CUDA and 'Memcpy HtoD' in e.name]
    assert not copies,copies


def test_codegen():
    import cu_flash_dism as cu
    root=Path(__file__).resolve().parents[1]
    for r,d,v in cu.configs:
        tag=f'r{r}_d{d}_v{v}'
        obj=root/f'build/object/{tag}/hard_forward.cu.dev.sm120a.o'
        sass=subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','-sass',str(obj)],text=True)
        assert 'hard_forward_kernel' in sass and 'hard_log_weights_kernel' in sass
        assert 'CALL' not in sass and 'MUFU.TANH' not in sass
        assert 'LDGSTS' in sass and 'UTMALDG' in sass and 'HMMA' in sass
        assert '/torch/include/' not in (root/f'build/deps/{tag}/hard_forward.cu.d').read_text()
