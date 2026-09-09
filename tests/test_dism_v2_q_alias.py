"""Q input reuse across workload resets and partially used three-slot rings."""
import pytest
import torch
from dism_v2.core import forward


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('n',(1,65,129,257,385,1024))
@pytest.mark.parametrize('direction',('q_from_k','k_from_q'))
@torch.no_grad()
def test_q_alias_workload_replay(n,direction):
    # Exceed the persistent grid even when N=1; key-tile counts vary by task.
    batch=torch.cuda.get_device_properties(0).multi_processor_count+3
    torch.manual_seed(773)
    a,b,v=[torch.randn(batch,1,n,64,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
    lse=torch.full((batch,1,n),4.,device='cuda')
    tau=torch.zeros(1,device='cuda')
    qlabel,klabel=[torch.randint(0,13,(batch,1,n),device='cuda',dtype=torch.int32)
                   for _ in range(2)]
    kwargs=dict(sm_scale=.125,direction=direction,hard_prob=.37,
                return_rng_state=True,save_boundaries=True)
    first=forward(a,b,v,lse,tau,qlabel,klabel,**kwargs,
                  generator=torch.Generator(device='cuda').manual_seed(912))
    for _ in range(3):
        second=forward(a,b,v,lse,tau,qlabel,klabel,**kwargs,rng_state=first[-1])
        for x,y in zip(first[:2],second[:2]):
            assert torch.isfinite(y).all()
            torch.testing.assert_close(x,y,atol=0,rtol=0)
        for field in ('vertical','horizontal'):
            torch.testing.assert_close(getattr(first[2],field),getattr(second[2],field),atol=0,rtol=0)
