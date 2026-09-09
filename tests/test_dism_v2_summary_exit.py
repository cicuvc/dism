"""Kernel-only persistent exit/reuse coverage, suitable for CUDA sanitizers."""
import itertools
import pytest
import torch
from dism_v2.core import forward

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("d,dv,direction,n", itertools.product(
    (32,64,128), (32,64,128), ("q_from_k","k_from_q"), (65,257)))
@torch.no_grad()
def test_summary_exit_replay(d,dv,direction,n):
    # More tasks than SMs; no dense Torch oracle in the sanitizer process.
    batch = torch.cuda.get_device_properties(0).multi_processor_count // 2 + 3
    torch.manual_seed(971)
    shape = (batch,2,n)
    a,b = [torch.randn((*shape,d),device="cuda",dtype=torch.bfloat16) for _ in range(2)]
    v = torch.randn((*shape,dv),device="cuda",dtype=torch.bfloat16)
    lse = torch.full(shape,4.,device="cuda")
    tau = torch.zeros(2,device="cuda")
    qlabel,klabel = [torch.randint(0,11,shape,device="cuda",dtype=torch.int32) for _ in range(2)]
    args = (a,b,v,lse,tau,qlabel,klabel)
    kwargs = dict(sm_scale=d**-.5,direction=direction,hard_prob=.37,
                  return_rng_state=True,save_boundaries=True)
    result = forward(*args,**kwargs,
        generator=torch.Generator(device="cuda").manual_seed(1729))
    replay = forward(*args,**kwargs,rng_state=result[-1])
    for x,y in zip(result[:2],replay[:2]):
        assert torch.isfinite(x).all()
        torch.testing.assert_close(x,y,atol=0,rtol=0)
    for field in ("vertical","horizontal"):
        torch.testing.assert_close(getattr(result[2],field),getattr(replay[2],field),atol=0,rtol=0)
