"""Logical tensor allocation redzones, complementary to uncached memcheck.

ATen factories called inside the C++ frontend/phase wrappers are intercepted.
Each allocation gets separately checked prefix/suffix guard bytes; it remains
256-byte aligned for TMA. This catches writes, not reads, and is not a proof of
all possible offsets or in-bounds cross-document correctness.
"""
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from test_multi_config import CONFIGS,inputs
from flash_dism.forward import forward_core
from flash_dism.backward import backward_core
from flash_dism.varlen import VarlenLayout,forward_varlen,backward_varlen


class TensorCanaries(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.allocations=[]

    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        result=func(*args,**(kwargs or {}))
        factory=func._schema.name in {
            'aten::empty','aten::empty_strided','aten::empty_like',
            'aten::full','aten::full_like','aten::zeros','aten::zeros_like',
            'aten::ones','aten::ones_like',
            'aten::_to_copy','aten::clone',
        }
        if not factory or not isinstance(result,torch.Tensor) or not result.is_cuda:
            return result
        # Use the actual strided footprint, not merely numel. All nonempty
        # production allocations currently have nonnegative strides.
        extent=1+sum((size-1)*stride for size,stride in zip(result.shape,result.stride())) if result.numel() else 0
        nbytes=extent*result.element_size()
        guard=4096
        storage=torch.full((nbytes+2*guard,),0xA5,dtype=torch.uint8,device=result.device)
        data=storage[guard:guard+nbytes].view(result.dtype)
        guarded=data.as_strided(result.shape,result.stride())
        # Preserve initialization for full/zeros; empty's contents are unspecified.
        if func._schema.name not in {'aten::empty','aten::empty_strided','aten::empty_like'}:
            guarded.copy_(result)
        self.allocations.append((storage,nbytes,str(func),tuple(result.shape),guarded.data_ptr()))
        return guarded

    def verify(self):
        torch.cuda.synchronize()
        assert len(self.allocations)>=8,'C++ ATen allocations were not intercepted'
        for storage,nbytes,name,shape,_ in self.allocations:
            assert (storage[:4096]==0xA5).all(),('prefix overwrite',name,shape)
            assert (storage[4096+nbytes:]==0xA5).all(),('suffix overwrite',name,shape)


@pytest.mark.parametrize('config',CONFIGS)
@pytest.mark.parametrize('packed',[False,True])
def test_core_tensor_redzones(config,packed):
    x=inputs(config,768,'mixed')
    kwargs={}
    if packed:
        kwargs['layout']=VarlenLayout.from_cu_seqlens(torch.tensor([0,256,256,768],dtype=torch.int32),768)
    forward=forward_varlen if packed else forward_core
    backward=backward_varlen if packed else backward_core
    expected,_,saved=forward(*x,save_state=True,**kwargs)
    dout=torch.randn_like(expected)
    gradients=backward(saved,dout)
    with TensorCanaries() as guards:
        actual,norm,state=forward(*x,save_state=True,**kwargs)
        result=backward(state,dout)
    # Ensure primary output/statistics buffers really received guards, not only
    # unrelated metadata temporaries. View-based returned outputs share pointers.
    pointers={item[-1] for item in guards.allocations}
    assert actual.data_ptr() in pointers and norm.data_ptr() in pointers
    guards.verify()
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    for name,value in result.items():
        torch.testing.assert_close(value,gradients[name],atol=2e-6,rtol=2e-5)


@pytest.mark.parametrize('side',['prefix','suffix'])
def test_canary_detector_positive_control(side):
    with TensorCanaries() as guards:
        tensors=[torch.empty(16,device='cuda') for _ in range(8)]
    guards.verify()
    storage,nbytes,*_=guards.allocations[0]
    storage[0 if side=='prefix' else 4096+nbytes]=0
    with pytest.raises(AssertionError,match=f'{side} overwrite'):
        guards.verify()
