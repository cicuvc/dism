import sys,torch
from flash_dism.forward import forward_core
from flash_dism.backward import backward_core
from test_v4_cuda import sample
x,delta=sample(gate='boundaries')
kwargs={} if '--legacy' in sys.argv else {'gate_delta':delta.cuda()}
out,_,state=forward_core(*[a.cuda() for a in x],save_state=True,ctas=1,**kwargs)
grad=backward_core(state,torch.ones_like(out),ctas=1)
assert torch.isfinite(out).all()
assert all(torch.isfinite(g).all() for g in grad.values())
torch.cuda.synchronize()
print('fixed forward/backward passed')
