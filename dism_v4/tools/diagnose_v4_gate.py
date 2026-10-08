import torch, importlib
from test_v4_cuda import sample,oracle
from flash_dism.forward import forward_core
from flash_dism.backward import backward_core
refmod=importlib.import_module('flash_dism.reference.dism_v4_ref')
for n in (512,2048):
 x,delta=sample(n,'hard','vary');x[7].zero_();x[8].zero_();x[-1].fill_(2.)
 delta[:,:,::32]=torch.inf;delta[:,:,1::32]=0.
 gpu=[torch.cat([a,a],0).cuda() if i!=11 else a.cuda() for i,a in enumerate(x)]
 dg=torch.cat([delta,delta*.25],0).cuda()
 out,_,state=forward_core(*gpu,gate_delta=dg,save_state=True,fp32_output=True,ctas=1)
 torch.manual_seed(951);do=torch.randn_like(out).bfloat16();actual=backward_core(state,do,fp32_output=True,ctas=1)['gate_delta'].cpu().double()
 exact=[];same=[]
 for b in range(2):
  _,g=oracle(x,dg[b:b+1].cpu(),do[b:b+1].cpu());exact.append(g['delta'])
  fn=refmod.dism_ref
  refmod.dism_ref=lambda *args:out[b:b+1].cpu().double()
  _,g=oracle(x,dg[b:b+1].cpu(),do[b:b+1].cpu());same.append(g['delta'])
  refmod.dism_ref=fn
 exact=torch.cat(exact);same=torch.cat(same)
 print(n,'reference norm',exact.norm().item(),'ideal error',(actual-exact).norm().item(),'same output error',(actual-same).norm().item(),'output correction effect',(same-exact).norm().item(),flush=True)
