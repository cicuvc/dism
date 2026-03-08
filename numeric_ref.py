import sys
sys.path.append('build/linux/x86_64/release')

import torch
import dism_C

u = torch.randn((32,1),dtype = torch.float32)
qwq = dism_C.BaselineNoPEAttnState(u,u,u)

print(qwq.fwd_k_cache)