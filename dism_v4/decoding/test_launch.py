"""Assert one CUDA compute kernel on an ordinary (non-rebuild) native step."""
import json
from pathlib import Path
import torch
from runtime import NativeDecodeCache

cache=NativeDecodeCache(2,16,32,32,[.7,.7],rebuild_interval=16)
labels=torch.zeros(3,2,device='cuda',dtype=torch.int32)
features=torch.randn(2,16,device='cuda',dtype=torch.bfloat16)
values=torch.randn(2,32,device='cuda',dtype=torch.bfloat16)
cache.step(labels,features,features,values)
torch.cuda.synchronize()
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as p:
    cache.step(labels,features,features,values)
    torch.cuda.synchronize()
path=Path(__file__).parent/'results'/'ordinary_step_trace.json'
p.export_chrome_trace(str(path))
trace=json.loads(path.read_text())
kernels=[e['name'] for e in trace['traceEvents'] if e.get('cat')=='kernel']
assert len(kernels)==1 and 'query_kernel' in kernels[0],kernels
print('ONE_QUERY_KERNEL_PASS',kernels)
