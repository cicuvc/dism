"""Small sanitizer workload with rebuilds, both dtypes, reset, and prime."""
import torch
from runtime import NativeDecodeCache,load_cuda

torch.manual_seed(337)
for dtype in (torch.float32,torch.bfloat16):
    h,n,r,d=2,41,16,32
    labels=torch.randint(0,3,(n,3,h),device='cuda',dtype=torch.int32)
    labels[:,2]=0;labels[::13,2]=1
    sk=torch.randn(n,h,r,device='cuda').to(dtype);sq=torch.randn_like(sk)
    v=torch.randn(n,h,d,device='cuda').to(dtype)
    cache=NativeDecodeCache(h,r,d,n,[0.,.7],rebuild_interval=7,sample_interval=3,materialize_threshold=19,rebuild_chunk=128,cache_dtype=dtype)
    linear=load_cuda().LinearCache(h,r,d,n,[0.,.7],sk[0])
    for c in (cache,linear):c.prime(labels[:11],sk[:11].transpose(0,1).contiguous(),v[:11].transpose(0,1).contiguous())
    for i in range(11,n):
        a=cache.step(labels[i],sk[i],sq[i],v[i]);b=linear.step(labels[i],sk[i],sq[i],v[i])
        torch.testing.assert_close(a,b,atol=3e-4,rtol=3e-4)
torch.cuda.synchronize()
print('SANITIZER_SMOKE_PASS')
