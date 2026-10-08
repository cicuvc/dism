"""Native CPU SAM planner and CUDA vector cache."""
import torch
from .build import load_cuda


class NativeDecodeCache:
    """Native control plane. Labels are packed int32 [3,BH] (key,query,reset).

    Vector inputs are contiguous [BH,R]/[BH,DV] in cache_dtype, and outputs
    are FP32. This low-level cache is stream-bound and capacity-bounded.
    Tau and model weights must remain fixed; finite soft gates are unsupported.
    """
    def __init__(self, heads, r, dv, capacity, tau, *, rebuild_interval=128,
                 sample_interval=None, materialize_threshold=None, rebuild_chunk=32,
                 device='cuda', cache_dtype=torch.bfloat16):
        self.extension=load_cuda()
        self.native=self.extension.NativeCache(heads,r,dv,capacity,list(tau),
            rebuild_interval,4*r if sample_interval is None else sample_interval,
            5*r if materialize_threshold is None else materialize_threshold,rebuild_chunk,
            torch.empty(0,device=device,dtype=cache_dtype))

    @torch.no_grad()
    def step(self, labels, sk, sq, v):
        return self.native.step(labels,sk,sq,v)

    def memory_stats(self):
        return self.native.stats()

    @torch.no_grad()
    def prime(self, labels, sk, v):
        """Initialize from prefill: labels [N,3,BH], sk/v [BH,N,channel].

        No prefill output is returned: use HardDismPrefill separately.
        Full per-document history is required, not a truncated tail.
        """
        self.native.prime(labels,sk,v)
