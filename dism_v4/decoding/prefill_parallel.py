"""Opt-in batch/head CPU planning using a reusable thread pool.

Native planning and chunk compilation release the GIL; Python list conversion
and exported-array copies do not. No GPU access or implicit model dispatch.
"""
from concurrent.futures import ThreadPoolExecutor
import numpy as np
if __package__:
    from .prefill import HardPrefillPlan, load_prefill
else:
    from prefill import HardPrefillPlan, load_prefill


class ParallelPrefillPlanner:
    def __init__(self, workers=8):
        if not isinstance(workers,int) or workers<1:
            raise ValueError("workers must be a positive integer")
        load_prefill()  # Build/import once before any workers enter native code.
        self.pool=ThreadPoolExecutor(max_workers=workers,thread_name_prefix="dism-prefill")

    def plan(self, queries, keys, tau, *, reset=None, chunk_size=16):
        """Return [(HardPrefillPlan, packed_chunks), ...] in B-major/H-minor order.

        Labels [B,H,N], tau [H] or [B,H], optional bool reset [B,H,N].
        Returned programs own their data. Do not mutate inputs during this call.
        Includes chunk packing; excludes tensor upload and GPU execution.
        """
        q,k=np.asarray(queries),np.asarray(keys)
        if q.ndim!=3 or q.shape!=k.shape or min(q.shape)<=0:
            raise ValueError("nonempty equally shaped labels [B,H,N] required")
        b,h,n=q.shape
        tau=np.asarray(tau,dtype=np.float64)
        if tau.shape==(h,):tau=np.broadcast_to(tau,(b,h))
        if tau.shape!=(b,h):raise ValueError("tau must have shape [H] or [B,H]")
        if reset is None:reset=np.zeros(q.shape,dtype=bool)
        reset=np.asarray(reset)
        if reset.shape!=q.shape or reset.dtype!=np.bool_:raise ValueError("reset must be bool [B,H,N]")
        if chunk_size not in (16,32,64):raise ValueError("chunk_size must be 16,32,64")
        def build(index):
            bi,hi=divmod(index,h)
            p=HardPrefillPlan(q[bi,hi],k[bi,hi],tau[bi,hi],reset=reset[bi,hi])
            return p,p.native.chunks(chunk_size)
        return list(self.pool.map(build,range(b*h)))

    def close(self):
        self.pool.shutdown(wait=True)

    def __enter__(self):return self
    def __exit__(self,*exc):self.close()
