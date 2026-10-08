"""Default batched hard prefill: parallel CPU planning and one GPU stream grid.

Reusable worker pool. Does not change model dispatch or initialize decode cache.
"""
import numpy as np
import torch
from .prefill_parallel import ParallelPrefillPlanner
from .prefill_triton import TritonPrefillPlan


def _merge_chunks(results, batch, heads, n):
    """Relabel local rows into flattened BNHD addresses; preserve event order."""
    fields={name:[] for name in ("rows","prefixes","weights","lengths","resets")}
    offsets=[np.zeros(1,dtype=np.int64)]
    chunk_base=0
    for index, (_, packed) in enumerate(results):
        b,h=divmod(index,heads)
        row=packed["rows"].astype(np.int64)
        is_key=row>=0
        local=np.where(is_key,row,-row-1)
        absolute=(b*n+local)*heads+h
        fields["rows"].append(np.where(is_key,absolute,-absolute-1).astype(np.int32))
        for name in ("prefixes","weights","lengths","resets"):
            fields[name].append(packed[name])
        offsets.append(packed["offsets"][1:]+chunk_base)
        chunk_base+=len(packed["lengths"])
    merged={name:np.concatenate(parts) for name,parts in fields.items()}
    merged["offsets"]=np.concatenate(offsets)
    return merged


class PreparedHardPrefill:
    def __init__(self, core, batch, heads, n):
        self.core=core
        self.batch,self.heads,self.n=batch,heads,n
        self.stream=torch.cuda.current_stream(core.device)

    def execute(self, sq, sk, value):
        expected=(self.batch,self.n,self.heads)
        if sq.ndim!=4 or sk.shape!=sq.shape or value.ndim!=4 or sq.shape[:3]!=expected or value.shape[:3]!=expected:
            raise ValueError("vectors must be sq/sk [B,N,H,R], value [B,N,H,DV]")
        if any(not x.is_contiguous() for x in (sq,sk,value)):
            raise ValueError("contiguous BNHD vectors required; no implicit transpose")
        if torch.cuda.current_stream(self.core.device)!=self.stream:
            raise ValueError("prepared prefill must execute on its upload stream")
        out=self.core.execute(sq.view(-1,sq.shape[-1]),sk.view(-1,sk.shape[-1]),
                              value.view(-1,value.shape[-1]))
        return out.view(*expected,value.shape[-1])

    def statistics(self):
        return dict(batch=self.batch,heads=self.heads,n=self.n,**self.core.statistics())


class HardDismPrefill:
    """Reuse this context-managed engine across layers/calls.

    Default 8 CPU workers, BF16 GEMMs, C16, FP32 output. Labels int32 [B,H,N],
    vectors contiguous BNHD. Single shared CUDA device/stream per prepared plan.
    Full-hard inference only; no finite soft delta, autograd or graph capture
    of preparation. No cache prime is implicit. workers=1 is a serial fallback.
    """
    def __init__(self, *, workers=8, chunk_size=16, mma_precision="bf16", device="cuda"):
        if chunk_size not in (16,32,64):raise ValueError("invalid chunk_size")
        if mma_precision not in ("bf16","tf32x3"):raise ValueError("invalid mma_precision")
        self.device=torch.device(device)
        if self.device.type!="cuda":raise ValueError("CUDA device required")
        if self.device.index is None:self.device=torch.device("cuda",torch.cuda.current_device())
        self.chunk_size,self.mma_precision=chunk_size,mma_precision
        self.workers=workers
        self.planner=ParallelPrefillPlanner(workers)

    def prepare(self, idx_q, idx_k, tau, *, reset=None):
        for x in (idx_q,idx_k):
            if not isinstance(x,torch.Tensor) or x.ndim!=3 or x.dtype!=torch.int32:
                raise ValueError("labels must be int32 tensors [B,H,N]")
        if idx_q.shape!=idx_k.shape or min(idx_q.shape)<=0 or idx_q.device!=idx_k.device:
            raise ValueError("nonempty matching label shapes/devices required")
        if idx_q.device.type!="cpu" and idx_q.device!=self.device:
            raise ValueError("labels must be CPU or on the engine CUDA device")
        b,h,n=idx_q.shape
        if b*h*n>np.iinfo(np.int32).max:
            raise ValueError("flattened row count exceeds int32")
        if reset is None:reset=torch.zeros_like(idx_q,dtype=torch.bool)
        if reset.shape!=idx_q.shape or reset.dtype!=torch.bool or reset.device!=idx_q.device:
            raise ValueError("reset must be bool [B,H,N] on the label device")
        # One bulk label D2H for the entire batch/head set.
        host=torch.stack((idx_q,idx_k,reset.to(torch.int32))).cpu().numpy()
        if isinstance(tau,torch.Tensor):tau=tau.detach().double().cpu().numpy()
        results=self.planner.plan(host[0],host[1],tau,reset=host[2].astype(bool),chunk_size=self.chunk_size)
        merged=_merge_chunks(results,b,h,n)
        del results  # Drop SAM programs/individual packed copies before upload.
        core=TritonPrefillPlan.from_packed(merged,b*h*n,device=self.device,
                                         chunk_size=self.chunk_size,mma_precision=self.mma_precision)
        return PreparedHardPrefill(core,b,h,n)

    def __call__(self, idx_q, idx_k, sq, sk, value, tau, *, reset=None):
        return self.prepare(idx_q,idx_k,tau,reset=reset).execute(sq,sk,value)

    def close(self):self.planner.close()
    def __enter__(self):return self
    def __exit__(self,*exc):self.close()
