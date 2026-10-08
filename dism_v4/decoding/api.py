"""BNHD adapter for plugging the hard-only core into model decoding.

This adapter does not compute model projections, convolution state, GDN/SWA, or
vocabulary selection. Feed their existing hard labels and readout/value vectors.
It deliberately does not silently substitute this cache for FLA's beam cache.
"""
import torch
from .runtime import NativeDecodeCache


class HardDismDecoder:
    def __init__(self, batch, heads, readout_dim, value_dim, capacity, tau, *,
                 device='cuda', cache_dtype=torch.bfloat16, planner_backend='cpu', **tuning):
        self.batch,self.heads,self.r,self.dv=batch,heads,readout_dim,value_dim
        self.capacity,self.position=capacity,0
        self.dtype=cache_dtype
        self.device=torch.device(device)
        if self.device.index is None:
            self.device=torch.device('cuda',torch.cuda.current_device())
        if isinstance(tau,torch.Tensor):
            if tau.shape!=(heads,):raise ValueError('tau must have shape [H]')
            tau=tau.detach().double().cpu().tolist()
        if len(tau)!=heads:raise ValueError('one fixed tau per head')
        if planner_backend not in ('cpu','gpu'):
            raise ValueError('planner_backend must be cpu or gpu')
        self.planner_backend=planner_backend
        self.rebuild_interval=tuning.get('rebuild_interval',128)
        self.snapshot_position=0
        core_type=NativeDecodeCache
        if planner_backend=='gpu':
            from .gpu_runtime import GpuPlannerCache
            core_type=GpuPlannerCache
        self.core=core_type(batch*heads,readout_dim,value_dim,capacity,
            list(tau)*batch,device=self.device,cache_dtype=cache_dtype,**tuning)

    def _prepare(self,idx_q,idx_k,sk,v,reset):
        if sk.ndim!=4 or sk.shape[0]!=self.batch or sk.shape[2:]!=(self.heads,self.r):
            raise ValueError('sk must be [B,N,H,R]')
        n=sk.shape[1]
        if n+self.position>self.capacity:raise ValueError('cache capacity exceeded')
        if n<=0 or v.shape!=(self.batch,n,self.heads,self.dv):raise ValueError('v shape')
        for label in (idx_q,idx_k):
            if label.shape!=(self.batch,self.heads,n) or label.dtype!=torch.int32:
                raise ValueError('labels must be int32 [B,H,N]; convert bounded vocabulary argmax explicitly')
        if reset is None:reset=torch.zeros_like(idx_q,dtype=torch.bool)
        if reset.shape!=idx_q.shape or reset.dtype!=torch.bool:raise ValueError('reset must be bool [B,H,N]')
        if any(x.device!=self.device for x in (idx_q,idx_k,sk,v,reset)):raise ValueError('device changed')
        if sk.dtype!=self.dtype or v.dtype!=self.dtype:raise ValueError('cache payload dtype changed')
        packed=torch.stack((idx_k.permute(2,0,1),idx_q.permute(2,0,1),reset.permute(2,0,1)),dim=1)
        return packed.reshape(n,3,self.batch*self.heads).to(torch.int32).contiguous()

    @torch.no_grad()
    def append(self,idx_q,idx_k,sq,sk,v,*,reset=None):
        """Return FP32 [B,N,H,DV]. Each new token includes its current key.

        reset=True stops the incoming match before consuming the current query,
        exactly matching v4 hard delta. Finite soft deltas are not accepted.
        """
        if sq.shape!=sk.shape or sq.device!=self.device or sq.dtype!=self.dtype:
            raise ValueError('sq must match sk shape/device/dtype')
        labels=self._prepare(idx_q,idx_k,sk,v,reset)
        outputs=[]
        for i in range(sk.shape[1]):
            if self.planner_backend=='gpu' and self.position-self.snapshot_position>=self.rebuild_interval:
                self.core.rebuild()
                self.snapshot_position=self.position
            out=self.core.step(labels[i],sk[:,i].reshape(-1,self.r).contiguous(),
                sq[:,i].reshape(-1,self.r).contiguous(),v[:,i].reshape(-1,self.dv).contiguous())
            # Device planner reuses its output allocation. Preserve older tokens
            # when appending a chunk; no extra clone is needed for a single token.
            if self.planner_backend=='gpu' and sk.shape[1]>1:out=out.clone()
            outputs.append(out.reshape(self.batch,self.heads,self.dv))
            self.position+=1
        return torch.stack(outputs,dim=1)

    @torch.no_grad()
    def prime(self,idx_q,idx_k,sk,v,*,reset=None):
        """Initialize from a complete prefill history; compute outputs separately."""
        if self.position:raise ValueError('prime requires empty cache')
        labels=self._prepare(idx_q,idx_k,sk,v,reset)
        n=sk.shape[1]
        self.core.prime(labels,sk.permute(0,2,1,3).reshape(-1,n,self.r).contiguous(),
                        v.permute(0,2,1,3).reshape(-1,n,self.dv).contiguous())
        self.position=n
        self.snapshot_position=n

    def memory_stats(self):
        if self.planner_backend=='gpu':
            result=self.core.cpu.memory_stats()
            result.pop('last_tasks_per_head',None) # CPU planner is idle between rebuilds.
            result.update(position=self.core.check_status(),planner_backend='gpu')
            return result
        return self.core.memory_stats()
