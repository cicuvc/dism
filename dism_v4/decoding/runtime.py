"""NativeDecodeCache is production inference; DecodeCache is a diagnostic baseline.

The diagnostic class retains Python task packing for comparing control planes.
Neither path depends on the training extension.
"""
import torch
if __package__:
    from .build import load_cuda
else:
    from build import load_cuda


class DecodeCache:
    def __init__(self, heads, r, dv, capacity, tau, *, rebuild_interval=128,
                 sample_interval=None, materialize_threshold=None, rebuild_chunk=32, device='cuda'):
        if __package__:
            from .test_planner import load_planner
        else:
            from test_planner import load_planner
        Planner=load_planner()
        self.extension=load_cuda()
        self.r,self.dv,self.capacity=r,dv,capacity
        self.tau=[float(x) for x in tau]
        if len(self.tau)!=heads: raise ValueError('one tau per flattened batch/head')
        self.planners=[Planner(r,rebuild_interval,sample_interval or 4*r,
                               materialize_threshold or 5*r,t) for t in self.tau]
        self.keys=torch.empty((heads,capacity,r),device=device)
        self.values=torch.empty((heads,capacity,dv),device=device)
        self.summaries=torch.empty((0,r*dv+1),device=device)
        self.offsets=[0]*heads
        self.mat_counts=[0]*heads
        self.position=0
        self.rebuild_chunk=rebuild_chunk

    def rebuild(self):
        outputs=[];offset=0
        for h,p in enumerate(self.planners):
            p.rebuild()
            links,lens,positions,order,mats,samples=p.topology()
            mi=[-1]*len(links);si=mi.copy()
            for i,n in enumerate(mats): mi[n]=i
            for i,n in enumerate(samples): si[n]=len(mats)+i
            topo=torch.tensor([links,lens,positions,order,mi,si],dtype=torch.int32,device=self.keys.device)
            out=self.extension.rebuild(self.keys[h],self.values[h],topo,len(mats)+len(samples),self.tau[h],self.rebuild_chunk)
            outputs.append(out); self.offsets[h]=offset; self.mat_counts[h]=len(mats)
            offset+=out.shape[0]
        self.summaries=torch.cat(outputs)

    def memory_stats(self):
        """Live vector storage, excluding allocator reserve and transient rebuild peak."""
        return dict(raw_bytes=sum(t.numel()*t.element_size() for t in (self.keys,self.values)),
                    summary_bytes=self.summaries.numel()*self.summaries.element_size(),
                    summary_matrices=self.summaries.shape[0],
                    snapshot_positions=[p.snapshot_size for p in self.planners],
                    position=self.position, capacity=self.capacity,
                    rebuild_chunk=self.rebuild_chunk)

    @torch.no_grad()
    def step(self, labels_k, labels_q, sk, sq, v, reset=None):
        if self.position>=self.capacity: raise ValueError('cache capacity exceeded')
        if any(p.needs_rebuild() for p in self.planners): self.rebuild()
        # One batched label transfer, not one synchronization per head.
        h=len(self.planners)
        if reset is None: reset=torch.zeros_like(labels_k)
        labels=torch.stack((labels_k,labels_q,reset)).to(device='cpu',dtype=torch.int64).tolist()
        plans=[p.append(int(labels[0][j]),int(labels[1][j]),bool(labels[2][j])) for j,p in enumerate(self.planners)]
        size=max(1,max(map(len,plans)))
        ids=torch.zeros((h,size,2),dtype=torch.int32,pin_memory=True)
        weights=torch.zeros((h,size),dtype=torch.float32,pin_memory=True)
        fallback=torch.tensor([p.fallback for p in self.planners],dtype=torch.float32,pin_memory=True)
        for j,plan in enumerate(plans):
            for i,(kind,index,c) in enumerate(plan):
                if kind: index+=self.offsets[j]+(self.mat_counts[j] if kind==2 else 0)
                ids[j,i,0]=kind;ids[j,i,1]=index;weights[j,i]=c
        device=self.keys.device
        out=self.extension.query(self.keys,self.values,sk.float().contiguous(),v.float().contiguous(),sq.float().contiguous(),self.summaries,
                                 ids.to(device,non_blocking=True),weights.to(device,non_blocking=True),fallback.to(device,non_blocking=True),self.position)
        self.position+=1
        return out


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

        No prefill output is returned: use the existing forward kernel for it.
        Full per-document history is required, not a truncated tail.
        """
        self.native.prime(labels,sk,v)
