"""Invocation-owned top-k event planning, eager and CUDA Graph training.

The CPU sees committed argmax history plus local candidates, never router
probabilities. Probabilities weight query events on device after publication.
"""
import atexit
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import torch

from .inference.build import load_topk_training
from .inference.training_planner import PlanBudgetExceeded
from .planner_graph import GraphPlannerSlot, GraphPlanPayload, upload_plan
from .kernels.topk_event import TopkEventBackward
from .hard_training import HardPlanTicket, _Lease


class NativeTopkPlanner:
    def __init__(self, workers=16):
        self.native = load_topk_training().Planner(workers)

    def prepare(self, labels, tau, candidates, hard, documents, budget):
        def allocate(size):
            if size > budget:
                raise PlanBudgetExceeded(f'topk metadata requires {size} bytes, budget {budget}')
            return torch.empty(size,dtype=torch.uint8,pin_memory=True).numpy()
        return self.native.prepare(labels.numpy(),tau.double().numpy(),candidates.numpy(),
            candidates.shape[-1],True,64,64,1,allocate,hard.numpy().view(np.uint8),documents)


class _Host:
    def __init__(self,raw):
        self.raw=raw;self.statistics=raw['statistics']
        self.storage=torch.from_numpy(raw['_storage'])
        self.groups=[*raw['buckets'],raw['multi'],raw['split']]
        self.multi={k:torch.from_numpy(v) for k,v in raw['multi'].items() if isinstance(v,np.ndarray)}

    def upload(self,*,device,stream):
        storage=self.storage.to(device,non_blocking=True)
        groups=[]
        for group in self.groups:
            dst={}
            for name,x in group.items():
                if not isinstance(x,np.ndarray):continue
                offset=x.ctypes.data-self.raw['_storage'].ctypes.data
                if offset<0 or offset+x.nbytes>self.storage.numel():raise ValueError('invalid arena view')
                dst[name]=storage[offset:offset+x.nbytes].view(torch.from_numpy(x).dtype)
            groups.append(dst)
        return SimpleNamespace(storage=storage,buckets=groups[:3],multi=groups[3],split=groups[4])


def upload_topk(raw,*,device,stream,budget):
    """Publish immutable sources and private weighted metadata in one descriptor."""
    s=raw['statistics'];blocks=s['key_blocks']
    slots=sum(c*w for c,w in zip(s['bucket_blocks'],(16,32,64)))+s['multi_chunks']*64
    extra=4*(slots+s['query_tiles']*64+blocks*(2*32*64+1))+32*16
    if max(slots,blocks*64,s['query_tiles']*64,s['batch']*s['heads']*s['n']*s['topk']*64)>2**31-1:
        raise ValueError('topk plan exceeds int32 addressing')
    # All queued transfers retire before a partial allocation can unwind.
    try:
        payload=upload_plan(_Host(raw),device=device,stream=stream,budget_bytes=budget-extra)
        with torch.cuda.stream(stream):
            def empty(n):
                x=torch.empty(n,dtype=torch.float32,device=device);payload.owners.append(x);return x
            words=payload.descriptor.tolist();sources=[]
            for i,group in enumerate([*payload.device.buckets,payload.device.multi]):
                sources.append(group['weights'].data_ptr())
                words[16*i+4]=empty(group['weights'].numel()).data_ptr()
            split=payload.device.split
            words+=sources+[blocks,s['query_tiles'],s['state_streams']]
            fields=('kr','kp','kw','kd','kl','kreset','ktau','offsets','qr','qp','qw','qd','ql','qcount','qblock','qoffsets')
            words += [split[k].data_ptr() for k in fields]+[0]
            words += [empty((blocks,32,64)).data_ptr(),empty((blocks,32,64)).data_ptr(),empty(blocks).data_ptr(),split['qw'].data_ptr()]
            words[81]=empty(split['qw'].numel()).data_ptr()
            words += [0]*(96-len(words))
            desc=torch.empty(len(words),dtype=torch.int64,pin_memory=True)
            desc.copy_(torch.from_numpy(np.asarray(words,dtype=np.int64)))
        payload.owners.extend((payload.descriptor,desc))
        return GraphPlanPayload(payload.host,payload.device,payload.owners,desc,payload.bytes+extra)
    except Exception:
        stream.synchronize()
        raise


class TopkGraphPlannerSlot(GraphPlannerSlot):
    def __init__(self,runtime,rows,heads,row_tokens,topk):
        if topk not in (1,2,4) or rows*heads*row_tokens*topk*64>2**31-1:
            raise ValueError('K=1/2/4 and int32 virtual vector addressing required')
        super().__init__(runtime,rows,heads,row_tokens)
        self.topk=topk
        self.budget-=2*rows*heads*row_tokens*topk*4+32*8
        if self.budget<1024:raise ValueError('candidate snapshot exceeds slot budget')
        host=torch.empty((rows,heads,row_tokens,topk),dtype=torch.int32,pin_memory=True)
        self.hosts+= (host,)
        self.snapshots+= (torch.empty_like(host,device=self.device),)
        self.descriptor=torch.zeros(96,dtype=torch.int64,device=self.device)
        self.consumer=TopkEventBackward(self.descriptor,self.choice)
        if not hasattr(runtime,'topk_native'):
            runtime.topk_native=NativeTopkPlanner(runtime.workers)

    def bind(self,kind,*,layout=None):
        if kind!='vocabulary':raise ValueError('topk slots require a vocabulary binding')
        return super().bind(kind,layout=layout)

    def snapshot_topk(self,iq,ik,hard,tau,ids):
        if ids.shape!=self.snapshots[3].shape or ids.dtype!=torch.int32 or ids.device!=self.device or not ids.is_contiguous():
            raise ValueError('candidate snapshot geometry mismatch')
        self.snapshots[3].copy_(ids)
        super().snapshot(iq,ik,hard,tau)

    def _produce(self):
        with torch.cuda.device(self.device),torch.no_grad():
            try:
                self.input_ready.synchronize()
                if self.cancelled.is_set():self.reason='cancelled';return
                labels,hard,tau,ids=self.hosts
                raw=self.runtime.topk_native.prepare(labels,tau,ids,hard,self.documents,self.budget//2)
                if self.cancelled.is_set():self.reason='cancelled';return
                with torch.cuda.stream(self.upload):
                    self.payload=upload_topk(raw,device=self.device,stream=self.upload,budget=self.budget)
                    self.descriptor.copy_(self.payload.descriptor,non_blocking=True)
                    self.runtime.writer.publish(self.upload,self.ready,self.epoch)
            except PlanBudgetExceeded as exc:self.reason=str(exc)
            finally:self.upload.synchronize()

    def backward_topk(self,sq,sk,v,do,prob,fallback):
        self.sample()
        event=self.consumer.backward(sq,sk,v,do,prob)
        quadratic=fallback(self.choice)
        return tuple(a+b for a,b in zip(event,quadratic))


class _EagerTicket:
    def __init__(self,planner,future,lease,abandoned):
        self.planner,self.future,self.lease,self.abandoned=planner,future,lease,abandoned
        self.consumer_stream=None

    def backward(self,sq,sk,v,do,prob,fallback):
        stream=torch.cuda.current_stream(self.planner.device)
        if self.consumer_stream is not None and self.consumer_stream!=stream.cuda_stream:
            raise RuntimeError('retained topk event backward must use one consumer stream')
        self.consumer_stream=stream.cuda_stream
        policy=self.planner.policy
        prepared=self.future.result() if policy=='event' or self.future.done() else None
        if policy=='quadratic' or prepared is None or (policy=='ready' and not prepared.ready.query()):
            self.abandoned.set();HardPlanTicket._record_route('q');return fallback(None)
        torch.cuda.current_stream().wait_event(prepared.ready)
        # Retained autograd graphs may run backward again. Preserve immutable
        # CPU metadata and keep all allocations live on every consumer stream.
        for owner in [prepared.descriptor,prepared.choice,prepared.payload.device.storage,*prepared.payload.owners]:
            if isinstance(owner,torch.Tensor) and owner.is_cuda:owner.record_stream(torch.cuda.current_stream())
        HardPlanTicket._record_route('e')
        return prepared.kernel.backward(sq,sk,v,do,prob)


class AsyncTopkPlanner:
    def __init__(self,device,workers=16,max_inflight=32,policy='ready'):
        if policy not in ('ready','event','quadratic'):raise ValueError('invalid hard backward policy')
        self.device=torch.device(device);self.policy=policy
        self.native=NativeTopkPlanner(workers)
        self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='dism-topk-plan')
        self.slots=threading.BoundedSemaphore(max_inflight)
        self.stream=torch.cuda.Stream(device=self.device);self.closed=False

    def close(self):
        self.closed=True;self.pool.shutdown(wait=True);self.stream.synchronize()

    def submit(self,iq,ik,hard,tau,ids,layout):
        if torch.cuda.is_current_stream_capturing():raise RuntimeError('topk capture requires an explicit graph binding')
        if self.closed:raise RuntimeError('topk planner is closed')
        if not self.slots.acquire(blocking=False):raise RuntimeError('topk planner max_inflight reached')
        lease=_Lease(self.slots)
        b,h,n=iq.shape
        if layout is None:documents=np.array([(r,0,n) for r in range(b)],dtype=np.int64)
        else:
            documents=[];cursor=0
            for start,length,*_ in layout.table.tolist():
                if start!=cursor or length<0 or start%256 or length%256:raise ValueError('invalid aligned documents')
                cursor+=length
                if length:
                    row=start//n
                    if (start+length-1)//n!=row:raise ValueError('document crosses training row')
                    documents.append((row,start%n,start%n+length))
            if cursor!=b*n:raise ValueError('incomplete documents')
            documents=np.array(documents,dtype=np.int64).reshape(-1,3)
        sources=(torch.stack((iq,ik)),hard,tau,ids)
        hosts=tuple(torch.empty(x.shape,dtype=x.dtype,pin_memory=True) for x in sources)
        # Copies on the producer stream finish before any subsequent mutation.
        try:
            for host,src in zip(hosts,sources):host.copy_(src,non_blocking=True)
            ready=torch.cuda.Event();ready.record();abandoned=threading.Event()
            future=self.pool.submit(self._prepare,hosts,documents,ready,abandoned,lease)
            return _EagerTicket(self,future,lease,abandoned)
        except Exception:
            # Submission/partial-copy failure must not release pinned targets
            # while the producer stream can still write into them.
            torch.cuda.current_stream(self.device).synchronize()
            raise

    def _prepare(self,hosts,documents,ready,abandoned,lease):
        with torch.cuda.device(self.device),torch.no_grad():
            ready.synchronize()
            if abandoned.is_set():return None
            labels,hard,tau,ids=hosts
            raw=self.native.prepare(labels,tau,ids,hard,documents,2**30)
            if abandoned.is_set():return None
            try:
                with torch.cuda.stream(self.stream):
                    payload=upload_topk(raw,device=self.device,stream=self.stream,budget=2**31)
                    descriptor=payload.descriptor.to(self.device,non_blocking=True)
                    choice=torch.ones(1,device=self.device,dtype=torch.int32)
                    event=torch.cuda.Event();event.record()
                    result=SimpleNamespace(payload=payload,descriptor=descriptor,choice=choice,ready=event,
                                           kernel=TopkEventBackward(descriptor,choice))
                # Pinned owners must survive even if a discarded Future's result
                # becomes unreachable before an outstanding H2D completes.
                self.stream.synchronize()
                return result
            except Exception:
                self.stream.synchronize();raise


_runtimes={}
_lock=threading.Lock()

def default_topk_planner(device):
    if torch.cuda.is_current_stream_capturing():raise RuntimeError('topk planner capture needs graph_binding')
    key=(os.getpid(),torch.device(device))
    with _lock:
        if key not in _runtimes:
            _runtimes[key]=AsyncTopkPlanner(device,workers=int(os.environ.get('DISM_PLANNER_WORKERS','16')),
                max_inflight=int(os.environ.get('DISM_PLANNER_MAX_INFLIGHT','32')),policy=os.environ.get('DISM_HARD_BACKWARD','ready'))
        return _runtimes[key]


def shutdown_topk_planners():
    with _lock:current=list(_runtimes.values());_runtimes.clear()
    for runtime in current:runtime.close()

atexit.register(shutdown_topk_planners)
