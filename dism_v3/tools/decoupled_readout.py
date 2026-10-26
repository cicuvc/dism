"""Isolated K-state-block / Q-read-tile prefill and analytic backward prototype.

No production dispatch changes. Query tiles share one FP32 matrix per K block.
The reverse-summary variant here gathers Q tiles per block; it does not allocate
a matrix per Q tile or perform shared-memory FP32 atomic reduction.
"""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import torch
import triton as tr
import triton.language as tl

from flash_dism.kernels.hard_backward_fused import _bf16_fp32_dot, _summary_scan
from flash_dism.inference.prefill_triton import TritonPrefillPlan


@tr.jit
def _key_summary(SK,V,H,A,KR,KP,KW,KL,RESET,R:tl.constexpr,D:tl.constexpr,
                 BK:tl.constexpr,BR:tl.constexpr,BD:tl.constexpr,PRECISE:tl.constexpr,BF16:tl.constexpr):
    block=tl.program_id(0);k=tl.arange(0,BK);r=tl.arange(0,BR);d=tl.arange(0,BD)
    length=tl.load(KL+block);row=tl.load(KR+block*BK+k);p=tl.load(KP+block*BK+k);w=tl.load(KW+block*BK+k)
    sk=tl.load(SK+row[:,None]*R+r[None,:],(k[:,None]<length)&(r[None,:]<R),0).to(tl.float32)
    v=tl.load(V+row[:,None]*D+d[None,:],(k[:,None]<length)&(d[None,:]<D),0).to(tl.float32)
    last=tl.load(KP+block*BK+length-1)
    scale=tl.where(k<length,w*tl.exp(tl.minimum(last-p,0.)),0.)
    if PRECISE and BF16:
        h=tl.trans(_bf16_fp32_dot(tl.trans(v),sk*scale[:,None]))
    elif BF16:
        h=tl.dot(tl.trans(sk).to(tl.bfloat16),(v*scale[:,None]).to(tl.bfloat16))
    else:
        h=tl.dot(tl.trans(sk),v*scale[:,None],input_precision="tf32x3")
    tl.store(H+block.to(tl.int64)*R*D+r[:,None]*D+d[None,:],h,(r[:,None]<R)&(d[None,:]<D))
    tl.store(A+block,tl.where(tl.load(RESET+block)!=0,0.,tl.exp(last)))


@tr.jit
def _forward_scan(H,A,OFFSETS,M:tl.constexpr,BLOCK:tl.constexpr):
    stream=tl.program_id(0);x=tl.arange(0,BLOCK)
    begin=tl.load(OFFSETS+stream);end=tl.load(OFFSETS+stream+1);state=tl.full((BLOCK,),0.,tl.float32)
    for block in range(begin,end):
        at=block.to(tl.int64)*M+x;summary=tl.load(H+at,x<M,0.);a=tl.load(A+block)
        tl.store(H+at,state,x<M);state=a*state+summary


@tr.jit
def _read_forward(SQ,SK,V,OUT,H,KR,KP,KW,KL,RESET,QR,QP,QW,QL,QCOUNT,QBLOCK,
                  R:tl.constexpr,D:tl.constexpr,BK:tl.constexpr,BQ:tl.constexpr,BR:tl.constexpr,BD:tl.constexpr,BF16:tl.constexpr):
    tile=tl.program_id(0);block=tl.load(QBLOCK+tile)
    q=tl.arange(0,BQ);k=tl.arange(0,BK);r=tl.arange(0,BR);d=tl.arange(0,BD)
    qlen=tl.load(QL+tile);klen=tl.load(KL+block)
    qi=tl.load(QR+tile*BQ+q);ki=tl.load(KR+block*BK+k)
    qp=tl.load(QP+tile*BQ+q);kp=tl.load(KP+block*BK+k)
    qw=tl.load(QW+tile*BQ+q);kw=tl.load(KW+block*BK+k);count=tl.load(QCOUNT+tile*BQ+q)
    sq=tl.load(SQ+qi[:,None]*R+r[None,:],(q[:,None]<qlen)&(r[None,:]<R),0).to(tl.float32)
    sk=tl.load(SK+ki[:,None]*R+r[None,:],(k[:,None]<klen)&(r[None,:]<R),0).to(tl.float32)
    v=tl.load(V+ki[:,None]*D+d[None,:],(k[:,None]<klen)&(d[None,:]<D),0).to(tl.float32)
    h=tl.load(H+block.to(tl.int64)*R*D+r[:,None]*D+d[None,:],(r[:,None]<R)&(d[None,:]<D),0)
    pair=tl.where((q[:,None]<qlen)&(k[None,:]<count[:,None]),qw[:,None]*kw[None,:]*tl.exp(tl.minimum(qp[:,None]-kp[None,:],0.)),0.)
    qs=tl.where(tl.load(RESET+block)==0,qw*tl.exp(qp),0.)
    if BF16:
        score=tl.dot(sq.to(tl.bfloat16),tl.trans(sk).to(tl.bfloat16))
        local=tl.dot((score*pair).to(tl.bfloat16),v.to(tl.bfloat16))
        history=tl.dot(sq.to(tl.bfloat16),h.to(tl.bfloat16))*qs[:,None]
    else:
        score=tl.dot(sq,tl.trans(sk),input_precision="tf32x3")
        local=tl.dot(score*pair,v,input_precision="tf32x3")
        history=tl.dot(sq,h,input_precision="tf32x3")*qs[:,None]
    tl.atomic_add(OUT+qi[:,None]*D+d[None,:],local+history,(q[:,None]<qlen)&(d[None,:]<D),sem="relaxed")


@tr.jit
def _query_summary(SQ,DO,G,OFFSETS,RESET,QR,QP,QW,QL,R:tl.constexpr,D:tl.constexpr,
                   BQ:tl.constexpr,BR:tl.constexpr,BD:tl.constexpr,BF16:tl.constexpr):
    block=tl.program_id(0);q=tl.arange(0,BQ);r=tl.arange(0,BR);d=tl.arange(0,BD)
    begin=tl.load(OFFSETS+block);end=tl.load(OFFSETS+block+1)
    total=tl.full((BR,BD),0.,tl.float32)
    # The first block has no dependence on a preceding state.
    if tl.load(RESET+block)==0:
        for tile in range(begin,end):
            length=tl.load(QL+tile);qi=tl.load(QR+tile*BQ+q)
            scale=tl.load(QW+tile*BQ+q)*tl.exp(tl.load(QP+tile*BQ+q))
            sq=tl.load(SQ+qi[:,None]*R+r[None,:],(q[:,None]<length)&(r[None,:]<R),0).to(tl.float32)
            do=tl.load(DO+qi[:,None]*D+d[None,:],(q[:,None]<length)&(d[None,:]<D),0).to(tl.float32)
            if BF16:piece=_bf16_fp32_dot(tl.trans(sq),do*scale[:,None])
            else:piece=tl.dot(tl.trans(sq),do*scale[:,None],input_precision="tf32x3")
            total+=piece
    tl.store(G+block.to(tl.int64)*R*D+r[:,None]*D+d[None,:],total,(r[:,None]<R)&(d[None,:]<D))


@tr.jit
def _query_backward(SQ,SK,V,DO,DSQ,DSK,DV,TAU,H,KR,KP,KW,KD,KL,RESET,KTAU,
                    QR,QP,QW,QD,QL,QCOUNT,QBLOCK,R:tl.constexpr,D:tl.constexpr,
                    BK:tl.constexpr,BQ:tl.constexpr,BR:tl.constexpr,BD:tl.constexpr,BF16:tl.constexpr):
    tile=tl.program_id(0);block=tl.load(QBLOCK+tile)
    q=tl.arange(0,BQ);k=tl.arange(0,BK);r=tl.arange(0,BR);d=tl.arange(0,BD)
    qlen=tl.load(QL+tile);klen=tl.load(KL+block)
    qi=tl.load(QR+tile*BQ+q);ki=tl.load(KR+block*BK+k)
    qp=tl.load(QP+tile*BQ+q);kp=tl.load(KP+block*BK+k)
    qw=tl.load(QW+tile*BQ+q);kw=tl.load(KW+block*BK+k)
    qdw=tl.load(QD+tile*BQ+q);kdw=tl.load(KD+block*BK+k);count=tl.load(QCOUNT+tile*BQ+q)
    sq=tl.load(SQ+qi[:,None]*R+r[None,:],(q[:,None]<qlen)&(r[None,:]<R),0).to(tl.float32)
    sk=tl.load(SK+ki[:,None]*R+r[None,:],(k[:,None]<klen)&(r[None,:]<R),0).to(tl.float32)
    v=tl.load(V+ki[:,None]*D+d[None,:],(k[:,None]<klen)&(d[None,:]<D),0).to(tl.float32)
    do=tl.load(DO+qi[:,None]*D+d[None,:],(q[:,None]<qlen)&(d[None,:]<D),0).to(tl.float32)
    pair=tl.where((q[:,None]<qlen)&(k[None,:]<count[:,None]),qw[:,None]*kw[None,:]*tl.exp(tl.minimum(qp[:,None]-kp[None,:],0.)),0.)
    if BF16:
        ds=tl.dot(do.to(tl.bfloat16),tl.trans(v).to(tl.bfloat16))*pair
        score=tl.dot(sq.to(tl.bfloat16),tl.trans(sk).to(tl.bfloat16))
    else:
        ds=tl.dot(do,tl.trans(v),input_precision="tf32x3")*pair
        score=tl.dot(sq,tl.trans(sk),input_precision="tf32x3")
    tau=tl.sum(tl.sum(ds*score*(qdw[:,None]+kdw[None,:]),1),0)
    if BF16:
        dv=tl.dot(tl.trans((score*pair).to(tl.bfloat16)),do.to(tl.bfloat16))
        dq=tl.dot(ds.to(tl.bfloat16),sk.to(tl.bfloat16))
        dk=tl.dot(tl.trans(ds.to(tl.bfloat16)),sq.to(tl.bfloat16))
    else:
        dv=tl.dot(tl.trans(score*pair),do,input_precision="tf32x3")
        dq=tl.dot(ds,sk,input_precision="tf32x3")
        dk=tl.dot(tl.trans(ds),sq,input_precision="tf32x3")
    h=tl.load(H+block.to(tl.int64)*R*D+r[:,None]*D+d[None,:],(r[:,None]<R)&(d[None,:]<D),0)
    if BF16:history=_bf16_fp32_dot(do,tl.trans(h))
    else:history=tl.dot(do,tl.trans(h),input_precision="tf32x3")
    qs=tl.where(tl.load(RESET+block)==0,qw*tl.exp(qp),0.)
    tau+=tl.sum(qdw*tl.sum(sq*history,1)*qs,0)
    tl.atomic_add(DSQ+qi[:,None]*R+r[None,:],dq+history*qs[:,None],(q[:,None]<qlen)&(r[None,:]<R),sem="relaxed")
    tl.atomic_add(DSK+ki[:,None]*R+r[None,:],dk,(k[:,None]<klen)&(r[None,:]<R),sem="relaxed")
    tl.atomic_add(DV+ki[:,None]*D+d[None,:],dv,(k[:,None]<klen)&(d[None,:]<D),sem="relaxed")
    tl.atomic_add(TAU+tl.load(KTAU+block),tau,sem="relaxed")


@tr.jit
def _key_backward(SK,V,DSK,DV,TAU,G,KR,KP,KW,KD,KL,KTAU,R:tl.constexpr,D:tl.constexpr,
                  BK:tl.constexpr,BR:tl.constexpr,BD:tl.constexpr,BF16:tl.constexpr):
    block=tl.program_id(0);k=tl.arange(0,BK);r=tl.arange(0,BR);d=tl.arange(0,BD)
    length=tl.load(KL+block);row=tl.load(KR+block*BK+k)
    p=tl.load(KP+block*BK+k);w=tl.load(KW+block*BK+k);dw=tl.load(KD+block*BK+k)
    last=tl.load(KP+block*BK+length-1);scale=tl.where(k<length,w*tl.exp(tl.minimum(last-p,0.)),0.)
    sk=tl.load(SK+row[:,None]*R+r[None,:],(k[:,None]<length)&(r[None,:]<R),0).to(tl.float32)
    v=tl.load(V+row[:,None]*D+d[None,:],(k[:,None]<length)&(d[None,:]<D),0).to(tl.float32)
    g=tl.load(G+block.to(tl.int64)*R*D+r[:,None]*D+d[None,:],(r[:,None]<R)&(d[None,:]<D),0)
    if BF16:
        dk=_bf16_fp32_dot(v,tl.trans(g))
        dv=tl.dot(sk.to(tl.bfloat16),g.to(tl.bfloat16))
    else:
        dk=tl.dot(v,tl.trans(g),input_precision="tf32x3")
        dv=tl.dot(sk,g,input_precision="tf32x3")
    tau=tl.sum(dw*tl.sum(sk*dk,1)*scale,0)
    tl.atomic_add(DSK+row[:,None]*R+r[None,:],dk*scale[:,None],(k[:,None]<length)&(r[None,:]<R),sem="relaxed")
    tl.atomic_add(DV+row[:,None]*D+d[None,:],dv*scale[:,None],(k[:,None]<length)&(d[None,:]<D),sem="relaxed")
    tl.atomic_add(TAU+tl.load(KTAU+block),tau,sem="relaxed")


def baseline_module(directory):
    """Keep baseline GPU kernels unchanged; allow virtual Q / physical K axes."""
    import flash_dism.kernels.hard_backward_fused as original
    source=Path(original.__file__).read_text()
    for old,new in [('sk.shape != sq.shape','sk.shape[1] != sq.shape[1]'),
                    ('value.shape[0] != self.n','value.shape[0] != self.physical_n'),
                    ('dout.shape != value.shape','dout.shape != (self.n, value.shape[1])')]:
        # Both the classic and compact consumers contain the same guard.
        if source.count(old)!=2:raise ValueError(f'baseline adapter anchor changed: {old}')
        source=source.replace(old,new)
    path=Path(directory)/'baseline_adapter.py';path.write_text(source)
    name='flash_dism.kernels._decoupled_baseline'
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
    return mod


class Executor:
    """Offline-prepared prototype, graph-replayable GPU execution only."""
    def __init__(self,raw,base_module,backward,precision='bf16'):
        if precision not in ('bf16','tf32x3'):raise ValueError('bf16 or tf32x3 required')
        self.stats=raw['statistics'];self.raw=raw;self.backward_mode=backward;self.bf16=precision=='bf16'
        s=self.stats;self.bk,self.bq=s['bk'],s['bq'];self.blocks=s['key_blocks'];self.tiles=s['query_tiles'];self.streams=s['state_streams']
        self.n=s['batch']*s['heads']*s['n'];self.nq=self.n*s['topk'];self.last_kernels=[]
        # The prototype and unchanged legacy kernels use int32 vector offsets.
        if self.nq>np.iinfo(np.int32).max//64:raise ValueError('vector address exceeds prototype int32 bound')
        legacy=dict(raw['multi'])
        if not backward and 'stream_ids' not in legacy:
            offsets=legacy['offsets'];spans=np.diff(offsets);longs=np.flatnonzero(spans>64)
            chunks=[np.arange(offsets[i],offsets[i+1],dtype=np.int64) for i in longs]
            lens=[len(x) for x in chunks]
            legacy.update(stream_ids=np.arange(len(spans),dtype=np.int64),chunk_ids=np.arange(offsets[-1],dtype=np.int64),
                          local_offsets=offsets[:-1],short_stream_ids=np.flatnonzero(spans<=64),long_stream_ids=longs,
                          long_chunk_ids=np.concatenate(chunks) if chunks else np.zeros(0,np.int64),
                          long_local_offsets=np.r_[0,np.cumsum(lens)[:-1]].astype(np.int64) if lens else np.zeros(0,np.int64))
        groups=[*raw['buckets'],legacy,raw['split']]
        # One pinned allocation and one H2D copy, including old/new metadata.
        spec=[];size=0;native_storage=raw.get('_storage')
        for group in groups:
            fields={}
            for name,x in group.items():
                if isinstance(x,np.ndarray):
                    if native_storage is None:
                        size=(size+7)//8*8;fields[name]=(size,x);size+=x.nbytes
                    else:
                        offset=x.ctypes.data-native_storage.ctypes.data
                        if offset<0 or offset+x.nbytes>native_storage.nbytes:raise ValueError('metadata is outside native arena')
                        fields[name]=(offset,x)
            spec.append(fields)
        if native_storage is None:
            host=torch.zeros(size,dtype=torch.uint8,pin_memory=True)
            for fields in spec:
                for start,x in fields.values():host[start:start+x.nbytes].copy_(torch.from_numpy(x.view(np.uint8)))
        else:
            host=torch.from_numpy(native_storage);size=host.numel()
            if not host.is_pinned():host=host.pin_memory()
        self.host_storage=host
        self.storage=host.to('cuda',non_blocking=True);self.metadata_bytes=size
        gpu=[]
        for fields in spec:
            gpu.append({name:self.storage[start:start+x.nbytes].view(torch.from_numpy(x).dtype) for name,(start,x) in fields.items()})
        self.data=gpu[-1];ready=torch.cuda.Event();ready.record()
        if backward:
            buckets=[dict(x,width=w) for x,w in zip(gpu[:3],(16,32,64))]
            prepared=SimpleNamespace(storage=self.storage,buckets=buckets,multi=gpu[3],statistics=s,ready=ready,
                                     host_multi_offsets=torch.from_numpy(raw['multi']['offsets']),wait=lambda:torch.cuda.current_stream().wait_event(ready))
            self.base=base_module.CompactFusedHardPrefillBackward.from_compact(prepared,mma_precision=precision)
            self.base.n=self.nq;self.base.physical_n=self.n
        else:
            self.base=TritonPrefillPlan.from_device_packed(gpu[3],dict(s,streams=s['multi_streams'],chunks=s['multi_chunks'],dispatch_threshold=64),mma_precision=precision)
            self.base.n=self.nq
            # Inputs are validated in forward(); only axis lengths differ for K>1.
            self.base._check_vectors=lambda *args:None
        torch.cuda.synchronize()
        if backward:self.base.ready=torch.cuda.Event();self.base.bucket_ready=None

    def _config(self,sq,sk,v,do=None):
        inputs=[sq,sk,v]+([] if do is None else [do])
        dtype=torch.bfloat16 if self.bf16 else torch.float32
        if any(x.ndim!=2 or x.device!=self.storage.device or x.dtype!=dtype or not x.is_contiguous() for x in inputs):
            raise ValueError('contiguous matrices on the plan CUDA device with matching precision required')
        r,d=sq.shape[1],v.shape[1]
        if r not in (16,32) or d not in (32,64):raise ValueError('R16/32 and DV32/64 required')
        if sq.shape[0]!=self.nq or sk.shape!=(self.n,r) or v.shape[0]!=self.n or (do is not None and do.shape!=(self.nq,d)):
            raise ValueError('physical K/V and virtual Q/dO shapes do not match plan')
        return r,d,max(32,r),max(32,d)

    def _launch(self,kernel,grid,*args,**kwargs):
        self.last_kernels.append(kernel[grid](*args,**kwargs,num_warps=4,num_stages=1))

    @torch.no_grad()
    def forward(self,sq,sk,v):
        r,d,br,bd=self._config(sq,sk,v)
        out=self.base.execute_dispatch(sq,sk,v);self.last_kernels=[]
        self.workspace_bytes=self.base.long_chunk_ids.numel()*2*br*bd*4
        if not self.blocks:return out
        a=self.data;h=torch.empty((self.blocks,r,d),device=sq.device,dtype=torch.float32);alpha=torch.empty(self.blocks,device=sq.device,dtype=torch.float32)
        self.workspace_bytes+=h.numel()*4+alpha.numel()*4
        self._launch(_key_summary,(self.blocks,),sk,v,h,alpha,*[a[k] for k in ('kr','kp','kw','kl','kreset')],r,d,self.bk,br,bd,False,self.bf16)
        self._launch(_forward_scan,(self.streams,),h,alpha,a['offsets'],r*d,r*d)
        if self.tiles:self._launch(_read_forward,(self.tiles,),sq,sk,v,out,h,*[a[k] for k in ('kr','kp','kw','kl','kreset','qr','qp','qw','ql','qcount','qblock')],r,d,self.bk,self.bq,br,bd,self.bf16)
        return out

    @torch.no_grad()
    def backward(self,sq,sk,v,do):
        r,d,br,bd=self._config(sq,sk,v,do)
        out=self.base.backward(sq,sk,v,do);self.last_kernels=list(self.base.last_kernels);self.workspace_bytes=self.base.matrix_snapshot_bytes
        if not self.blocks:return out
        a=self.data;states=torch.empty((2,self.blocks,r,d),device=sq.device,dtype=torch.float32);h,g=states.unbind();alpha=torch.empty(self.blocks,device=sq.device,dtype=torch.float32)
        self.workspace_bytes+=states.numel()*4+alpha.numel()*4
        self._launch(_key_summary,(self.blocks,),sk,v,h,alpha,*[a[k] for k in ('kr','kp','kw','kl','kreset')],r,d,self.bk,br,bd,True,self.bf16)
        self._launch(_query_summary,(self.blocks,),sq,do,g,*[a[k] for k in ('qoffsets','kreset','qr','qp','qw','ql')],r,d,self.bq,br,bd,self.bf16)
        self._launch(_summary_scan,(self.streams,1),h,g,alpha,a['offsets'],r*d,r*d)
        grads=[out[k] for k in ('sq_vec','sk_vec','v','rtau')]
        if self.tiles:self._launch(_query_backward,(self.tiles,),sq,sk,v,do,*grads,h,*[a[k] for k in ('kr','kp','kw','kd','kl','kreset','ktau','qr','qp','qw','qd','ql','qcount','qblock')],r,d,self.bk,self.bq,br,bd,self.bf16)
        self._launch(_key_backward,(self.blocks,),sk,v,out['sk_vec'],out['v'],out['rtau'],g,*[a[k] for k in ('kr','kp','kw','kd','kl','ktau')],r,d,self.bk,br,bd,self.bf16)
        return out
