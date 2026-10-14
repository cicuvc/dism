"""Process-local full-quadratic CUDA comparison for bench_model_training.

Loads only the SHA-verified pre-optimization artifact already in this worktree.
The production cu_flash_dism import, source and extension remain untouched.
Current interpolation (atomic/autotuned) and all other model code are retained,
so this isolates the core path rather than reconstructing an old training stack.
"""
import hashlib
import importlib.util
from pathlib import Path

import torch
import cu_flash_dism as cu
import flash_dism.compiler as c

BASELINE_SHA256 = 'e9e7bd6895c5f841f82594967f5afcc761d502acc77e65a34e838e43dd098a1e'


def install(path):
    path = Path(path).resolve()
    root = Path(__file__).resolve().parents[1]
    if not path.is_relative_to(root/'build'):
        raise ValueError('legacy artifact must belong to this worktree build directory')
    if hashlib.sha256(path.read_bytes()).hexdigest() != BASELINE_SHA256:
        raise ValueError('legacy CUDA artifact does not match the documented baseline SHA256')
    spec = importlib.util.spec_from_file_location('_dism_bench_legacy.cu_flash_dism',path)
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)

    def token_hard(hard,q):
        if q.shape[0] == hard.shape[0]:
            return hard[:,:,None].expand(-1,-1,q.shape[1]).contiguous()
        return hard.T.repeat_interleave(q.shape[1]//hard.shape[0],dim=1).unsqueeze(0).contiguous()

    @c._voc_forward.register_kernel('cuda')
    def forward(q,k,sq,sk,v,eq,ek,tau,beta,direction,hard,boundaries,layout,need_backward):
        b=float(beta)
        expanded=token_hard(hard,q)
        legacy.prepare_embedding(q,k,sq,sk,v,eq,ek,tau,direction,expanded,False)
        query,key,qe,ke=c.prepare_embedding(q,k,eq,ek)
        qfk,kfq,lk,lq,_,_,ik,iq=c.emb_fwd_wrapper(query,key,qe,ke,b,tau.detach())
        qsk,ksk=(qfk,kfq) if b==1. else (qfk*b,kfq*b)
        qv,kv=c.select_operands(q,k,qsk,ksk,direction)
        table=layout.view(-1,7) if layout is not None else c._table(boundaries,q.shape[1])
        output,norm,state=legacy.core_forward(
            (qv,kv,sq,sk,v,lq.transpose(1,2),lk.transpose(1,2),iq,ik,direction,expanded,tau),
            table,0,True,False,True)
        return [output,qfk,kfq,lk,lq,ik,iq,state['vertical'],state['operands'][12],norm,
                output.clone(),cu.hold_training_plan(None)]

    @c.voc_backward.register_kernel('cuda')
    def backward(inputs,saved,boundaries,dout,layout=None):
        q,k,sq,sk,v,eq,ek,tau,beta,direction,hard=inputs
        b=float(beta)
        output,qfk,kfq,lk,lq,ik,iq,vertical,boundary,norm,_,_=saved
        expanded=token_hard(hard,q)
        query,key,qe,ke=c.prepare_embedding(q,k,eq,ek)
        qsk,ksk=(qfk,kfq) if b==1. else (qfk*b,kfq*b)
        qv,kv=c.select_operands(q,k,qsk,ksk,direction)
        raw=(qv,kv,sq,sk,v,lq.transpose(1,2),lk.transpose(1,2),iq,ik,direction,expanded,tau)
        packed=boundaries is not None or layout is not None
        prepared=list(legacy.prepare_operands(raw,packed,True))+[boundary]
        g,_=legacy.core_backward(prepared,vertical,output,norm.flatten() if packed else norm,
            dout.to(torch.bfloat16).contiguous(),
            layout.view(-1,7) if layout is not None else c._table(boundaries,q.shape[1]),0,False,False)
        doq,dok=c.split_interpolation_gradients(g['q_vec'],g['k_vec'],direction)
        dq,dk,dqe,dke=c.emb_bwd_wrapper(query,key,qe,ke,qfk,kfq,lk,lq,doq*b,dok*b,
            g['k_lse'].transpose(1,2).float(),g['q_lse'].transpose(1,2).float(),b,tau.detach())
        dq,dk,dsq=c.merge_token_gradients(g['q_vec'],g['k_vec'],dq,dk,g['sq_vec'],direction)
        dqe,dke=c.cast_vocabulary_gradients(dqe,dke,eq,ek)
        return [x.contiguous() for x in (dq,dk,dsq,g['sk_vec'].to(sk.dtype),g['v'].to(v.dtype),
                                        dqe,dke,g['rtau'].to(tau.dtype))]
    return dict(path=str(path),sha256=BASELINE_SHA256,
                comparison='old full-quadratic core; current atomic interpolation and remaining model')
