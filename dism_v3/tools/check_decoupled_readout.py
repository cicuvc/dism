"""Dense FP64 oracle and unchanged-legacy-format checks for decoupled readout."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from decoupled_readout import Executor,baseline_module
from check_hard_prefill_training import assert_gradient,metrics


def load_native(path):
    spec=importlib.util.spec_from_file_location('_dism_decoupled',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def oracle(q,k,candidates,tau,xs):
    n,topk=candidates.shape;history=np.zeros((n,n),np.int32)
    for i in range(n):
        for j in range(n):
            if q[i]==k[j]:history[i,j]=1+(history[i-1,j-1] if i and j else 0)
    lengths=np.zeros((n,topk,n),np.int64)
    for i in range(n):
        for a in range(topk):
            for j in range(i+1):
                if candidates[i,a]==k[j]:lengths[i,a,j]=1+(history[i-1,j-1] if i and j else 0)
    sq,sk,v=[torch.tensor(x.cpu().double().numpy(),requires_grad=True) for x in xs[:3]]
    t=torch.tensor(float(tau),dtype=torch.float64,requires_grad=True)
    f=torch.cat((torch.zeros(1,dtype=torch.float64),torch.exp(t*torch.arange(1,n+1,dtype=torch.float64)).cumsum(0)))
    weights=f[torch.from_numpy(lengths)];weights=weights/(1+weights.sum(-1,keepdim=True))
    score=torch.einsum('iar,jr->iaj',sq.reshape(n,topk,-1),sk)
    out=torch.einsum('iaj,jd->iad',score*weights,v).reshape(n*topk,-1)
    grad=torch.autograd.grad(out,(sq,sk,v,t),xs[3].cpu().double())
    return out.detach(),tuple(g.detach() for g in grad)


def legacy_check(planner):
    from flash_dism.inference.training_planner import NativeTrainingPlanner
    from flash_dism.inference.prefill_native import NativePrefillPlanner
    rng=np.random.default_rng(81);count=0
    for n in [1,65,257]:
        labels=rng.integers(0,4,(2,2,3,n),dtype=np.int32);tau=np.array([0.,.5,-.2]);cand=np.empty(0,np.int32)
        for grad in [False,True]:
            raw=planner.prepare(labels,tau,cand,1,grad,64,32,-1)
            if grad:
                old=NativeTrainingPlanner(2).prepare(labels,tau,compact=True,chunk_size=64)
                for a,b in zip(raw['buckets'],old.buckets):
                    for field in ['rows','prefixes','weights','dlog','lengths','stream_heads']:np.testing.assert_array_equal(a[field],b[field].numpy())
                for field,b in old.multi.items():np.testing.assert_array_equal(raw['multi'][field],b.numpy())
            else:
                old=NativePrefillPlanner(2).plan(labels,tau)
                for field in ['rows','prefixes','weights','lengths','resets','offsets']:np.testing.assert_array_equal(raw['multi'][field],old.data[field].numpy())
            count+=1
    return count


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--directory',type=Path,required=True)
    ap.add_argument('--quick',action='store_true')
    ap.add_argument('--rank',type=int,choices=[16,32],default=16)
    ap.add_argument('--value-dim',type=int,choices=[32,64],default=32)
    ap.add_argument('--native',type=Path)
    ap.add_argument('--direct',action='store_true')
    args=ap.parse_args();root=args.directory;torch.set_num_threads(1)
    module=load_native(args.native or root/'native/_dism_decoupled.so');planner=module.Planner(4);base=baseline_module(root)
    def prepare(*params):
        if args.direct:return planner.prepare_direct(*params,allocator=lambda size:torch.empty(size,dtype=torch.uint8,pin_memory=True).numpy())
        return planner.prepare(*params)
    result=dict(legacy_cases=legacy_check(planner),cases=[])
    print('legacy exact PASS',result['legacy_cases'],flush=True)
    rng=np.random.default_rng(31)
    cases=[(65,'repeat',1,.5,32,32,'bf16'),(257,'repeat',4,.5,64,64,'bf16'),(65,'sparse',4,.5,64,32,'bf16'),(17,'empty',4,0.,64,64,'bf16')] if args.quick else [
        (n,pattern,k,tau,bk,bq,precision)
        for n,pattern,k,tau in [(17,'random',1,0.),(65,'repeat',1,.5),(129,'random',4,-.2),(257,'sparse',4,.5),(129,'repeat',4,.5)]
        for bk,bq in [(16,16),(32,32),(64,32),(64,64)] for precision in ['bf16','tf32x3']]
    for n,pattern,k,tau,bk,bq,precision in cases:
        labels=rng.integers(0,7,(2,1,1,n),dtype=np.int32)
        if pattern=='repeat':labels.fill(0)
        elif pattern=='sparse':labels[0].fill(0);labels[1,0,0]=np.arange(n)+1;labels[1,0,0,[0,n//2]]=0
        elif pattern=='empty':labels[1]+=1000
        candidates=np.stack([(labels[0]+a)%7 for a in range(k)],axis=-1).astype(np.int32)
        dtype=torch.bfloat16 if precision=='bf16' else torch.float32;torch.manual_seed(72)
        sq=(torch.randn(n,args.rank,device='cuda')*.2).to(dtype).repeat_interleave(k,0)
        xs=[sq,*[(torch.randn(n,d,device='cuda')*.2).to(dtype) for d in [args.rank,args.value_dim]],(torch.randn(n*k,args.value_dim,device='cuda')*.2).to(dtype)]
        expected,grad=oracle(*labels[:,0,0],candidates[0,0],tau,xs)
        for threshold in [0,1]:
            record=dict(n=n,pattern=pattern,topk=k,tau=tau,bk=bk,bq=bq,precision=precision,threshold=threshold,errors={})
            raw=prepare(labels,np.array([tau]),candidates,k,False,bk,bq,threshold)
            forward=Executor(raw,base,False,precision);out=forward.forward(*xs[:3])
            assert_gradient(out.cpu(),expected,'out');record['errors']['out']=metrics(out.cpu(),expected)
            raw=prepare(labels,np.array([tau]),candidates,k,True,bk,bq,threshold)
            backward=Executor(raw,base,True,precision);got=backward.backward(*xs)
            for name,ref in zip(['sq_vec','sk_vec','v','rtau'],grad):
                ref=ref.reshape(got[name].shape);assert_gradient(got[name].cpu(),ref,name)
                record['errors'][name]=metrics(got[name].cpu(),ref)
            torch.testing.assert_close(got['rtau'].cpu().double(),grad[-1].reshape(1),rtol=2e-4,atol=2e-6)
            result['cases'].append(record)
            name=('direct_' if args.direct else '')+('checks_quick.json' if args.quick else 'checks.json')
            (root/name).write_text(json.dumps(result,indent=2)+'\n')
        print('PASS',n,pattern,k,bk,bq,precision,flush=True)


if __name__=='__main__':main()
