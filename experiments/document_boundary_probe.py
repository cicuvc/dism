"""Frozen-checkpoint document boundary interventions; no training or kernel edits."""
import os
os.environ.setdefault('DISM_TILE_LSE','tanh_finite')
os.environ.setdefault('DISM_BWD_OPT','13')
import json
import hashlib
from pathlib import Path
from contextlib import contextmanager
import numpy as np
import torch
from numba import njit


@njit
def hard_mass(q,k,tau,boundary):
    """Exact natural-log hard recurrence using actual labels; O(H*N) scratch."""
    heads,n=q.shape
    result=np.zeros((heads,n-boundary,3),np.float64)
    for h in range(heads):
        chain=np.zeros(n,np.int32)
        scores=np.empty(n,np.float64)
        for i in range(n):
            maximum=0.
            for j in range(i,-1,-1):
                chain[j]=(chain[j-1]+1 if j>0 and i>0 else 1) if q[h,i]==k[h,j] else 0
                l=chain[j]
                w=-np.inf if l==0 else tau[h]*l+np.log(-np.expm1(-tau[h]*l))-np.log(-np.expm1(-tau[h]))
                scores[j]=w
                maximum=max(maximum,w)
            if i>=boundary:
                total=np.exp(-maximum); cross=0.;long_cross=0.;matches=0.
                for j in range(i+1):
                    weight=np.exp(scores[j]-maximum)
                    total+=weight
                    if j<boundary:
                        cross+=weight
                        if chain[j]>=4:long_cross+=weight
                        if chain[j]>0:matches+=1
                result[h,i-boundary,0]=cross/total
                result[h,i-boundary,1]=long_cross/total
                result[h,i-boundary,2]=matches/boundary
    return result


def test_hard_mass():
    q=np.array([[0,1,0,1,0,1]],dtype=np.int32);k=q.copy();tau=np.array([.7])
    got=hard_mass(q,k,tau,3)
    w=torch.full((6,6),-torch.inf,dtype=torch.float64)
    for i in range(6):
        for j in range(i+1):
            if q[0,i]==k[0,j]:
                w[i,j]=.7+(torch.nn.functional.softplus(w[i-1,j-1]) if i and j else 0.)
    p=torch.softmax(torch.cat([w,torch.zeros(6,1,dtype=torch.float64)],1),1)
    np.testing.assert_allclose(got[0,:,0],p[3:,:3].sum(1).numpy(),atol=1e-12)


@contextmanager
def intervention(model,boundary,mode):
    from dism_v2 import autograd
    changes=[]
    original_core=autograd.voc_dism
    try:
        if mode=='dism_core':
            def core(q,k,v,*args,**kwargs):
                return torch.cat([original_core(q[:,:,a:b].contiguous(),k[:,:,a:b].contiguous(),
                    v[:,:,a:b].contiguous(),*args,**kwargs) for a,b in [(0,boundary),(boundary,q.shape[2])]],2)
            autograd.voc_dism=core
        for block in model.blocks:
            module=block.dism if mode in ('dism','both') else None
            if module is not None:
                old=module.forward
                def split_dism(x,*args,_old=old,**kwargs):
                    return torch.cat([_old(x[:,:boundary].contiguous(),*args,**kwargs),
                                      _old(x[:,boundary:].contiguous(),*args,**kwargs)],1)
                changes.append((module,old));module.forward=split_dism
            module=block.gdn if mode in ('gdn','both') else None
            if module is not None:
                old=module.forward
                def split_gdn(x,*args,_old=old,**kwargs):
                    left=_old(x[:,:boundary].contiguous(),*args,**kwargs)
                    right=_old(x[:,boundary:].contiguous(),*args,**kwargs)
                    return (torch.cat([left[0],right[0]],1),*right[1:])
                changes.append((module,old));module.forward=split_gdn
        yield
    finally:
        autograd.voc_dism=original_core
        for module,old in changes:module.forward=old


@contextmanager
def pointwise_partition(model,boundary,length):
    """All arms use identical tokenwise GEMM/norm shapes, including intact."""
    changed=[]
    try:
        if boundary:
            for module in model.modules():
                if isinstance(module,(torch.nn.Linear,torch.nn.LayerNorm)):
                    old=module.forward
                    def forward(x,*args,_old=old,**kwargs):
                        if x.ndim==3 and x.shape[1]==length:
                            return torch.cat([_old(x[:,:boundary].contiguous(),*args,**kwargs),
                                              _old(x[:,boundary:].contiguous(),*args,**kwargs)],1)
                        return _old(x,*args,**kwargs)
                    changed.append((module,old));module.forward=forward
        yield
    finally:
        for module,old in changed:module.forward=old


@torch.inference_mode()
def main():
    from dataclasses import replace
    from dism_v2.eval_lm_positions import load_checkpoint
    from dism_v2.train_lm import load_tokenizer
    from dism_v2.lm_data import split_files
    from dism_v2 import embedding
    import pyarrow.parquet as pq
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    test_hard_mass()
    root=Path('/media/cicuvc/c63abdf1-0e56-4153-9228-95df5a2f239b/cicuvc/dism-lm-runs/gdn-dism-vocab-silu-384-72m-30k-20260911')
    out=root/'document-boundary-32-canonical';out.mkdir(exist_ok=False)
    model,meta=load_checkpoint(root/'latest.pt')
    model.config=replace(model.config,context=4096)
    tok=load_tokenizer(meta['config']['tokenizer'])
    _,files=split_files(meta['config']['data'])
    docs=[];seen=set()
    for file in files:
        for batch in pq.ParquetFile(file).iter_batches(batch_size=64,columns=['text']):
            ids=tok(batch.column(0).to_pylist(),add_special_tokens=False,return_attention_mask=False)['input_ids']
            for tokens in ids:
                digest=hashlib.sha256(np.array(tokens,dtype=np.int32).tobytes()).hexdigest()
                if len(tokens)>=1537 and digest not in seen:
                    docs.append(tokens);seen.add(digest)
            if len(docs)>=96:break
        if len(docs)>=96:break
    assert len(docs)>=96
    docs=docs[:96]
    (out/'data.json').write_text(json.dumps(dict(document_token_hashes=[hashlib.sha256(np.array(d,dtype=np.int32).tobytes()).hexdigest() for d in docs],
        rule='First96 distinct held-out documents with >=1537 tokens; B first32, A next64; B target first512 next tokens.')))
    modes=['intact','dism_core','dism','gdn','both']
    losses=torch.empty(32,2,3,5,512);alone=torch.empty(32,512)
    masses=[]
    layers=[b for b in model.blocks if b.dism is not None]
    def evaluate(tokens,start):
        x=torch.tensor([tokens],device='cuda')
        gen=torch.Generator(device='cuda').manual_seed(779)
        with pointwise_partition(model,start,len(tokens)),torch.autocast('cuda',dtype=torch.bfloat16):
            h=model.forward_features(x,1.,gen)[:,start:]
        target=torch.tensor(docs[index][1:513],device='cuda')
        parts=[]
        for t in range(0,512,256):
            with torch.autocast('cuda',dtype=torch.bfloat16):z=model.lm_head(h[:,t:t+256]).float().reshape(-1,50257)
            z=30*torch.tanh(z/30)
            parts.append(torch.nn.functional.cross_entropy(z,target[t:t+256],reduction='none').cpu())
        value=torch.cat(parts)
        assert torch.isfinite(value).all()
        return value
    for index in range(32):
        target=docs[index][:512]
        alone[index]=evaluate(target,0)
        for variant in range(2):
            prefix=docs[32+2*index+variant]
            for li,length in enumerate((128,512,1536)):
                # EOS at the end of A; B starts at a fresh independent document.
                tokens=prefix[:length-1]+[tok.eos_token_id]+target
                for mi,mode in enumerate(modes):
                    capture=index<8 and variant==0 and length==1536 and mode=='intact'
                    old=embedding.forward;labels=[]
                    def traced(*args,**kwargs):
                        raw=old(*args,**kwargs)
                        labels.append((raw[7][0].cpu().numpy(),raw[6][0].cpu().numpy()))
                        return raw
                    try:
                        if capture:embedding.forward=traced
                        with intervention(model,length,mode):losses[index,variant,li,mi]=evaluate(tokens,length)
                    finally:embedding.forward=old
                    if capture:
                        assert len(labels)==3
                        for layer,(q,k) in enumerate(labels):
                            tau=torch.nn.functional.softplus(layers[layer].dism.log_sel_tau.float()).cpu().numpy()
                            masses.append(dict(document=index,layer=4*(layer+1),values=hard_mass(q,k,tau,length)))
        torch.save(dict(losses=losses[:index+1],alone=alone[:index+1],modes=modes,lengths=[128,512,1536],masses=masses),out/'results.pt')
        print(json.dumps(dict(documents=index+1,mean_nll=float(losses[:index+1,:,:,0].mean()),
                             both_minus_alone=float((losses[:index+1,:,:,4]-alone[:index+1,None,None]).abs().mean()))),flush=True)
    print(out,flush=True)


if __name__=='__main__':main()
