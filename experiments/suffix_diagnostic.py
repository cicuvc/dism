"""Small paired label-shuffle/suffix diagnostic, not a causal retrieval ablation."""
import os
os.environ.setdefault('DISM_TILE_LSE','tanh_finite')
os.environ.setdefault('DISM_BWD_OPT','13')
import json
import hashlib
from pathlib import Path
import numpy as np
import torch
from numba import njit,prange,set_num_threads


@njit(parallel=True)
def suffix_stats(q,k):
    m,n=q.shape
    maxima=np.zeros((m,n),np.int16)
    distant=np.zeros((m,n),np.int16)
    hist=np.zeros((m,17),np.int64)
    farhist=np.zeros((m,17),np.int64)
    for a in prange(m):
        dp=np.zeros(n,np.int32)
        for i in range(n):
            for j in range(i,-1,-1):
                length=(dp[j-1]+1 if j>0 else 1) if q[a,i]==k[a,j] else 0
                dp[j]=length
                hist[a,min(length,16)]+=1
                maxima[a,i]=max(maxima[a,i],length)
                if i-j>=128:
                    farhist[a,min(length,16)]+=1
                    distant[a,i]=max(distant[a,i],length)
    return maxima,distant,hist,farhist


def summarize(result,b,l,h,n):
    maxima,distant,hist,farhist=result
    mm=maxima.reshape(b,l,h,n); dd=distant.reshape(b,l,h,n)
    return dict(query_fraction={str(t):float((mm[...,128:]>=t).mean()) for t in (1,2,4,8,16)},
        distant_query_fraction={str(t):float((dd[...,128:]>=t).mean()) for t in (1,2,4,8,16)},
        pair_fraction={str(t):float(hist[:,t:].sum()/hist.sum()) for t in (1,2,4,8,16)},
        distant_pair_fraction={str(t):float(farhist[:,t:].sum()/farhist.sum()) for t in (1,2,4,8,16)},
        per_layer_distant_ge4=(dd[...,128:]>=4).mean((0,3)).tolist())


def selftest():
    q=np.array([[0,1,0,1,0],[1,1,1,1,1]],dtype=np.int16)
    k=q.copy(); mx,_,hist,_=suffix_stats(q,k)
    for a in range(2):
        expected=[]
        for i in range(5):
            lens=[]
            for j in range(i+1):
                z=0
                while z<=min(i,j) and q[a,i-z]==k[a,j-z]: z+=1
                lens.append(z)
            expected.append(max(lens))
        assert np.array_equal(mx[a],expected)
    assert hist.sum()==30


@torch.inference_mode()
def main():
    torch.set_num_threads(2);set_num_threads(4);selftest()
    from dism_v2.eval_lm_positions import load_checkpoint,evaluate_context
    from dism_v2.eval_vocab_load import load_metrics
    from dism_v2 import embedding
    root=Path('/media/cicuvc/c63abdf1-0e56-4153-9228-95df5a2f239b/cicuvc/dism-lm-runs')
    study=root/'activation-study-3k-20260910'
    out=study/'suffix-diagnostic-64-20260911';out.mkdir(exist_ok=False)
    bundle=torch.load(root/'swa-eval-bundle-20260910.pt',map_location='cpu',weights_only=False)
    x,y=bundle['x'][:64,:2048].long(),bundle['y'][:64,:2048].long()
    reports={}
    for arm in ('baseline','width384_72m','gdn_dism_384_matched'):
        model,meta=load_checkpoint(study/arm/'latest.pt')
        assert meta['step']==3000
        layer_ids=[i+1 for i,bk in enumerate(model.blocks) if bk.dism is not None]
        labels=[];losses=[];captured=[]
        original=embedding.forward
        def traced(*args,**kwargs):
            result=original(*args,**kwargs)
            captured.append(torch.stack((result[7],result[6]),1).cpu().short())
            return result
        try:
            embedding.forward=traced
            for i in range(64):
                captured.clear()
                nll=evaluate_context(model,x[i:i+1].cuda(),y[i:i+1].cuda(),1.,779)
                assert torch.isfinite(nll).all() and len(captured)==len(layer_ids)
                labels.append(torch.stack(captured,2)) # [1,side,layer,head,N]
                losses.append(nll)
                if i==0:
                    embedding.forward=original
                    control=evaluate_context(model,x[:1].cuda(),y[:1].cuda(),1.,779)
                    torch.testing.assert_close(nll,control,atol=0,rtol=0)
                    embedding.forward=traced
                if (i+1)%16==0:print(arm,'captured',i+1,flush=True)
        finally:embedding.forward=original
        lab=torch.cat(labels).numpy();nll=torch.cat(losses).numpy()
        b,_,l,h,n=lab.shape
        q=np.ascontiguousarray(lab[:,0].reshape(-1,n));k=np.ascontiguousarray(lab[:,1].reshape(-1,n))
        del model;torch.cuda.empty_cache()
        result=suffix_stats(q,k)
        report=dict(nll=float(nll.mean()),layer_ids=layer_ids,real=summarize(result,b,l,h,n),shuffle=[])
        # Preserve each sequence/head K marginal exactly. Same position permutation
        # across layers/heads of a sequence; Q untouched. Diagnostic only, not inference.
        for seed in (910,911,912):
            rng=np.random.default_rng(seed);shuffled=lab[:,1].copy()
            for bi in range(b):shuffled[bi]=np.take(shuffled[bi],rng.permutation(n),axis=-1)
            null=suffix_stats(q,np.ascontiguousarray(shuffled.reshape(-1,n)))
            report['shuffle'].append(dict(seed=seed,**summarize(null,b,l,h,n)))
            print(arm,'shuffle',seed,flush=True)
        far=result[1].reshape(b,l,h,n)
        score=(far>=4).mean((1,2))
        # Remove each sequence x256-position-block mean; still observational.
        sc=score.reshape(b,8,256); ll=nll.reshape(b,8,256)
        centered_s=(sc-sc.mean(-1,keepdims=True)).reshape(-1)
        centered_l=(ll-ll.mean(-1,keepdims=True)).reshape(-1)
        report['nll_correlation_far_ge4_head_fraction']=float(np.corrcoef(score[:,128:].ravel(),nll[:,128:].ravel())[0,1])
        report['within_sequence_positionblock_correlation']=float(np.corrcoef(centered_s,centered_l)[0,1])
        report['vocab']=[dict(layer=layer_ids[li],head=hi+1,**{side:load_metrics(torch.from_numpy(np.bincount(lab[:,si,li,hi].ravel(),minlength=512))) for si,side in enumerate(('q','k'))}) for li in range(l) for hi in range(h)]
        np.savez_compressed(out/(arm+'.npz'),labels=lab,nll=nll,max_suffix=result[0].reshape(b,l,h,n),max_distant_suffix=far)
        reports[arm]=report
        (out/'report.json').write_text(json.dumps(dict(sequences=64,context=2048,token_sha256=hashlib.sha256(x.numpy().tobytes()+y.numpy().tobytes()).hexdigest(),models=reports,
            caveats=['Shuffle preserves K marginals but removes local/context/positional structure; not a causal intervention on model outputs.',
                     'Query fractions exclude first128 positions; distant means i-j>=128.',
                     'NLL association is observational; not an estimate of retrieval benefit.',
                     'Heads/layers are not independent samples; three shuffles do not measure training-seed uncertainty.',
                     'These64 frozen sequences differ from the4096-sequence packed-prefix study.']),indent=2)+'\n')
        print(arm,report['nll'],report['real'],flush=True)


if __name__=='__main__':main()
