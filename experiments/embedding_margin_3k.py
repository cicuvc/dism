"""Sample actual interpolation inputs; measure softmax confidence without mutation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import torch

NAMES=('p1','p2','probability_margin','logit_margin','entropy','effective_choices',
       'token_norm','selected_embedding_norm','selected_cosine')


def metrics(logits):
    top=logits.topk(2,dim=-1)
    lse=logits.logsumexp(-1)
    p=logits.softmax(-1)
    p12=(top.values-lse[...,None]).exp()
    entropy=-(p*(logits-lse[...,None])).sum(-1)
    return torch.stack((p12[...,0],p12[...,1],p12[...,0]-p12[...,1],
                        top.values[...,0]-top.values[...,1],entropy,entropy.exp()),-1),top.indices[...,0]


def describe(v):
    v=v.double().reshape(-1,len(NAMES))
    out={name:dict(mean=float(v[:,i].mean()),std=float(v[:,i].std()),
                  p01=float(v[:,i].quantile(.01)),p99=float(v[:,i].quantile(.99)),
                  p10=float(v[:,i].quantile(.1)),median=float(v[:,i].median()),
                  p90=float(v[:,i].quantile(.9))) for i,name in enumerate(NAMES)}
    out.update(p1_gt_half=float((v[:,0]>.5).double().mean()),p1_gt_09=float((v[:,0]>.9).double().mean()),
               margin_lt_001=float((v[:,2]<.01).double().mean()),
               logit_margin_lt_01=float((v[:,3]<.1).double().mean()))
    out['token_norm_p1_correlation']=float(torch.corrcoef(v[:,[6,0]].T)[0,1])
    out['token_norm_logit_margin_correlation']=float(torch.corrcoef(v[:,[6,3]].T)[0,1])
    return out


def norm_stats(x):
    x=x.double().flatten()
    return dict(mean=float(x.mean()),std=float(x.std()),
                quantiles={str(q):float(x.quantile(q)) for q in (.01,.1,.5,.9,.99)})


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    os.environ['DISM_TILE_LSE']='tanh_finite';os.environ['DISM_BWD_OPT']='13'
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    from dism_v2.eval_lm_positions import load_checkpoint
    from dism_v2 import embedding
    bundle=torch.load(a.bundle,map_location='cpu',weights_only=False)
    x=bundle['x'][:64,:2048].long()
    report=dict(sequences=64,context=2048,sample_stride=8,tokens_per_head=16384,hard_prob=1,
                input_sha256=hashlib.sha256(x.numpy().tobytes()).hexdigest(),models={})
    for mode in ('baseline','vocab_silu','no_qk_silu'):
        model,meta=load_checkpoint(a.root/mode/'latest.pt')
        assert meta['step']==3000
        samples=[[[] for _ in range(2)] for _ in range(15)]
        original=embedding.forward;call=0;disagree=0;count=0;maxerr=0.
        def traced(q,k,eq,ek,scale=1.,**kwargs):
            nonlocal call,disagree,count,maxerr
            raw=original(q,k,eq,ek,scale,**kwargs)
            with torch.autocast('cuda',enabled=False):
                for side,(z,e,label,ptop) in enumerate(((q,eq,raw[7],raw[5]),(k,ek,raw[6],raw[4]))):
                    z=z[:,:,::8].permute(1,0,2,3).reshape(4,-1,64).float()
                    logits=(z@e.float().transpose(-1,-2))*scale
                    value,ids=metrics(logits)
                    zn=z.norm(dim=-1)
                    en=e.float().norm(dim=-1).gather(1,ids)
                    cosine=logits.gather(-1,ids[...,None]).squeeze(-1)/(scale*zn*en).clamp_min(1e-20)
                    value=torch.cat((value,torch.stack((zn,en,cosine),-1)),-1)
                    actual=label[:,:,::8].permute(1,0,2).reshape(4,-1)
                    actual_p=ptop[:,:,::8].permute(1,0,2).reshape(4,-1)
                    disagree+=int((ids!=actual).sum());count+=ids.numel()
                    maxerr=max(maxerr,float((value[...,0]-actual_p).abs().max()))
                    samples[call%15][side].append(value.cpu())
            call+=1
            return raw
        try:
            embedding.forward=traced
            for start in range(0,64,2):
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    out=model.forward_features(x[start:start+2].cuda(),1.,torch.Generator(device='cuda').manual_seed(779))
                assert torch.isfinite(out).all()
                if start==0:
                    embedding.forward=original
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        ref=model.forward_features(x[:2].cuda(),1.,torch.Generator(device='cuda').manual_seed(779))
                    torch.testing.assert_close(out,ref,atol=0,rtol=0)
                    embedding.forward=traced
        finally:embedding.forward=original
        assert call==32*15
        values=torch.stack([torch.stack([torch.cat(s,dim=1) for s in layer]) for layer in samples])
        assert values.shape==(15,2,4,16384,9) and torch.isfinite(values).all()
        torch.save(values,a.output/f'{mode}.pt')
        r=dict(argmax_disagreements=disagree,sampled_rows=count,p1_max_abs_error_vs_cuda=maxerr,
               unchanged_forward_bitwise=True,sides={side:describe(values[:,i]) for i,side in enumerate(('q','k'))},
               heads=[dict(layer=l+1,side=side,head=h+1,stats=describe(values[l,i,h]))
                      for l in range(15) for i,side in enumerate(('q','k')) for h in range(4)])
        codebooks={side:dict(raw=[],effective=[]) for side in ('q','k')}
        for block in model.blocks:
            effective=block.dism.expanded_vocabularies(torch.bfloat16)
            for side,raw,eff in zip(('q','k'),(block.dism.q_voc,block.dism.k_voc),effective):
                codebooks[side]['raw'].append(raw.float().norm(dim=-1).cpu())
                codebooks[side]['effective'].append(eff.float().norm(dim=-1).cpu())
        codebooks={side:{kind:torch.stack(v) for kind,v in d.items()} for side,d in codebooks.items()}
        torch.save(codebooks,a.output/f'{mode}_codebook_norms.pt')
        r['codebooks']={side:{kind:norm_stats(v) for kind,v in d.items()} for side,d in codebooks.items()}
        r['metric_names']=NAMES
        report['models'][mode]=r
        print(json.dumps(dict(mode=mode,**{k:v for k,v in r.items() if k!='heads'})),flush=True)
        del model,values,samples
        torch.cuda.empty_cache()
    report['notes']=['Natural-log logits with actual embedding scale1; no extra 1/sqrt(D).',
        'Replay uses actual BF16 inputs/effective codebooks and FP32 GEMM/softmax.',
        'Softmax measured during pure-hard final-model inference, not during training.',
        'This is per-input confidence, distinct from marginal label utilization.',
        'Sampling every8th token may have positional sampling bias; not an all-token census.']
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
