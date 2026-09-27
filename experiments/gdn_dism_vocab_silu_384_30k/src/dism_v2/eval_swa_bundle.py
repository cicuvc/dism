"""A100-only forward evaluation of SWA checkpoint on verified paired input bundle."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
import torch
from torch.nn import functional as F


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('New output directory required')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    bundle=torch.load(a.bundle,map_location='cpu',weights_only=True)
    cfg=saved['config']
    from .lm_model import LMConfig,DecoderLM
    c=LMConfig(**cfg['model'])
    assert c.architecture=='swa_only' and saved['step']==30000
    for key in ('steps','batch','micro_batch','seed','lr','weight_decay','warmup','data','tokenizer'):
        assert cfg[key]==bundle['training_config'][key],key
    for key in ('width','layers','heads','head_dim','context','window','rope_theta','softcap','vocab_size'):
        assert cfg['model'][key]==bundle['training_config']['model'][key],key
    model=DecoderLM(replace(c,context=8192)).cuda().eval()
    model.load_state_dict(saved['model'],strict=True)
    parameter_count=sum(p.numel() for p in model.parameters())
    assert parameter_count==49641856
    del saved
    a.output.mkdir(parents=True)
    def cap(logits):return logits if c.softcap is None else c.softcap*torch.tanh(logits/c.softcap)
    def features(x):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            return model.forward_features(x,1.,None)
    def nll(x,y):
        h=features(x).reshape(-1,c.width)
        losses=[]
        for offset in range(0,len(h),256):
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits=model.lm_head(h[offset:offset+256]).float()
            losses.append(F.cross_entropy(cap(logits),y.reshape(-1)[offset:offset+256],reduction='none'))
        out=torch.cat(losses).reshape_as(y).cpu()
        assert torch.isfinite(out).all()
        return out
    start=time.perf_counter()
    loss={n:[] for n in bundle['lengths']}
    for batch in range(0,len(bundle['x']),4):
        x=bundle['x'][batch:batch+4].long().cuda();y=bundle['y'][batch:batch+4].long().cuda()
        for n in loss:
            loss[n].append(torch.cat([nll(x[:,i:i+n].contiguous(),y[:,i:i+n].contiguous()) for i in range(0,8192,n)],1))
        if (batch+4)%32==0:print(json.dumps(dict(phase='positions',sequences=batch+4,seconds=time.perf_counter()-start)),flush=True)
    loss={n:torch.cat(v) for n,v in loss.items()}
    torch.save(loss,a.output/'position_losses.pt')
    candidate_ids=torch.tensor(bundle['candidate_ids'],device='cuda')
    def answer(ids,target):
        h=features(ids[None].long().cuda())
        with torch.autocast('cuda',dtype=torch.bfloat16):logits=model.lm_head(h[:,-1]).float()[0]
        logits=cap(logits)
        assert torch.isfinite(logits).all()
        lp=logits.log_softmax(-1);scores=logits[candidate_ids];pred=logits.argmax().item()
        return dict(greedy_token_id=pred,exact_match=pred==candidate_ids[target].item(),
                    candidate_correct=scores.argmax().item()==target,candidate_top1=bundle['colors'][scores.argmax().item()],
                    target_nll=-lp[candidate_ids[target]].item(),candidate_logits=scores.tolist())
    rows=[]
    for i,case in enumerate(bundle['cases']):
        r={k:v for k,v in case.items() if k not in ('ids','absent','counterfactual')}
        r.update(answer(case['ids'],case['target']))
        if case['condition']=='needle':
            r['absent']=answer(case['absent'],case['target'])
            r['counterfactual']=answer(case['counterfactual'],case['alternative'])
        rows.append(r)
        if (i+1)%64==0:print(json.dumps(dict(phase='niah',cases=i+1,seconds=time.perf_counter()-start)),flush=True)
    report=dict(checkpoint=str(a.checkpoint),step=30000,parameters=parameter_count,config=cfg,
        gpu=torch.cuda.get_device_name(),torch=torch.__version__,seconds=time.perf_counter()-start,
        paired_xy_sha256=bundle['paired_xy_sha256'],candidate_colors=bundle['colors'],niah=rows,
        mean_nll={n:v.double().mean().item() for n,v in loss.items()},
        receptive_field_predecessors=c.layers*(c.window-1),all_losses_logits_finite=True)
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(finished=True,mean_nll=report['mean_nll'],seconds=report['seconds'])),flush=True)


if __name__=='__main__':main()
