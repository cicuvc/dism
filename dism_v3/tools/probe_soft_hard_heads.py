"""Checkpoint-only per-head DISM pre-combine output and loss-gradient comparison."""
import argparse
import json
import math
from pathlib import Path
import torch
import pyarrow.parquet as pq
from nanochat.models import ModelSpec, build_model
from transformers import AutoTokenizer


def measures(a,b):
    # [tokens, heads, channels], FP64 reductions on CPU; keep positional bins.
    a=a.double();b=b.double();dims=(0,2)
    aa=(a*a).sum(dims);bb=(b*b).sum(dims);ab=(a*b).sum(dims)
    dd=((b-a)**2).sum(dims);count=a.shape[0]*a.shape[2]
    return dict(soft_rms=(aa/count).sqrt().tolist(),hard_rms=(bb/count).sqrt().tolist(),
        cosine=(ab/(aa*bb).sqrt().clamp_min(1e-30)).tolist(),
        relative_l2=(dd/aa.clamp_min(1e-30)).sqrt().tolist(),
        hard_soft_norm_ratio=(bb/aa.clamp_min(1e-30)).sqrt().tolist())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--contexts',type=int,nargs='+',default=[2048,8192])
    p.add_argument('--documents',type=int,default=4)
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.30)
    checkpoint=Path(args.checkpoint)
    meta=json.loads(checkpoint.with_name(checkpoint.name.replace('model_','meta_')).with_suffix('.json').read_text())
    with torch.device('meta'):model=build_model(ModelSpec.from_dict(meta['model_spec']))
    model.to_empty(device='cuda');model.init_weights()
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model.load_state_dict({k.removeprefix('_orig_mod.'):v for k,v in state.items()},assign=True)
    del state
    model.cuda().eval()
    from flash_mxfp8 import refresh_fp8_modules
    refresh_fp8_modules(model)
    # Evaluation mode controls only path selection, gradients remain enabled.
    model.mxfp8_ce=None;model.mxfp8_ce_norm=None
    tokenizer=AutoTokenizer.from_pretrained('/home/cicuvc/downstream-eval/tokenizer',local_files_only=True)
    bos=tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    rows=pq.read_table('/home/cicuvc/downstream-eval/data/pg19/test.parquet').to_pylist()
    tokens=[tokenizer.encode(r['text'],add_special_tokens=False) for r in rows[:args.documents]]
    captured={}
    snapshots={}
    for i,layer in enumerate(model.layers):
        if getattr(layer.attn,'is_pure_mixer',False):continue
        original=layer.attn._combine_cuda
        def combine(value,*a,index=i,fn=original,**kw):
            value.retain_grad();captured[index]=value
            snapshots[index]=value.detach().float().cpu().clone()
            return fn(value,*a,**kw)
        layer.attn._combine_cuda=combine
    results=[]
    counter=model.rng_counter.clone()
    for length in args.contexts:
        model.config.sequence_len=length
        for doc,t in enumerate(tokens):
            assert len(t)>=length
            ids=torch.tensor([[bos]+t[:length-1]],device='cuda')
            target=torch.tensor([t[:length]],device='cuda')
            cu=torch.tensor([0,length],device='cuda',dtype=torch.int32)
            paths={}
            for hard in [False,True]:
                model.zero_grad(set_to_none=True);captured.clear();model.eval_hard=hard
                loss=model(ids,target,cu_seqlens=cu)
                loss.backward()
                for i,v in captured.items():
                    torch.testing.assert_close(v.detach().float().cpu(),snapshots[i],rtol=0,atol=0)
                paths[hard]=(float(loss.detach()),{i:(v.detach().float().cpu().squeeze(0),v.grad.detach().float().cpu().squeeze(0)*length) for i,v in captured.items()})
                assert math.isfinite(float(loss.detach()))
                assert torch.equal(counter,model.rng_counter)
                print(length,doc,'hard' if hard else 'soft',float(loss.detach()),'peakGB',torch.cuda.max_memory_allocated()/1e9,flush=True)
                del loss
            for i in paths[False][1]:
                a,ga=paths[False][1][i];b,gb=paths[True][1][i]
                assert all(torch.isfinite(v).all() for v in [a,b,ga,gb])
                results.append(dict(context=length,document=doc,layer=i,soft_loss=paths[False][0],hard_loss=paths[True][0],
                    output=measures(a,b),gradient=measures(ga,gb),
                    position_bins=[dict(start=start,end=min(start+1024,length),output=measures(a[start:start+1024],b[start:start+1024]),gradient=measures(ga[start:start+1024],gb[start:start+1024])) for start in range(0,length,1024)]))
            (out/'results.json').write_text(json.dumps(dict(checkpoint=str(checkpoint),model_spec=meta['model_spec'],
                protocol='Same BOS-prefixed PG19 prefixes; all-soft versus all-hard end-to-end; raw DISM output before GDN addition, norm and gate; BF16 CE; gradients multiplied by token count to remove mean-loss length scaling; no optimizer updates; eager quadratic hard backward',records=results),indent=2)+'\n')
            captured.clear();del paths
    (out/'complete').write_text('ok\n')


if __name__=='__main__':main()
