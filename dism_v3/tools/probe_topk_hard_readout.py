"""Read-only checkpoint screen: native v3 soft, top-k current-Q hard reads.

Native mainline labels are untouched. Candidate states never persist. Default
uses native + (FP32 candidate mixture - FP32 top1) to anchor k=1 exactly;
direct reconstruction is a separate numerical control. No training or kernels.
"""
import argparse
from contextlib import contextmanager
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import time

import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer
from nanochat.models import ModelSpec,build_model
from nanochat.niah_eval import _score
from topk_hard_reference import read_candidates


@contextmanager
def candidate_reads(model,k,records,*,direct=False,strength=1.):
    import flash_dism.compiler as compiler
    original=compiler.voc_forward
    layers=[i for i,l in enumerate(model.layers) if not getattr(l.attn,'is_pure_mixer',False)]
    active=[];handles=[]
    for i in layers:
        handles.append(model.layers[i].attn.register_forward_pre_hook(lambda m,a,i=i:active.__setitem__(slice(None),[i])))
    def wrapped(q,key,sq,sk,v,eq,ek,tau,beta,direction,hard,boundaries,layout=None,*,direction_true=False):
        if torch.is_grad_enabled():
            raise RuntimeError('top-k probe is inference-only; no training backward is installed')
        assert direction_true and bool(hard.all())
        result=list(original(q,key,sq,sk,v,eq,ek,tau,beta,direction,hard,boundaries,layout,direction_true=True))
        b,n,h,_=q.shape
        assert b==1,'diagnostic uses physical batch 1'
        def flat(x):return x.float().permute(0,2,1,3).reshape(b*h,n,x.shape[-1]).contiguous()
        with torch.autocast('cuda',enabled=False):
            logits=torch.einsum('bnhd,hcd->bnhc',q.float(),eq.to(torch.bfloat16).float())*float(beta)
            logits=flat(logits)
            iq,ik=result[6].reshape(b*h,n),result[5].reshape(b*h,n)
            if boundaries is None:seg=torch.zeros((b*h,n),device=q.device,dtype=torch.int32)
            else:
                assert boundaries[0]==0 and boundaries[-1]==n
                seg=torch.bucketize(torch.arange(n,device=q.device),boundaries[1:-1],right=True)[None].expand(b*h,-1)
            mixed,base,mass=read_candidates(logits,iq,ik,flat(sq),flat(sk),flat(v),tau.float(),seg,k)
            native=flat(result[0])
            delta=mixed-base
            replacement=mixed if direct else native+strength*delta
            result[0]=replacement.reshape(b,h,n,v.shape[-1]).permute(0,2,1,3).contiguous().to(v.dtype)
            assert bool(torch.isfinite(result[0]).all())
            records.append(dict(layer=active[0],mass_mean=float(mass.mean()),
                mass_q10=float(mass.quantile(.1)),mass_median=float(mass.median()),
                native_label_disagreement=float((iq!=logits.argmax(-1)).float().mean()),
                top1_native_relative_l2=float((base-native).norm()/native.norm().clamp_min(1e-30)),
                change_relative_l2=float(delta.norm()/native.norm().clamp_min(1e-30))))
        return tuple(result)
    compiler.voc_forward=wrapped
    try:yield
    finally:
        compiler.voc_forward=original
        for hook in handles:hook.remove()


def atomic_json(path,data):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(data,indent=2)+'\n');temporary.replace(path)


def load_model(checkpoint):
    meta=json.loads(checkpoint.with_name(checkpoint.name.replace('model_','meta_')).with_suffix('.json').read_text())
    with torch.device('meta'):model=build_model(ModelSpec.from_dict(meta['model_spec']))
    model.to_empty(device='cuda');model.init_weights()
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model.load_state_dict({k.removeprefix('_orig_mod.'):v for k,v in state.items()},assign=True)
    del state
    model.cuda().eval();model.eval_hard=True
    model.mxfp8_ce=None;model.mxfp8_ce_norm=None
    from flash_mxfp8 import refresh_fp8_modules
    refresh_fp8_modules(model)
    return model,meta


def niah_suite(tok,hay,contexts,depths,trials,seed):
    """Full phrase insertion, matched-length absence, BOS and common target."""
    cells=[]
    bos=tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    query=tok.encode('\nThe special magic number is',add_special_tokens=False)
    for n in contexts:
        for depth in depths:
            rng=random.Random(f'{seed}:{n}:{depth}')
            for trial in range(trials):
                number=rng.randint(10000,99999)
                needle=tok.encode(f'\nThe special magic number is {number}.',add_special_tokens=False)
                assert needle[:len(query)]==query
                target=needle[len(query):][:6]
                start=n-6
                available=start-1-len(query)-len(needle)
                offset=rng.randrange(len(hay)-available-len(needle))
                background=hay[offset:offset+available]
                position=int(depth*available)
                case=dict(context=n,depth=depth,trial=trial,number=number,target_start=start,target=target)
                for condition in ('present','absent'):
                    insert=needle if condition=='present' else hay[offset+available:offset+available+len(needle)]
                    head=[bos]+background[:position]+insert+background[position:]+query
                    assert len(head)==start
                    case[condition]=head+target+[tok.eos_token_id]*(n-start-len(target))
                cells.append(case)
    return cells


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm',choices=['baseline','ln_scheduled'],default='baseline')
    p.add_argument('--output',required=True)
    p.add_argument('--documents',type=int,default=4)
    p.add_argument('--contexts',type=int,nargs='+',default=[2048,8192])
    p.add_argument('--niah-contexts',type=int,nargs='*',default=[8192])
    p.add_argument('--trials',type=int,default=2)
    p.add_argument('--modes',nargs='+',default=['hard','soft','k2','k4','k8'])
    p.add_argument('--strength',type=float,default=1.)
    p.add_argument('--direct',action='store_true')
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    repo=Path(__file__).resolve().parents[2]
    for name in ('flash_dism','cu_flash_dism'):
        assert Path(importlib.util.find_spec(name).origin).resolve().is_relative_to(repo/'dism_v3')
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');torch.manual_seed(42)
    checkpoints=json.loads((repo/'dism_v3/build/soft_hard_heads/checkpoints.json').read_text())
    checkpoint=Path(checkpoints[a.arm]);model,meta=load_model(checkpoint)
    counters={name:v.detach().cpu().clone() for name,v in model.named_buffers()}
    tok=AutoTokenizer.from_pretrained('/home/cicuvc/downstream-eval/tokenizer',local_files_only=True)
    bos=tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    parquet=pq.ParquetFile('/home/cicuvc/downstream-eval/data/pg19/test.parquet')
    texts=[]
    try:
        for batch in parquet.iter_batches(batch_size=1,columns=['text']):
            texts.append(tok.encode(batch.column('text')[0].as_py(),add_special_tokens=False))
            if len(texts)>=max(4,a.documents):break
    finally:parquet.close()
    suite=niah_suite(tok,sum(texts,[]),a.niah_contexts,[.05,.95],a.trials,20261025)
    atomic_json(out/'inputs.json',dict(niah=suite,pg19=[t[:max(a.contexts)] for t in texts[:a.documents]],bos=bos))
    data=dict(args=vars(a),checkpoint=str(checkpoint),model_spec=meta['model_spec'],
              source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              inputs_sha256=hashlib.sha256((out/'inputs.json').read_bytes()).hexdigest(),pg19=[],niah=[])
    def clear():
        for m in model.modules():
            if type(m).__module__.startswith('flash_mxfp8.'):
                for key in ('_forward_states','_forward_plans'):
                    cache=getattr(m,key,None)
                    if isinstance(cache,dict):cache.clear()
        gc.collect()
    def run(mode,fn):
        stats=[];model.eval_hard=mode!='soft'
        torch.cuda.synchronize();start=time.perf_counter()
        if mode.startswith('k') or mode=='sham':
            k=1 if mode=='sham' else int(mode[1:])
            with candidate_reads(model,k,stats,direct=a.direct or mode=='sham',strength=a.strength):value=fn()
        else:value=fn()
        torch.cuda.synchronize()
        return value,stats,time.perf_counter()-start
    for n in a.contexts:
        clear();model.config.sequence_len=n
        for doc,tokens in enumerate(texts[:a.documents]):
            assert len(tokens)>=n
            x=torch.tensor([[bos]+tokens[:n-1]],device='cuda')
            y=torch.tensor([tokens[:n]],device='cuda');cu=torch.tensor([0,n],device='cuda',dtype=torch.int32)
            def score():return model(x,y,cu_seqlens=cu,loss_reduction='none').double().cpu()
            reference=None
            modes=a.modes+((['sham'] if a.direct else ['sham','k1']) if doc==0 and 'hard' in a.modes else [])
            for mode in modes:
                value,stats,seconds=run(mode,score)
                assert bool(torch.isfinite(value).all())
                if mode=='hard':reference=value
                if mode=='k1' and not a.direct:assert torch.equal(value,reference),'k1 must exactly reproduce native'
                record=dict(context=n,document=doc,mode=mode,nll=float(value.mean()),seconds=seconds,heads=stats,
                    position_bins=[float(v.mean()) for v in value.flatten().split(1024)])
                if reference is not None:record.update(delta_nll=float((value-reference).mean()),max_token_delta=float((value-reference).abs().max()))
                data['pg19'].append(record);atomic_json(out/'results.json',data)
                print(json.dumps({k:v for k,v in record.items() if k not in ('heads','position_bins')}),flush=True)
    for case in suite:
        n=case['context'];clear();model.config.sequence_len=n
        for mode in a.modes:
            record={k:case[k] for k in ('context','depth','trial','number')};record['mode']=mode
            for condition in ('present','absent'):
                value,stats,seconds=run(mode,lambda:_score(model,case[condition],case['target_start'],case['target'],torch.device('cuda')))
                loss,hits,total,correct=value
                record[condition]=dict(nll=loss,hits=hits,total=total,all_correct=correct,seconds=seconds,heads=stats)
            record['gain']=record['absent']['nll']-record['present']['nll']
            data['niah'].append(record);atomic_json(out/'results.json',data)
            print('niah',case['depth'],case['trial'],mode,record['present']['nll'],record['gain'],flush=True)
    for name,value in model.named_buffers():assert torch.equal(value.cpu(),counters[name]),name
    data['buffers_unchanged']=True;atomic_json(out/'results.json',data)
    (out/'complete').write_text('ok\n')


if __name__=='__main__':main()
