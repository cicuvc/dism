"""Isolated short training screen for input scale and sigmoid low-rank gates.

Local data only, fresh models, no production defaults/checkpoints changed.
All arms share cached packed batches and optimizer/hard-sampling schedules.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'dism_v3/python'), str(ROOT/'nanochat')]
os.environ['DISM_HARD_PLANNER'] = '0'
os.environ['DISM_HARD_BACKWARD'] = 'quadratic'
os.environ.setdefault('OMP_NUM_THREADS', '1')
import torch
from torch.nn import functional as F
from nanochat.models import ModelSpec, build_model, PretrainOptimizerConfig
from nanochat.tokenizer import HuggingFaceTokenizer
from nanochat.dataloader import tokenizing_distributed_data_loader_with_state_bos_bestfit
from nanochat.cuda_graph import PackedTrainingGraph
from nanochat.document_loss import DocumentLossAccumulator


def rms(x):
    return x.detach().float().square().mean().sqrt().item()


def data_cache(out, steps):
    path = out/'batches.pt'
    if path.exists():
        value = torch.load(path, weights_only=False)
        assert len(value['train']) >= steps
        return value
    storage = Path('/media/cicuvc/c63abdf1-0e56-4153-9228-95df5a2f239b/cicuvc')
    tok = HuggingFaceTokenizer.from_pretrained(str(storage/'dism-runs/dism-fwedu-valres-72m-2500m/tokenizer'))
    result = {}
    for split, count in [('train',steps),('val',8)]:
        loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
            tok,8,2048,split=split,device='cpu',data_dir=str(storage/'slimpajama6b/data'),
            seq_align=256,aligned_segment_ends=True,shuffle=split=='train',shuffle_seed=1337)
        result[split] = []
        for i in range(count):
            x,y,cu,_,_ = next(loader)
            result[split].append(tuple(t.clone() for t in (x,y,cu)))
        print('cached',split,count,flush=True)
    torch.save(result,path)
    return result


def build(spec, arm, perturb, batch):
    torch.manual_seed(42)
    with torch.device('meta'):
        model = build_model(spec)
    model.to_empty(device='cuda'); model.init_weights()
    # Activation scaling preserves the embedding parameterization and optimizer.
    if arm in ('input_scale','combined'):
        model.embedding.register_forward_hook(lambda m,args,out: (out.float()*50.).to(out.dtype))
    sigmoid = arm in ('sigmoid','combined')
    inputs = {}
    handles = []
    for i,l in enumerate(model.layers):
        def save(m,args,i=i):
            inputs[i] = args[0].detach()
        handles.append(l.attn.g_proj_down.register_forward_pre_hook(save))
    model.eval(); model.eval_hard=False
    with torch.no_grad():
        model(batch[0],batch[1],cu_seqlens=batch[2])
    for h in handles:h.remove()
    scales=[]
    for i,l in enumerate(model.layers):
        attn=l.attn
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            # Match on unit-RMS input, avoiding preservation of the first-layer
            # gate suppression that this candidate is intended to remove.
            x=F.rms_norm(inputs[i].float(),(inputs[i].shape[-1],),eps=1e-6).to(torch.bfloat16)
            z=attn.g_proj_down(x)
            target=rms(F.silu(attn.g_proj_up(z)))
            raw=rms(attn.g_proj_up(z.sigmoid()))
            scale=target/raw
            scales.append(scale)
        if sigmoid:
            attn.g_proj_up.register_forward_pre_hook(lambda m,args: (args[0].sigmoid(),))
        # Same unfused expression for every arm: avoid fused-vs-unfused norm
        # being a confounder in the candidate gate comparison.
        def norm(u,g,m=attn.norm,s=scale,use_sigmoid=sigmoid):
            v=u.float()*torch.rsqrt(u.float().square().mean(-1,keepdim=True)+m.eps)
            gate=g.float()*s if use_sigmoid else F.silu(g.float())
            return (v*m.weight.float()*gate).to(u.dtype)
        attn.norm.forward=norm
    del inputs
    actual=0.
    if perturb:
        with torch.no_grad():
            p=model.embedding.weight
            before=p.clone()
            gen=torch.Generator(device='cuda').manual_seed(991)
            p.copy_((p.float()+torch.randn(p.shape,device='cuda',generator=gen)*2e-5).to(p.dtype))
            actual=rms(p.float()-before.float())/rms(before)
    model.train()
    return model,dict(gate_scales=scales,embedding_relative_perturbation=actual)


@torch.no_grad()
def evaluate(model,batches):
    model.eval(); total=0.;count=0.
    metrics={}
    for hard in [False,True]:
        model.eval_hard=hard;total=0.;count=0.
        for x,y,cu in batches:
            s,n=model(x,y,cu_seqlens=cu,loss_reduction='document_sum')
            total+=s.item();count+=n.item()
        metrics['hard' if hard else 'soft']=total/count
    model.train()
    return metrics


def run(spec,arm,perturb,cache,args):
    key=f'{arm}_'+('perturbed' if perturb else 'reference')
    path=args.output/f'{key}.json'
    if path.exists() and json.loads(path.read_text()).get('complete'):
        return
    first=tuple(t.cuda() for t in cache['train'][0])
    validation=[tuple(t.cuda() for t in b) for b in cache['val']]
    model,details=build(spec,arm,perturb,first)
    optimizer=model.setup_pretraining_optimizer(PretrainOptimizerConfig(8e-4,8e-4,8e-4,8e-4,.01))
    result=dict(arm=arm,perturb=perturb,details=details,steps=[],validation=[],complete=False)
    result['validation'].append(dict(step=0,**evaluate(model,validation)))
    model.set_training_step(0)
    graph=PackedTrainingGraph(model,*first,microbatches=1,planner=False)
    started=time.monotonic()
    try:
        for step in range(args.steps):
            model.set_training_step(step)
            model.zero_grad(set_to_none=True)
            batch=tuple(t.cuda() for t in cache['train'][step])
            numerator,count=graph.forward_backward(*batch)
            accumulator=DocumentLossAccumulator(1,'cuda');accumulator.add(numerator,count)
            loss=accumulator.finalize(model.parameters())
            bias=model.layers[0].attn.o_proj.bias.grad.float().norm().item()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.).item()
            if not torch.isfinite(torch.tensor(norm)):
                raise FloatingPointError((key,step,norm))
            for group in optimizer.param_groups:
                group['lr']=8e-4*min((step+1)/500.,1.)
            optimizer.step()
            result['steps'].append(dict(step=step,loss=loss.item(),grad_norm=norm,first_bias_grad=bias,bias_squared_share=bias*bias/max(norm*norm,1e-30)))
            if (step+1)%50==0 or step+1==args.steps:
                result['validation'].append(dict(step=step+1,**evaluate(model,validation)))
                result['seconds']=time.monotonic()-started
                path.write_text(json.dumps(result,indent=2)+'\n')
                print(key,step+1,'loss',result['steps'][-1]['loss'],'val',result['validation'][-1],flush=True)
        result['complete']=True
        result['rng_counter']=model.rng_counter.item()
        path.write_text(json.dumps(result,indent=2)+'\n')
    finally:
        graph.close()
    del model,graph,optimizer,first,validation,batch,numerator,count,loss
    gc.collect();torch.cuda.empty_cache()


def collect(output):
    rows=[]
    for arm in ['baseline','input_scale','sigmoid','combined']:
        runs=[json.loads((output/f'{arm}_{suffix}.json').read_text()) for suffix in ['reference','perturbed']]
        assert all(r['complete'] and len(r['steps'])==300 and r['rng_counter']==300 for r in runs)
        rows.append(dict(arm=arm,
            soft_mean=sum(r['validation'][-1]['soft'] for r in runs)/2,
            hard_mean=sum(r['validation'][-1]['hard'] for r in runs)/2,
            soft_pair_gap=abs(runs[0]['validation'][-1]['soft']-runs[1]['validation'][-1]['soft']),
            hard_pair_gap=abs(runs[0]['validation'][-1]['hard']-runs[1]['validation'][-1]['hard']),
            initial_bias_share=sum(r['steps'][0]['bias_squared_share'] for r in runs)/2,
            clip_fraction=sum(s['grad_norm']>1 for r in runs for s in r['steps'])/600))
    (output/'comparison.json').write_text(json.dumps(rows,indent=2)+'\n')
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps',type=int,default=300)
    parser.add_argument('--output',type=Path,default=ROOT/'dism_v3/build/initialization_screen')
    parser.add_argument('--arm',choices=['baseline','input_scale','sigmoid','combined'])
    parser.add_argument('--perturb',action='store_true')
    args=parser.parse_args()
    if not args.output.resolve().is_relative_to(ROOT/'dism_v3/build'):
        parser.error('output must be under dism_v3/build')
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.set_float32_matmul_precision('high')
    ref=Path((ROOT/'dism_v3/build/layout_diagnosis/new_training_run.txt').read_text().strip())
    spec=ModelSpec.from_dict(json.loads((ref/'model_spec.json').read_text()))
    protocol=dict(steps=args.steps,packed_tokens_per_step=16384,seed=42,shuffle_seed=1337,
                  warmup=500,lr=8e-4,clip=1.,cuda_graph=True,planner=False,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  model_spec=spec.to_dict(),note='screen: 1/8 production global batch; same original hard annealing schedule; deliberate embedding perturbation, not independent initialization seeds')
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    if args.arm:
        cache=data_cache(args.output,args.steps)
        run(spec,args.arm,args.perturb,cache,args)
    else:
        # FLA/MXFP8 global caches can retain storage from previous models.
        # Process isolation ensures every arm starts with the same memory state.
        for arm in ['baseline','input_scale','sigmoid','combined']:
            for perturb in [False,True]:
                key=f'{arm}_'+('perturbed' if perturb else 'reference')
                path=args.output/f'{key}.json'
                if path.exists() and json.loads(path.read_text()).get('complete'):
                    continue
                command=[sys.executable,str(Path(__file__).resolve()),'--steps',str(args.steps),'--output',str(args.output),'--arm',arm]
                if perturb:command.append('--perturb')
                subprocess.run(command,check=True)
        if args.steps==300:
            print(json.dumps(collect(args.output),indent=2),flush=True)


if __name__=='__main__':main()
