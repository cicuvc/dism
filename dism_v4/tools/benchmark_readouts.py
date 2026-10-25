"""Frozen-weight, paired arithmetic/Jensen timings; no optimizer updates or jobs.

The protected baseline is imported from a recorded Git revision in memory.
Its purpose is to separate removal of the slow fallback from the Jensen change.
"""
import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
import flash_dism_v4.blocked as current
import flash_dism_v4.module as module


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind',choices=['operator','model','mqar'],default='operator')
    p.add_argument('--batch',type=int,default=1)
    p.add_argument('--length',type=int,default=2048)
    p.add_argument('--heads',type=int,default=12)
    p.add_argument('--vocab',type=int,default=512)
    p.add_argument('--value-dim',type=int,default=64)
    p.add_argument('--steps',type=int,default=5)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--baseline-commit',default='50bec9f')
    p.add_argument('--checkpoint',default='dism_v4/build/mqar/v128_n256_h05_seed12345/best_soft.pt')
    p.add_argument('--output',required=True)
    args=p.parse_args()
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');torch.manual_seed(123)
    repo=Path(__file__).resolve().parents[2]
    source=subprocess.check_output(['git','show',args.baseline_commit+':dism_v4/python/flash_dism_v4/blocked.py'],cwd=repo,text=True)
    legacy=types.ModuleType('flash_dism_v4._benchmark_protected')
    legacy.__package__='flash_dism_v4'
    exec(compile(source,'<protected baseline>','exec'),legacy.__dict__)
    original=module.dism_v4_blocked_from_embeddings
    parameters=[];rows=[];reference_grads={};reference_losses={}
    n=args.length
    if args.kind=='operator':
        q=torch.randn(args.batch,n,args.heads,args.vocab,device='cuda',requires_grad=True)
        k=torch.randn_like(q,requires_grad=True)
        v=torch.randn(args.batch,n,args.heads,args.value_dim,device='cuda',requires_grad=True)
        tau=torch.linspace(0,2,args.heads,device='cuda',requires_grad=True)
        cot=torch.randn_like(v)
        parameters=[q,k,v,tau]
    elif args.kind=='model':
        from nanochat.models import ModelSpec,build_model
        cfg=json.loads((repo/'nanochat/configs/dism_v4_123m_ggggggdddddd_slimpajama_3b.json').read_text())
        cfg['config']['sequence_len']=n
        model=build_model(ModelSpec.from_dict(cfg)).cuda();model.init_weights()
        ids=torch.randint(model.config.vocab_size,(args.batch,n+1),device='cuda')
        x,y=ids[:,:-1].contiguous(),ids[:,1:].contiguous()
        cu=torch.arange(args.batch+1,device='cuda',dtype=torch.int32)*n
        parameters=list(model.parameters())
    else:
        sys.path.insert(0,str(Path.home()/'cs/project/mad-lab'))
        from mqar_quick import MadV4
        from mad.model.language_model import LanguageModel
        from mad.model.layers.mlp import SwiGLU
        from mad.data.instances import generate_in_context_recall_instance as generate
        ck=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
        a=ck['args'];n=a['length']-1
        cfg=dict(dim=128,max_length=a['length'],codebook=a['codebook'],hard_prob=.5)
        model=LanguageModel(a['vocab'],[MadV4,SwiGLU,MadV4,SwiGLU],[cfg.copy() for _ in range(4)],
                            dim=128,max_length=a['length'],norm=torch.nn.LayerNorm).cuda()
        model.load_state_dict(ck['model'])
        rng=np.random.default_rng(a['seed']+1)
        pairs=[generate(vocab_size=a['vocab'],seq_len=a['length'],is_training=True,rng=rng,
                        multi_query=True,noise_vocab_size=0) for _ in range(args.batch)]
        x,y=(torch.from_numpy(np.stack(z)).long().cuda() for z in zip(*pairs))
        parameters=list(model.parameters())
        del ck

    try:
        for probability in (0.,.5):
            for label,helper,readout in [('arithmetic_protected',legacy,'arithmetic'),
                                          ('arithmetic',current,'arithmetic'),('jensen',current,'jensen')]:
                def dispatch(*pos,**kwargs):
                    if helper is legacy:kwargs.pop('soft_readout',None)
                    else:kwargs['soft_readout']=readout
                    return helper.dism_v4_blocked_from_embeddings(*pos,**kwargs)
                module.dism_v4_blocked_from_embeddings=dispatch
                if args.kind=='operator':
                    flags=torch.zeros((args.batch,args.heads),device='cuda',dtype=torch.bool)
                    if probability:flags[:,::2]=True
                elif args.kind=='model':
                    model.train();model.hard_probability.fill_(probability)
                else:
                    model.train()
                    for layer in model.modules():
                        if isinstance(layer,MadV4):layer.hard_prob=probability

                def step():
                    for parameter in parameters:parameter.grad=None
                    if args.kind=='operator':
                        kwargs=dict(hard=flags,key_mode='soft')
                        if helper is current:kwargs['soft_readout']=readout
                        out=helper.dism_v4_blocked(q,k,v,tau,**kwargs)
                        loss=(out*cot).sum()
                    elif args.kind=='model':
                        model.rng_counter.zero_()  # same row/head flags in every timed replay
                        loss=model(x,y,cu_seqlens=cu)
                    else:
                        torch.manual_seed(902)  # same row/head flags; no weight updates
                        with torch.autocast('cuda',dtype=torch.bfloat16):out=model(x)
                        loss=F.cross_entropy(out.float().flatten(0,1),y.flatten())
                    loss.backward()
                    return loss.detach()

                for _ in range(2):step()
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                diagnostic_reference_bytes=sum(t.numel()*t.element_size() for t in reference_grads.values())
                durations=[]
                for _ in range(args.repeats):
                    start=time.perf_counter()
                    for _ in range(args.steps):loss=step()
                    torch.cuda.synchronize()
                    durations.append((time.perf_counter()-start)/args.steps)
                seconds=statistics.median(durations)
                peak=torch.cuda.max_memory_allocated()
                grads=torch.cat([(torch.zeros_like(t) if t.grad is None else t.grad).detach().float().flatten() for t in parameters])
                finite=bool(torch.isfinite(loss)) and bool(torch.isfinite(grads).all())
                assert finite
                if label=='arithmetic_protected':
                    reference_grads[probability]=grads.clone()
                    reference_losses[probability]=float(loss)
                reference=reference_grads[probability]
                row=dict(kind=args.kind,variant=label,hard_probability=probability,batch=args.batch,length=n,
                         seconds=seconds,seconds_samples=durations,tokens_per_second=args.batch*n/seconds,
                         peak_allocated_bytes=peak,diagnostic_reference_bytes=diagnostic_reference_bytes,
                         peak_without_reference_bytes=peak-diagnostic_reference_bytes,
                         loss=float(loss),loss_difference=float(loss)-reference_losses[probability],
                         grad_relative_l2=float((grads-reference).norm()/reference.norm().clamp_min(1e-12)),
                         grad_cosine=float(F.cosine_similarity(grads,reference,dim=0)),finite=finite)
                rows.append(row);print(json.dumps(row),flush=True)
                del grads
    finally:
        module.dism_v4_blocked_from_embeddings=original
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(dict(config=vars(args),baseline_sha256=hashlib.sha256(source.encode()).hexdigest(),
                                   parameter_count=sum(t.numel() for t in parameters),results=rows),indent=2)+'\n')


if __name__=='__main__':main()
