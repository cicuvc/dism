"""Opt-in constant-per-sequence tau scaling; paired NIAH, no checkpoint writes.

This evaluates whole forward passes, not isolated answer rows. The scaling
length is the known sequence budget, frozen for every query in that forward.
It is not a token-by-token tau schedule or a SAM performance benchmark.
"""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import time

import torch

from niah_circuit_probe import ARMS, RUNS, ROOT, atomic_json, digest, evaluate, make_suite
from niah_merge_screen import PATHS


def affine(rule, length, reference=2048, fixed_length=16384):
    if rule == 'baseline':
        return 1., 0.
    n = fixed_length if rule.startswith('fixed_') else length
    if rule.endswith('mul'):
        return max(1., math.log(n)/math.log(reference)), 0.
    if rule.endswith('add'):
        return 1., .5*max(0., math.log(n/reference))
    raise ValueError(rule)


@contextmanager
def scale_tau(model, arm, scope, factor, offset):
    import flash_dism.compiler as compiler
    original = compiler.voc_forward
    active, handles, calls = [], [], []
    for layer in range(6, 12):
        def pre(_module, _args, layer=layer):
            active[:] = [layer]
        handles.append(model.layers[layer].attn.register_forward_pre_hook(pre))

    def wrapper(q,k,sq,sk,v,eq,ek,tau,beta,direction,hard,boundaries,layout=None,*,direction_true=False):
        assert direction_true and bool(hard.all())
        layer = active[0]
        calls.append(layer)
        if factor != 1. or offset != 0.:
            modified = tau.clone()
            heads = list(range(tau.numel())) if scope == 'all' else [h for l,h in PATHS[arm] if l == layer]
            modified[heads] = tau[heads]*factor+offset
            tau = modified
        return original(q,k,sq,sk,v,eq,ek,tau,beta,direction,hard,boundaries,layout,direction_true=direction_true)

    compiler.voc_forward = wrapper
    try:
        yield calls
    finally:
        compiler.voc_forward = original
        for h in handles:
            h.remove()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--trials',type=int,default=8)
    p.add_argument('--seed',type=int,default=20261017)
    p.add_argument('--contexts',type=int,nargs='+',default=[2048,8192,16384])
    p.add_argument('--depths',type=float,nargs='+',default=[.05,.95])
    p.add_argument('--arms',nargs='+',choices=list(ARMS),default=list(ARMS))
    p.add_argument('--tokenizer',default='/home/cicuvc/downstream-eval/tokenizer')
    p.add_argument('--pg19',default='/home/cicuvc/downstream-eval/data/pg19/test.parquet')
    a=p.parse_args()
    if a.out.exists():p.error('output already exists')
    if a.trials < 1:p.error('trials must be positive')
    for name in ('flash_dism','cu_flash_dism'):
        origin=Path(importlib.util.find_spec(name).origin).resolve()
        if not origin.is_relative_to(ROOT/'dism_v3'):raise RuntimeError(f'foreign module {origin}')
    suite=make_suite(a.contexts,a.depths,a.trials,a.seed,a.tokenizer,a.pg19)
    a.out.mkdir(parents=True)
    atomic_json(a.out/'inputs.json',suite)
    spec=importlib.util.spec_from_file_location('original_niah',RUNS/ARMS['old'][0]/'diagnostics/niah.py')
    harness=importlib.util.module_from_spec(spec);spec.loader.exec_module(harness)
    torch.cuda.set_per_process_memory_fraction(.30)
    torch.manual_seed(a.seed)
    cases=[('baseline','all','baseline')]+[(f'{scope}_{rule}',scope,rule)
        for scope in ('selected','all') for rule in ('log_mul','log_add','fixed_mul','fixed_add')]
    for arm in a.arms:
        subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.free,utilization.gpu','--format=csv'],check=True)
        if torch.cuda.mem_get_info()[0] < 10*1024**3:raise RuntimeError('less than 10 GiB free')
        run,tag=ARMS[arm]
        checkpoint=RUNS/run/'base_checkpoints'/tag/'model_019074.pt'
        model,meta=harness.load_ckpt(checkpoint)
        model.eval_hard=True
        buffers={k:v.detach().cpu().clone() for k,v in model.named_buffers()}
        result=dict(arm=arm,checkpoint=str(checkpoint),model_spec=meta['model_spec'],
            suite_sha256=digest(suite),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            reference_length=2048,fixed_control_length=16384,add_alpha=.5,selected_paths=PATHS[arm],
            semantics=__doc__,cells=[])
        for cell in suite['cells']:
            n=cell['context'];model.config.sequence_len=n
            entry=dict(context=n,depth=cell['depth'],measurements={});result['cells'].append(entry)
            unique={}
            for name,scope,rule in cases:
                factor,offset=affine(rule,n)
                signature=('identity',) if (factor,offset)==(1.,0.) else (scope,factor,offset)
                if signature in unique:
                    entry['measurements'][name]=dict(entry['measurements'][unique[signature]],equivalent_to=unique[signature])
                    continue
                begin=time.monotonic();torch.cuda.reset_peak_memory_stats()
                measurements=dict(factor=factor,offset=offset,scope=scope)
                for condition in ('present','absent'):
                    with scale_tau(model,arm,scope,factor,offset) as calls:
                        measurements[condition]=evaluate(model,cell['samples'],condition,1)
                    assert calls==list(range(6,12))*a.trials
                measurements.update(seconds=time.monotonic()-begin,peak_allocated=torch.cuda.max_memory_allocated())
                entry['measurements'][name]=measurements;unique[signature]=name
                atomic_json(a.out/f'{arm}.json',result)
                print(arm,n,cell['depth'],name,f"{measurements['seconds']:.1f}s",flush=True)
            replay=evaluate(model,cell['samples'][:1],'present',1)[0]
            assert replay==entry['measurements']['baseline']['present'][0], 'baseline replay drift'
            atomic_json(a.out/f'{arm}.json',result)
        for k,v in model.named_buffers():
            assert torch.equal(v.detach().cpu(),buffers[k]), k
        result['buffers_unchanged']=True
        atomic_json(a.out/f'{arm}.json',result)
        del model
        torch.cuda.empty_cache()


if __name__=='__main__':main()
