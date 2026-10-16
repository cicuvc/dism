"""Refine a saved QB calibration on identical data, then evaluate frozen prices."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import time

import torch

from qb_fwedu_probe import Observer, quantile_prices, run_batches, TOKENIZER
from niah_circuit_probe import ARMS, RUNS, atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--iterations',type=int,default=128)
    a=p.parse_args()
    if a.out.exists(): p.error('choose a fresh output directory')
    source=json.loads((a.source/'results.json').read_text())
    assert source['buffers_unchanged']
    a.out.mkdir(parents=True)
    subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.free,utilization.gpu','--format=csv'],check=True)
    if torch.cuda.mem_get_info()[0]<10*1024**3: raise RuntimeError('less than 10 GiB free')
    torch.cuda.set_per_process_memory_fraction(.30)
    torch.backends.cuda.matmul.allow_tf32=False
    inputs=torch.load(a.source/'inputs.pt',map_location='cpu',weights_only=True)
    original_prices=torch.load(a.source/'prices.pt',map_location='cpu',weights_only=True)
    from nanochat.tokenizer import HuggingFaceTokenizer
    token_bytes=HuggingFaceTokenizer.from_pretrained(TOKENIZER).get_token_bytes(device='cuda')
    spec=importlib.util.spec_from_file_location('original_niah',RUNS/ARMS['old'][0]/'diagnostics/niah.py')
    harness=importlib.util.module_from_spec(spec);spec.loader.exec_module(harness)
    model,_=harness.load_ckpt(source['checkpoint']);model.eval_hard=True
    buffers={k:v.detach().cpu().clone() for k,v in model.named_buffers()}
    observer=Observer(model,collect=True,analyze=False)
    with observer.hooked(): replay=run_batches(model,inputs['calibration'],token_bytes,observer)
    assert replay==source['calibration_baseline']
    result=dict(source_directory=str(a.source.resolve()),checkpoint=source['checkpoint'],input_sha256=source['input_sha256'],
                initial_iterations=source['iterations'],additional_iterations=a.iterations,
                probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                solver_sha256=hashlib.sha256(Path(__file__).with_name('qb_fwedu_probe.py').read_bytes()).hexdigest(),
                fits={},arms={'baseline':source['arms']['baseline']})
    prices={}
    for layer in range(6,12):
        scores=torch.cat(observer.scores.pop(layer),dim=1)
        prices[layer]=[]
        for head in range(scores.shape[0]):
            price,fit=quantile_prices(scores[head].cuda(),a.iterations,initial=original_prices[layer][head])
            prices[layer].append(price);result['fits'][f'L{layer:02d}H{head}']=fit
            print('refine',layer,head,'active',fit['calibrated']['active'],'max load',fit['calibrated']['max_load_ratio'],flush=True)
        prices[layer]=torch.stack(prices[layer]);del scores
    del observer
    torch.save({k:v.cpu() for k,v in prices.items()},a.out/'prices.pt')
    atomic_json(a.out/'results.json',result)
    observer=Observer(model,prices=prices,scale=1.)
    torch.cuda.reset_peak_memory_stats();begin=time.monotonic()
    with observer.hooked(): metrics=run_batches(model,inputs['evaluation'],token_bytes,observer)
    result['arms']['k_price_100_refined']=dict(**metrics,heads=observer.report(),seconds=time.monotonic()-begin,
        peak_allocated=torch.cuda.max_memory_allocated(),price_scale=1.,only=None)
    for k,v in model.named_buffers(): assert torch.equal(v.cpu(),buffers[k]),k
    result['buffers_unchanged']=True
    atomic_json(a.out/'results.json',result)
    print('refined NLL',metrics['nll'],'BPB',metrics['bpb'],flush=True)


if __name__=='__main__':
    main()
