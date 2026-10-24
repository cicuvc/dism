"""Matched explicit-norm 2x2 screen: RMS/LayerNorm and original/sigmoid gate."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import torch
from torch.nn import functional as F
import probe_initialization_training as screen
from nanochat.models import ModelSpec

ROOT=Path(__file__).resolve().parents[2]
ORIGINAL_BUILD=screen.build


def build(spec,arm,perturb,batch):
    sigmoid=arm=='ln_sigmoid'
    model,details=ORIGINAL_BUILD(spec,'sigmoid' if sigmoid else 'baseline',perturb,batch)
    for i,layer in enumerate(model.layers):
        norm=layer.attn.norm
        scale=details['gate_scales'][i]
        def forward(u,g,m=norm,s=scale,use_sigmoid=sigmoid):
            centered=u.float()-u.float().mean(-1,keepdim=True)
            v=centered*torch.rsqrt(centered.square().mean(-1,keepdim=True)+m.eps)
            gate=g.float()*s if use_sigmoid else F.silu(g.float())
            return (v*m.weight.float()*gate).to(u.dtype)
        norm.forward=forward
    details['output_norm']='explicit per-head LayerNorm, original weight, no added bias'
    return model,details


def collect(output,baseline):
    rows=[]
    for arm,directory in [('baseline',baseline),('sigmoid',baseline),('ln_only',output),('ln_sigmoid',output)]:
        runs=[json.loads((directory/f'{arm}_{suffix}.json').read_text()) for suffix in ['reference','perturbed']]
        assert all(r['complete'] and len(r['steps'])==300 and r['rng_counter']==300 for r in runs)
        rows.append(dict(arm=arm,
            validation=[dict(step=a['step'],soft=(a['soft']+b['soft'])/2,hard=(a['hard']+b['hard'])/2)
                        for a,b in zip(runs[0]['validation'],runs[1]['validation'])],
            soft_pair_gap=abs(runs[0]['validation'][-1]['soft']-runs[1]['validation'][-1]['soft']),
            individual_soft=[r['validation'][-1]['soft'] for r in runs]))
    (output/'comparison.json').write_text(json.dumps(rows,indent=2)+'\n')
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm',choices=['ln_only','ln_sigmoid'])
    p.add_argument('--perturb',action='store_true')
    args=p.parse_args();args.steps=300
    args.output=ROOT/'dism_v3/build/layernorm_sigmoid_screen'
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.set_float32_matmul_precision('high')
    baseline=ROOT/'dism_v3/build/initialization_screen'
    original=json.loads((baseline/'protocol.json').read_text())
    spec=ModelSpec.from_dict(original['model_spec'])
    cache_path=baseline/'batches.pt'
    if args.arm:
        screen.build=build
        screen.run(spec,args.arm,args.perturb,torch.load(cache_path,weights_only=False),args)
    else:
        protocol=dict(original,baseline_reused=str(baseline),
                      source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      training_loop_sha256=hashlib.sha256(Path(screen.__file__).read_bytes()).hexdigest(),
                      cache_sha256=hashlib.sha256(cache_path.read_bytes()).hexdigest(),
                      intervention='2x2: explicit FP32 norm then gate then BF16 cast in all arms; LN subtracts mean; sigmoid gate uses original fixed calibration; SQ scale1, no input scaling')
        (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
        for arm in ['ln_only','ln_sigmoid']:
            for perturb in [False,True]:
                command=[sys.executable,str(Path(__file__).resolve()),'--arm',arm]
                if perturb:command.append('--perturb')
                subprocess.run(command,check=True)
        print(json.dumps(collect(args.output,baseline),indent=2),flush=True)


if __name__=='__main__':main()
