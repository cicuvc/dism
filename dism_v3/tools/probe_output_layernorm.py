"""Replace per-head mixer output RMSNorm by LayerNorm, with no added bias."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import probe_initialization_training as screen
import probe_sq_scaling as sq_screen
import torch
from nanochat.models import ModelSpec
from fla.modules.fused_norm_gate import FusedLayerNormGated

ROOT=Path(__file__).resolve().parents[2]


def build(spec,arm,perturb,batch):
    model,details=sq_screen.build(spec,'baseline',perturb,batch)
    params={name:id(p) for name,p in model.named_parameters()}
    for layer in model.layers:
        old=layer.attn.norm
        norm=FusedLayerNormGated(old.hidden_size,elementwise_affine=True,bias=False,
                                activation=old.activation,eps=old.eps,
                                device=old.weight.device,dtype=old.weight.dtype)
        norm.weight=old.weight
        layer.attn.norm=norm
    assert params=={name:id(p) for name,p in model.named_parameters()}
    details['norm_gate']='per-head FLA LayerNorm (no bias), then original SiLU gate'
    return model,details


def collect(output,baseline):
    rows=[]
    for arm,directory in [('baseline',baseline),('layernorm',output)]:
        runs=[json.loads((directory/f'{arm}_{suffix}.json').read_text()) for suffix in ['reference','perturbed']]
        assert all(r['complete'] and len(r['steps'])==300 and r['rng_counter']==300 for r in runs)
        rows.append(dict(arm=arm,
            validation=[dict(step=a['step'],soft=(a['soft']+b['soft'])/2,hard=(a['hard']+b['hard'])/2)
                        for a,b in zip(runs[0]['validation'],runs[1]['validation'])],
            soft_pair_gap=abs(runs[0]['validation'][-1]['soft']-runs[1]['validation'][-1]['soft']),
            clip_fraction=sum(s['grad_norm']>1 for r in runs for s in r['steps'])/600))
    (output/'comparison.json').write_text(json.dumps(rows,indent=2)+'\n')
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--perturb',action='store_true')
    args=parser.parse_args();args.steps=300
    args.output=ROOT/'dism_v3/build/output_layernorm_screen'
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.set_float32_matmul_precision('high')
    ref=Path((ROOT/'dism_v3/build/layout_diagnosis/new_training_run.txt').read_text().strip())
    spec=ModelSpec.from_dict(json.loads((ref/'model_spec.json').read_text()))
    cache_path=ROOT/'dism_v3/build/initialization_screen/batches.pt'
    baseline=ROOT/'dism_v3/build/sq_scaling_screen'
    previous=json.loads((baseline/'protocol.json').read_text())
    assert hashlib.sha256(cache_path.read_bytes()).hexdigest()==previous['cache_sha256']
    assert hashlib.sha256(Path(screen.__file__).read_bytes()).hexdigest()==previous['training_loop_sha256']
    assert spec.to_dict()==previous['model_spec']
    if args.worker:
        cache=torch.load(cache_path,weights_only=False)
        screen.build=build
        screen.run(spec,'layernorm',args.perturb,cache,args)
    else:
        protocol=dict(previous,baseline_reused=str(baseline),
                      intervention='all 12 mixer output norms: RMSNorm -> LayerNorm, same 64 channels per head, original weight, no bias, eps1e-5, SiLU gate unchanged; residual-stream postnorms unchanged',
                      source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        protocol.pop('arms',None)
        protocol['note']='Only mixer output normalization changes; SQ scale 1; paired deliberate embedding perturbation, not independent seeds'
        (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
        for perturb in [False,True]:
            cmd=[sys.executable,str(Path(__file__).resolve()),'--worker']
            if perturb:cmd.append('--perturb')
            subprocess.run(cmd,check=True)
        print(json.dumps(collect(args.output,baseline),indent=2),flush=True)


if __name__=='__main__':main()
