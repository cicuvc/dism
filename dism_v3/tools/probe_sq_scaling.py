"""Paired 300-update SQ scaling screen with unchanged production norm/gate.

Only the post-SiLU query readout is scaled; SK and selector are unchanged.
Uses the frozen local batches and training loop from the initialization screen.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import probe_initialization_training as screen
import torch
from nanochat.models import ModelSpec, build_model

ROOT = Path(__file__).resolve().parents[2]
ARMS = {'baseline': 1., 'sq_half': .5, 'sq_rsqrt': 32**-.5}


def build(spec, arm, perturb, batch):
    torch.manual_seed(42)
    with torch.device('meta'):
        model = build_model(spec)
    model.to_empty(device='cuda'); model.init_weights()
    scale = ARMS[arm]
    assert model.config.readout_dim == 32
    for layer in model.layers:
        attn = layer.attn
        if getattr(attn, 'is_pure_mixer', False):
            continue
        original = attn._activate_readout
        def activate(projected, *, is_key=False, heads=None, fn=original):
            out = fn(projected, is_key=is_key, heads=heads)
            return out if is_key or scale == 1. else (out*scale).contiguous()
        attn._activate_readout = activate
    actual = 0.
    if perturb:
        with torch.no_grad():
            p = model.embedding.weight
            before = p.clone()
            gen = torch.Generator(device='cuda').manual_seed(991)
            p.copy_((p.float()+torch.randn(p.shape,device='cuda',generator=gen)*2e-5).to(p.dtype))
            actual = screen.rms(p.float()-before.float())/screen.rms(before)
    model.train()
    return model, dict(sq_scale=scale, embedding_relative_perturbation=actual,
                       norm_gate='unchanged production FLA RMSNorm followed by SiLU gate')


def collect(output):
    rows=[]
    for arm,scale in ARMS.items():
        runs=[json.loads((output/f'{arm}_{suffix}.json').read_text()) for suffix in ['reference','perturbed']]
        assert all(r['complete'] and len(r['steps'])==300 and r['rng_counter']==300 for r in runs)
        rows.append(dict(arm=arm,scale=scale,
            validation=[dict(step=a['step'],soft=(a['soft']+b['soft'])/2,hard=(a['hard']+b['hard'])/2)
                        for a,b in zip(runs[0]['validation'],runs[1]['validation'])],
            soft_pair_gap=abs(runs[0]['validation'][-1]['soft']-runs[1]['validation'][-1]['soft']),
            clip_fraction=sum(s['grad_norm']>1 for r in runs for s in r['steps'])/600))
    (output/'comparison.json').write_text(json.dumps(rows,indent=2)+'\n')
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=list(ARMS))
    parser.add_argument('--perturb',action='store_true')
    args=parser.parse_args()
    args.steps=300
    args.output=ROOT/'dism_v3/build/sq_scaling_screen'
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.set_float32_matmul_precision('high')
    ref=Path((ROOT/'dism_v3/build/layout_diagnosis/new_training_run.txt').read_text().strip())
    spec=ModelSpec.from_dict(json.loads((ref/'model_spec.json').read_text()))
    cache_path=ROOT/'dism_v3/build/initialization_screen/batches.pt'
    if args.arm:
        cache=torch.load(cache_path,weights_only=False)
        screen.build=build
        screen.run(spec,args.arm,args.perturb,cache,args)
    else:
        protocol=dict(steps=300,packed_tokens_per_step=16384,seed=42,lr=8e-4,warmup=500,
                      cuda_graph=True,planner=False,arms=ARMS,model_spec=spec.to_dict(),
                      cache=str(cache_path),cache_sha256=hashlib.sha256(cache_path.read_bytes()).hexdigest(),
                      source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      training_loop_sha256=hashlib.sha256(Path(screen.__file__).read_bytes()).hexdigest(),
                      note='Original production gate and norm unchanged; only SQ after SiLU is scaled; pairs share seed with deliberate small embedding perturbation')
        (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
        for arm in ARMS:
            for perturb in [False,True]:
                command=[sys.executable,str(Path(__file__).resolve()),'--arm',arm]
                if perturb:command.append('--perturb')
                subprocess.run(command,check=True)
        print(json.dumps(collect(args.output),indent=2),flush=True)


if __name__=='__main__':main()
