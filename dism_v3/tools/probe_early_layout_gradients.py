"""Fresh full-size layout gradient probes, including identical-block reordering."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import torch
import probe_sq_scaling as base
from nanochat.models import ModelSpec

ROOT=Path(__file__).resolve().parents[2]
ORDER=[0,1,2,6,7,8,9,10,11,3,4,5]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',choices=['old','middle_native','middle_reordered'])
    args=p.parse_args()
    out=ROOT/'dism_v3/build/early_layout_gradients';out.mkdir(parents=True,exist_ok=True)
    if args.case is None:
        for case in ['old','middle_native','middle_reordered']:
            subprocess.run([sys.executable,str(Path(__file__).resolve()),'--case',case],check=True)
        return
    torch.set_num_threads(2);torch.set_float32_matmul_precision('high')
    run=Path((ROOT/'dism_v3/build/layout_diagnosis/new_training_run.txt').read_text().strip())
    config=json.loads((run/'model_spec.json').read_text())
    if args.case=='middle_native':config['config']['layer_layout']='gggddddddggg'
    spec=ModelSpec.from_dict(config)
    cache=torch.load(ROOT/'dism_v3/build/initialization_screen/batches.pt',weights_only=False)
    batch=tuple(t.cuda() for t in cache['train'][0])
    model,_=base.build(spec,'baseline',False,batch)
    if args.case=='middle_reordered':
        layers=list(model.layers)
        model.layers=torch.nn.ModuleList([layers[i] for i in ORDER])
        model.config.layer_layout='gggddddddggg'
        for i,l in enumerate(model.layers):
            if hasattr(l.attn,'layer_idx'):l.attn.layer_idx=i
    tensors={};handles=[]
    def capture(key,value):
        tensors[key]=value
        if value.requires_grad:value.retain_grad()
    for i,l in enumerate(model.layers):
        def ah(m,args,out,i=i):capture(f'{i}/mixer',out[0])
        def bh(m,args,out,i=i):capture(f'{i}/block',out)
        handles.append(l.attn.register_forward_hook(ah));handles.append(l.register_forward_hook(bh))
        if not getattr(l.attn,'is_pure_mixer',False):
            fn=l.attn._combine_cuda
            def combine(d,*args,fn=fn,i=i,**kwargs):
                capture(f'{i}/dism',d)
                result=fn(d,*args,**kwargs)
                capture(f'{i}/sum',result)
                tensors[f'{i}/gdn_approx']=(result.detach().float()-d.detach().float())
                return result
            l.attn._combine_cuda=combine
    model.train();model.set_training_step(0)
    loss=model(batch[0],batch[1],cu_seqlens=batch[2]);loss.backward()
    def stat(t):
        return dict(rms=base.screen.rms(t),norm=t.detach().float().norm().item())
    result=dict(case=args.case,layout=model.config.layer_layout,loss=loss.item(),
                protocol='fixed first 8x2048 packed training batch, fresh seed42, production RMSNorm/gate, all-soft eager backward; no optimizer updates',
                activations={k:dict(value=stat(t),gradient=stat(t.grad) if t.grad is not None else None) for k,t in tensors.items()},
                parameters={n:dict(weight=stat(p),gradient=stat(p.grad) if p.grad is not None else None) for n,p in model.named_parameters()})
    (out/f'{args.case}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(args.case,loss.item(),flush=True)


if __name__=='__main__':main()
