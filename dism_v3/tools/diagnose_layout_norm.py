"""Measure finite soft/hard perturbations immediately before/after tail norms."""
import argparse,json
from pathlib import Path
import torch
from diagnose_layer_layout import ROOT,load,batches,Probe


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint-dir',type=Path,required=True)
    p.add_argument('--batch-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if not a.output.resolve().is_relative_to(ROOT/'dism_v3/build'):
        raise ValueError('output must be under dism_v3/build')
    if a.output.exists():raise FileExistsError(a.output)
    torch.set_num_threads(1)
    model,meta=load(a.checkpoint_dir,22888);probe=Probe(model)
    data=batches(a.batch_dir,22888,4);states={};handles=[];results=[]
    for i in range(probe.layers[-1],12):
        for name,module in [('attn',model.layers[i].fused_attn_norm),('mlp',model.layers[i].mlp_norm)]:
            def capture(module,args,out,key=f'{i}/{name}'):
                if isinstance(out,tuple):
                    states[key+'/pre']=args[0].detach().float()+args[1].detach().float()
                    states[key+'/post']=out[0].detach().clone()
                else:
                    states[key+'/pre']=args[0].detach().clone()
                    states[key+'/post']=out.detach().clone()
            handles.append(module.register_forward_hook(capture))
    try:
        for bi,batch in enumerate(data):
            probe.mask(4,1.);probe.evaluate(batch);reference=states.copy();states.clear()
            probe.flags[5].fill_(False);probe.evaluate(batch)
            metrics={}
            for key,value in states.items():
                ref=reference[key].float();delta=value.float()-ref
                metrics[key]=dict(delta_rms=delta.square().mean().sqrt().item(),
                    baseline_rms=ref.square().mean().sqrt().item(),relative_rms=(delta.square().sum()/ref.square().sum()).sqrt().item())
            results.append(dict(batch=bi,stages=metrics));states.clear()
            a.output.write_text(json.dumps(results,indent=2)+'\n')
    finally:
        for h in handles:h.remove()
        probe.close()

if __name__=='__main__':main()
