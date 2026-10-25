"""Local validation/NIAH ablations and tail residual-scaling diagnostics.

Temporary forward interventions only; no weight updates or checkpoint writes.
"""
import argparse,json
from pathlib import Path
from contextlib import contextmanager
import torch
from diagnose_layer_layout import ROOT,load,Probe,batches
from nanochat.tokenizer import get_tokenizer_from_spec
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.niah_eval import ShortNIAH


@contextmanager
def tail_scale(model,layers,kind,scale):
    handles=[]; originals=[]
    try:
        for index in layers:
            layer=model.layers[index]
            if kind in ('attention','both'):
                def hook(module,args,out):return (out[0]*scale,*out[1:])
                handles.append(layer.attn.register_forward_hook(hook))
            if kind in ('mlp','both'):
                fn=layer.mlp.forward_quantized
                originals.append((layer.mlp,fn))
                layer.mlp.forward_quantized=lambda *args,_fn=fn,**kwargs:_fn(*args,**kwargs)*scale
        yield
    finally:
        for h in handles:h.remove()
        for module,fn in originals:module.forward_quantized=fn


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint-dir',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--validation-batches',type=int,default=8)
    args=ap.parse_args()
    if not args.output.resolve().is_relative_to(ROOT/'dism_v3/build'):
        raise ValueError('output must be under dism_v3/build')
    if args.output.exists():raise FileExistsError(args.output)
    torch.set_num_threads(1)
    model,meta=load(args.checkpoint_dir,22888); probe=Probe(model)
    tokenizer=get_tokenizer_from_spec(meta['tokenizer_spec']);cfg=meta['user_config']
    loader=tokenizing_distributed_data_loader_bos_bestfit(tokenizer,8,2048,split='val',device='cuda',
        seq_align=256,data_dir=cfg['data_dir'],shuffle=False,shuffle_seed=1337,aligned_segment_ends=True)
    # The loader reuses staging buffers; each diagnostic batch owns its storage.
    data=[tuple(t.clone() for t in next(loader)[:3]) for _ in range(args.validation_batches)]
    def evaluate():
        records=[probe.evaluate(b) for b in data]
        tokens=sum(r['valid_tokens'] for r in records);docs=sum(r['documents'] for r in records)
        return dict(token_nll=sum(r['token_nll']*r['valid_tokens'] for r in records)/tokens,
                    document_nll=sum(r['document_nll']*r['documents'] for r in records)/docs,
                    valid_tokens=tokens,documents=docs,batches=records)
    result=dict(checkpoint=str(args.checkpoint_dir),validation=[],niah=[],propagation={})
    def save():args.output.write_text(json.dumps(result,indent=2)+'\n')
    def record(name):
        metric=evaluate();result['validation'].append(dict(name=name,**metric));save()
        print(name,metric['token_nll'],flush=True)
    try:
        probe.mask(8,1.);record('baseline_hard')
        probe.mask(8,0.);record('baseline_soft');probe.mask(8,1.)
        probe.scales={j:0. for j in range(6)};record('all_dism_off');probe.scales.clear()
        for j in range(6):
            probe.scales[j]=0.;record(f'dism{j}_off');probe.scales.clear()
        if probe.layers[-1]==8:
            for layers in ([11],[9,10,11]):
                for kind in ('attention','mlp','both'):
                    for scale in (.75,1.25):
                        with tail_scale(model,layers,kind,scale):record(f'tail{layers}_{kind}_{scale}')
            # Locate amplification within the unchanged tail, including both norms.
            states={}; handles=[]
            for i in (8,9,10,11):
                for name,module in [('attn',model.layers[i].fused_attn_norm),('mlp',model.layers[i].mlp_norm)]:
                    def capture(module,args,out,key=f'{i}/{name}'):
                        states[key]=(out[0] if isinstance(out,tuple) else out).detach().clone()
                    handles.append(module.register_forward_hook(capture))
            try:
                probe.mask(8,1.);probe.evaluate(data[0]);reference=states.copy();states.clear()
                probe.flags[5].fill_(False);probe.evaluate(data[0])
                for key,value in states.items():
                    ref=reference[key].float();delta=value.float()-ref
                    result['propagation'][key]=dict(delta_rms=delta.square().mean().sqrt().item(),
                        relative_rms=(delta.square().sum()/ref.square().sum()).sqrt().item())
            finally:
                for h in handles:h.remove()
            save()
        probe.flags.clear()
        niah=ShortNIAH(trials=2)
        for j in [-1,*range(6)]:
            probe.scales.clear()
            if j>=0:probe.scales[j]=0.
            nr=niah.evaluate(model,tokenizer)
            result['niah'].append(dict(ablated_ordinal=j,**nr));save()
            print('NIAH',j,nr['metrics']['niah/present_nll'],flush=True)
    finally:probe.close()

if __name__=='__main__':main()
