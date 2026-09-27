"""Paired post-norm/pre-output-projection DISM head removal,64 sequences."""
import os
os.environ.setdefault('DISM_TILE_LSE','tanh_finite')
os.environ.setdefault('DISM_BWD_OPT','13')
import json
import time
import hashlib
from pathlib import Path
import numpy as np
import torch


@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    from dism_v2.eval_lm_positions import load_checkpoint,evaluate_context
    root=Path('/media/cicuvc/c63abdf1-0e56-4153-9228-95df5a2f239b/cicuvc/dism-lm-runs')
    study=root/'activation-study-3k-20260910'
    out=study/'head-ablation-64-20260911';out.mkdir(exist_ok=False)
    bundle=torch.load(root/'swa-eval-bundle-20260910.pt',map_location='cpu',weights_only=False)
    x,y=bundle['x'][:64,:2048].long(),bundle['y'][:64,:2048].long()
    prior=json.loads((study/'suffix-diagnostic-64-20260911/report.json').read_text())
    digest=hashlib.sha256(x.numpy().tobytes()+y.numpy().tobytes()).hexdigest()
    assert digest==prior['token_sha256']
    report=dict(token_sha256=digest,sequences=64,context=2048,models={},
        caveats=['Post-norm/pre-o_proj zeroing retains norm statistics and bias; drops one head readout, not its computation.',
                 'Targets selected by suffix enrichment on these same64 sequences: exploratory,not held-out confirmation.',
                 'Controls chosen in the same layer by closest real >=4 match rate,not necessarily unenriched.',
                 'NLL increase shows head readout contribution,not specifically benefit from long suffix recurrence.',
                 'Per-sequence SE does not measure training-seed uncertainty; packed sequences may be correlated.'])
    for arm in ('baseline','width384_72m','gdn_dism_384_matched'):
        model,meta=load_checkpoint(study/arm/'latest.pt')
        r=prior['models'][arm]
        real=np.array(r['real']['per_layer_distant_ge4'])
        null=np.mean([t['per_layer_distant_ge4'] for t in r['shuffle']],0)
        li,hi=np.unravel_index(np.argmax(real-null),real.shape)
        layer=r['layer_ids'][li]
        candidates=[j for j in range(real.shape[1]) if j!=hi]
        control=min(candidates,key=lambda j:abs(real[li,j]-real[li,hi]))
        selection=dict(layer=layer,target_head=int(hi+1),control_head=int(control+1),
            target_real=float(real[li,hi]),target_shuffle=float(null[li,hi]),
            control_real=float(real[li,control]),control_shuffle=float(null[li,control]))
        losses={}
        for name,head in [('intact',None),('target',hi),('control',control)]:
            calls=[0]
            def remove(module,args):
                calls[0]+=1
                z=args[0].clone()
                z[...,int(head)*64:(int(head)+1)*64]=0
                return (z,*args[1:])
            hook=None if head is None else model.blocks[layer-1].dism.o_proj.register_forward_pre_hook(remove)
            pieces=[]
            try:
                for i in range(64):
                    nll=evaluate_context(model,x[i:i+1].cuda(),y[i:i+1].cuda(),1.,779)
                    assert torch.isfinite(nll).all()
                    pieces.append(nll)
                    torch.cuda.synchronize();time.sleep(.1)
            finally:
                if hook is not None:hook.remove()
            assert calls[0]==(0 if head is None else 64)
            losses[name]=torch.cat(pieces)
            if name=='intact':
                old=np.load(study/'suffix-diagnostic-64-20260911'/(arm+'.npz'))['nll']
                torch.testing.assert_close(losses[name],torch.from_numpy(old),atol=0,rtol=0)
            print(arm,name,float(losses[name].mean()),selection,flush=True)
        delta_stats={}
        for name in ('target','control'):
            delta=(losses[name]-losses['intact']).double()
            seq=delta.mean(-1)
            prior_arrays=np.load(study/'suffix-diagnostic-64-20260911'/(arm+'.npz'))
            head_index=hi if name=='target' else control
            mask=torch.from_numpy(prior_arrays['max_distant_suffix'][:,li,head_index]>=4)
            mask[:,:128]=False
            valid=torch.ones_like(mask);valid[:,:128]=False
            delta_stats[name]=dict(nll=float(losses[name].double().mean()),delta=float(seq.mean()),
                se=float(seq.std()/8),per_sequence_delta=seq.tolist(),
                long_match_tokens=int(mask.sum()),
                delta_on_long_match=float(delta[mask].mean()) if mask.any() else None,
                delta_on_other=float(delta[valid&~mask].mean()))
        report['models'][arm]=dict(selection=selection,intact_nll=float(losses['intact'].double().mean()),ablation=delta_stats)
        torch.save(losses,out/(arm+'.pt'))
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        del model;torch.cuda.empty_cache()


if __name__=='__main__':main()
