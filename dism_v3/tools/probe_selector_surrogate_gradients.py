"""Same-input single-layer hard intervention and soft-Jacobian selector VJPs."""
import argparse,json,math,gc
from pathlib import Path
import torch
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from nanochat.models import ModelSpec,build_model
import flash_dism.compiler as compiler
from probe_soft_hard_heads import measures


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--audit',action='store_true')
    p.add_argument('--dump-operands',action='store_true')
    p.add_argument('--layers',type=int,nargs='+')
    p.add_argument('--document-start',type=int,default=0)
    p.add_argument('--documents',type=int,default=4);p.add_argument('--contexts',type=int,nargs='+',default=[2048,8192])
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.30)
    ck=Path(a.checkpoint);meta=json.loads(ck.with_name(ck.name.replace('model_','meta_')).with_suffix('.json').read_text())
    with torch.device('meta'):model=build_model(ModelSpec.from_dict(meta['model_spec']))
    model.to_empty(device='cuda');model.init_weights()
    state=torch.load(ck,map_location='cpu',weights_only=True)
    model.load_state_dict({k.removeprefix('_orig_mod.'):v for k,v in state.items()},assign=True);del state
    model.cuda().eval();model.eval_hard=False;model.mxfp8_ce=None;model.mxfp8_ce_norm=None
    from flash_mxfp8 import refresh_fp8_modules
    refresh_fp8_modules(model)
    tokenizer=AutoTokenizer.from_pretrained('/home/cicuvc/downstream-eval/tokenizer',local_files_only=True)
    bos=tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    rows=pq.read_table('/home/cicuvc/downstream-eval/data/pg19/test.parquet').to_pylist()[a.document_start:a.document_start+a.documents]
    tokens=[tokenizer.encode(r['text'],add_special_tokens=False) for r in rows]
    layers=[i for i,l in enumerate(model.layers) if not getattr(l.attn,'is_pure_mixer',False)]
    norms={}
    if a.dump_operands:
        for i in layers:
            def save_norm(module,args,index=i):norms[index]=args[0].detach().cpu().clone()
            model.layers[i].attn.norm.register_forward_pre_hook(save_norm)
    original=compiler.voc_forward;capture={};inputs={};index=0;target_layer=None;save_inputs=False
    def wrapped(*args,**kwargs):
        nonlocal index
        layer=layers[index];index+=1
        args=list(args)
        if save_inputs:
            inputs[layer]=([x.detach().cpu().clone() if torch.is_tensor(x) else x for x in args],kwargs.copy())
        elif layer==target_layer:
            # Verify upstream operands match the all-soft reference exactly.
            for actual,expected in zip(args[:8],inputs[layer][0][:8]):
                torch.testing.assert_close(actual.detach().cpu(),expected,rtol=0,atol=0)
            args[10]=torch.ones_like(args[10])
        result=original(*args,**kwargs);capture[layer]=result[0]
        return result
    compiler.voc_forward=wrapped
    def release_workspaces():
        # Partial VJPs intentionally stop before upstream FFNs; release their
        # unconsumed MXFP8 forward/backward caches between independent probes.
        torch.cuda.synchronize()
        for module in model.modules():
            if type(module).__module__.startswith('flash_mxfp8.'):
                for name in ('_forward_states','_backward_states','_forward_plans','_backward_plans','_states'):
                    value=getattr(module,name,None)
                    if isinstance(value,dict):value.clear()
        gc.collect()
    records=[];rng=model.rng_counter.clone()
    for n in a.contexts:
        model.config.sequence_len=n
        for doc,t in enumerate(tokens,start=a.document_start):
            assert len(t)>=n
            ids=torch.tensor([[bos]+t[:n-1]],device='cuda');labels=torch.tensor([t[:n]],device='cuda')
            cu=torch.tensor([0,n],device='cuda',dtype=torch.int32)
            index=0;save_inputs=True;target_layer=None;capture.clear();inputs.clear()
            loss=model(ids,labels,cu_seqlens=cu);soft_loss=float(loss.detach())
            soft_norms=dict(norms)
            soft_outputs={i:capture[i].detach().cpu().clone() for i in layers} if a.dump_operands else {}
            gs=torch.autograd.grad(loss,[capture[i] for i in layers])
            gs={i:g.detach().cpu() for i,g in zip(layers,gs)}
            del loss;capture.clear();release_workspaces()
            for layer in (a.layers or layers):
                index=0;save_inputs=False;target_layer=layer;capture.clear()
                loss=model(ids,labels,cu_seqlens=cu);hard_loss=float(loss.detach())
                gh=torch.autograd.grad(loss,capture[layer])[0].detach()
                if a.dump_operands:
                    torch.save(dict(inputs=inputs[layer],g_soft=gs[layer],g_hard=gh.cpu(),
                        u_soft=soft_norms[layer],u_hard=norms[layer],o_soft=soft_outputs[layer],
                        norm_type=model.config.mixer_output,layer=layer,document=doc,context=n),
                        out/f'operands_{n}_{doc}_{layer}.pt')
                del loss;capture.clear();release_workspaces()
                vals,kw=inputs[layer]
                vals=[v.cuda() if torch.is_tensor(v) and j != 12 else v for j,v in enumerate(vals)]
                selected=[0,1,5,6]
                for j in selected:vals[j].requires_grad_(True)
                output=original(*vals,**kw)[0]
                # Both VJPs use this identical all-soft graph and its own caches.
                gsoft=torch.autograd.grad(output,[vals[j] for j in selected],gs[layer].cuda(),retain_graph=True)
                gsur=torch.autograd.grad(output,[vals[j] for j in selected],gh)
                audit=None
                if a.audit:
                    fresh=original(*vals,**kw)[0]
                    check=torch.autograd.grad(fresh,[vals[j] for j in selected],gh)
                    audit={name:float((g.float()-ref.float()).norm()/ref.float().norm().clamp_min(1e-30)) for name,g,ref in zip(['q','k','q_vocab','k_vocab'],check,gsur)}
                    del fresh,check
                stats={}
                for name,g1,g2 in zip(['q','k','q_vocab','k_vocab'],gsoft,gsur):
                    assert torch.isfinite(g1).all() and torch.isfinite(g2).all()
                    if name.endswith('vocab'):g1=g1.transpose(0,1);g2=g2.transpose(0,1)
                    else:g1=g1.squeeze(0);g2=g2.squeeze(0)
                    stats[name]=measures(g1.detach().cpu().float()*n,g2.detach().cpu().float()*n)
                    # Rename hard-labelled generic columns: both gradients are soft VJPs.
                assert torch.equal(rng,model.rng_counter)
                assert math.isfinite(hard_loss)
                records.append(dict(context=n,document=doc,layer=layer,soft_loss=soft_loss,local_hard_loss=hard_loss,
                    fresh_graph_relative_error=audit,upstream=measures(gs[layer].float().squeeze(0)*n,gh.cpu().float().squeeze(0)*n),selectors=stats))
                del output,gsoft,gsur,vals,gh,g1,g2
                release_workspaces()
                (out/'results.json').write_text(json.dumps(dict(checkpoint=str(ck),model_spec=meta['model_spec'],
                    protocol='All-soft baseline; switch one DISM layer (all heads) to hard, exact same upstream operands, downstream layers remain soft. Compare J_soft^T g_soft vs J_soft^T g_local_hard at identical soft operands. q/k are DISM operator inputs; vocabulary gradients are parameter gradients. Generic hard_* metric labels mean surrogate, not true hard derivatives. Gradients rescaled by N.',records=records),indent=2))
                print(n,doc,layer,'loss',soft_loss,hard_loss,'peakGB',torch.cuda.max_memory_allocated()/1e9,flush=True)
    compiler.voc_forward=original
    (out/'complete').write_text('ok\n')


if __name__=='__main__':main()
