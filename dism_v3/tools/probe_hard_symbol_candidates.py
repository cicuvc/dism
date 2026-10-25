"""Local hard-label counterfactuals; weights, other labels and inputs held fixed."""
import argparse,json,gc,math
from pathlib import Path
import torch
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from nanochat.models import ModelSpec,build_model
import flash_dism.compiler as compiler
import flash_dism.hard_forward as hard_module


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.27);torch.backends.cuda.matmul.allow_tf32=False
    ck=Path(a.checkpoint);meta=json.loads(ck.with_name(ck.name.replace('model_','meta_')).with_suffix('.json').read_text())
    with torch.device('meta'):model=build_model(ModelSpec.from_dict(meta['model_spec']))
    model.to_empty(device='cuda');model.init_weights();state=torch.load(ck,map_location='cpu',weights_only=True)
    model.load_state_dict({k.removeprefix('_orig_mod.'):v for k,v in state.items()},assign=True);del state
    model.cuda().eval();model.eval_hard=True;model.mxfp8_ce=None;model.mxfp8_ce_norm=None
    from flash_mxfp8 import refresh_fp8_modules
    refresh_fp8_modules(model)
    tok=AutoTokenizer.from_pretrained('/home/cicuvc/downstream-eval/tokenizer',local_files_only=True)
    bos=tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    rows=pq.read_table('/home/cicuvc/downstream-eval/data/pg19/test.parquet').to_pylist()[:2]
    tokens=[tok.encode(r['text'],add_special_tokens=False) for r in rows]
    layers=[i for i,l in enumerate(model.layers) if not getattr(l.attn,'is_pure_mixer',False)]
    target_layers=[6,9];heads=[0,1]
    original_voc=compiler.voc_forward;original_hard=hard_module.forward_hard
    count=0;current=None;intervention=None;collect=False;operands={};labels={};raw={};raw_change=None
    def voc(*args,**kwargs):
        nonlocal count,current
        current=layers[count];count+=1
        if collect and current in target_layers:
            operands[current]=[args[i].detach().cpu().clone() for i in [0,1,5,6]]
        return original_voc(*args,**kwargs)
    def hard(sq,sk,v,iq,ik,flags,tau,**kwargs):
        nonlocal raw_change
        if collect:
            labels[current]=(iq.cpu().clone(),ik.cpu().clone())
        else:
            # Coordinate intervention on the complete hard assignment: do not
            # let downstream argmax flips confound the single changed node.
            iq=labels[current][0].to(iq.device).contiguous()
            ik=labels[current][1].to(ik.device).contiguous()
        if intervention is not None and current==intervention['layer']:
            h=intervention['head'];pos=intervention['position'];symbol=intervention['symbol']
            if intervention['side']=='q':iq=iq.clone();iq[0,h,pos]=symbol
            else:ik=ik.clone();ik[0,h,pos]=symbol
        result=original_hard(sq,sk,v,iq,ik,flags,tau,**kwargs)
        if collect and current in target_layers:raw[current]=result[0].cpu().clone()
        if intervention is not None and current==intervention['layer']:
            delta=result[0][0,:,intervention['head']].float().cpu()-raw[current][0,:,intervention['head']].float()
            raw_change=dict(rms=float(delta.square().mean().sqrt()),changed_tokens=int((delta!=0).any(-1).sum()))
        return result
    compiler.voc_forward=voc;hard_module.forward_hard=hard
    rng=model.rng_counter.clone();records=[];baselines=[]
    for n,doc in [(2048,0),(2048,1),(8192,0)]:
        for m in model.modules():
            if type(m).__module__.startswith('flash_mxfp8.'):
                for name in ['_forward_states','_forward_plans']:
                    v=getattr(m,name,None)
                    if isinstance(v,dict):v.clear()
        gc.collect();model.config.sequence_len=n
        t=tokens[doc];assert len(t)>=n
        ids=torch.tensor([[bos]+t[:n-1]],device='cuda');y=torch.tensor([t[:n]],device='cuda');cu=torch.tensor([0,n],device='cuda',dtype=torch.int32)
        def evaluate(change=None):
            nonlocal count,intervention
            count=0;intervention=change
            loss=model(ids,y,cu_seqlens=cu,loss_reduction='none').detach().double().cpu()
            assert torch.isfinite(loss).all() and torch.equal(model.rng_counter,rng)
            return loss
        collect=True;base=evaluate();collect=False
        null=evaluate();noise=float((base-null).abs().max());assert noise==0,'Identical replay is not exact'
        baselines.append(dict(context=n,document=doc,nll=float(base.mean()),repeat_max_error=noise))
        for layer in target_layers:
            for head in heads:
                for side in ['q','k']:
                    vec=operands[layer][0 if side=='q' else 1][0,:,head].float()
                    vocab=operands[layer][2 if side=='q' else 3][head].to(torch.bfloat16).float()
                    logits=vec@vocab.T
                    top=logits.topk(4,-1);margin=top.values[:,0]-top.values[:,1]
                    ambiguous=n//4+int(margin[n//4:n//2].argmin())
                    for selection,pos in [('fixed',n//2),('small_margin',ambiguous)]:
                        original=int(labels[layer][0 if side=='q' else 1][0,head,pos])
                        candidates=[int(c) for c in top.indices[pos] if int(c)!=original][:3]
                        prob=logits[pos].softmax(-1)
                        common=dict(layer=layer,head=head,position=pos,side=side)
                        noop=evaluate(dict(common,symbol=original));assert torch.equal(noop,base),'Label override no-op mismatch'
                        local=[]
                        for candidate in candidates:
                            cost=evaluate(dict(common,symbol=candidate));delta=cost-base
                            assert float(delta[:pos].abs().max())==0,'Noncausal prefix change'
                            opp=labels[layer][1 if side=='q' else 0][0,head]
                            opp=opp[:pos+1] if side=='q' else opp[pos:]
                            local.append(dict(symbol=candidate,probability=float(prob[candidate]),
                                delta_nll=float(delta.mean()),delta_nll_sum=float(delta.sum()),
                                changed_loss_tokens=int((delta!=0).sum()),max_token_loss_change=float(delta.abs().max()),
                                old_direct_matches=int((opp==original).sum()),new_direct_matches=int((opp==candidate).sum()),
                                raw_output_change=raw_change))
                        best=min(local,key=lambda x:x['delta_nll_sum'])
                        repeated=evaluate(dict(common,symbol=best['symbol']))
                        repeat_delta=float((repeated-base).sum())
                        assert repeat_delta==best['delta_nll_sum'],'Best candidate repeat mismatch'
                        records.append(dict(context=n,document=doc,selection=selection,original_symbol=original,
                            original_probability=float(prob[original]),logit_margin=float(margin[pos]),
                            entropy=float(-(prob*prob.clamp_min(1e-30).log()).sum()),**common,candidates=local,
                            repeated_best_delta=repeat_delta,no_op_max_error=float((noop-base).abs().max())))
                        (out/'results.json').write_text(json.dumps(dict(checkpoint=str(ck),model_spec=meta['model_spec'],
                            protocol='All-hard inference; one Q or K label at one token/head replaced by top-logit candidates; all other hard labels frozen to baseline across all layers; model params unchanged; BF16 CE summed in FP64; fixed physical batch1; prefix causality and repeated/no-op checks. Probability ranking is selector confidence, not a soft loss gradient.',baselines=baselines,records=records),indent=2)+'\n')
                        print(n,doc,layer,head,side,selection,pos,'best_delta_sum',best['delta_nll_sum'],flush=True)
    compiler.voc_forward=original_voc;hard_module.forward_hard=original_hard
    (out/'complete').write_text('ok\n')


if __name__=='__main__':main()
