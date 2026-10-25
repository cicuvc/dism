"""Fixed candidate panel across checkpoints: soft preferences vs hard costs."""
import argparse,gc,json,math
from pathlib import Path
import torch
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from nanochat.models import ModelSpec,build_model
import flash_dism.compiler as compiler
import flash_dism.kernels.interp_true as interp


def dense_preferences(vals,g,nodes,output):
    """Independent-symbol logit/probability derivatives, FP64 dense soft oracle."""
    result={}
    with torch.no_grad():
        for h in sorted({x['head'] for x in nodes}):
            dev=lambda x:x.to(device='cuda',dtype=torch.float64)
            q,k,sq,sk,v=[dev(x[0,:,h]) for x in vals[:5]]
            eq,ek=[dev(x[h].to(torch.bfloat16)) for x in vals[5:7]]
            assert vals[8]==1.
            n=q.shape[0];tau=dev(vals[7][h]);pk=(k@ek.T).softmax(-1)
            logpq=(q@eq.T).log_softmax(-1)
            scores=logpq@pk.T+tau
            for i in range(1,n):scores[i,1:]+=torch.nn.functional.softplus(scores[i-1,:-1])
            scores.masked_fill_(torch.ones((n,n),device='cuda',dtype=torch.bool).triu(1),-torch.inf)
            mx=scores.amax(-1,keepdim=True).clamp_min(0)
            p=(scores-mx).exp();p/=p.sum(-1,keepdim=True)+(-mx).exp()
            readout=sq@sk.T;o=(p*readout)@v;cot=dev(g[0,:,h])*n
            t=p*(readout*(cot@v.T)-(cot*o).sum(-1,keepdim=True))
            for i in range(n-1,0,-1):
                previous=scores[i-1,:-1]
                alpha=torch.where(previous>20,torch.ones_like(previous),previous.sigmoid())
                t[i-1,:-1]+=t[i,1:]*alpha
            ref=dev(output[0,:,h]);output_error=float((ref-o).norm()/ref.norm().clamp_min(1e-30))
            for x in nodes:
                if x['head']!=h:continue
                pos=x['position']
                if x['side']=='q':
                    probability=logpq[pos].exp();dlogp=t[pos]@pk
                    gp=dlogp/probability.clamp_min(1e-300)
                    gz=dlogp-probability*dlogp.sum()
                else:
                    probability=pk[pos];gp=t[:,pos]@logpq
                    gz=probability*(gp-(probability*gp).sum())
                assert torch.isfinite(gp).all() and torch.isfinite(gz).all()
                result[(h,x['side'],pos)]=dict(probability=probability.cpu().tolist(),prob_gradient=gp.cpu().tolist(),logit_gradient=gz.cpu().tolist(),
                    logit_gradient_norm=float(gz.norm()),oracle_output_relative_error=output_error)
            del scores,p,readout,t,o
    return result


def verify_derivative():
    # Validate the independent-symbol derivative used for candidate rankings.
    torch.manual_seed(73);n,vocab,d=8,5,3
    zq=torch.randn(n,vocab,dtype=torch.float64,requires_grad=True)
    zk=torch.randn(n,vocab,dtype=torch.float64,requires_grad=True)
    value=torch.randn(n,d,dtype=torch.float64);s=torch.randn(n,n,dtype=torch.float64);g=torch.randn(n,d,dtype=torch.float64)
    logq=zq.log_softmax(-1);pk=zk.softmax(-1);m=logq@pk.T+.2
    rows=[];prev=torch.full((n,),-torch.inf,dtype=torch.float64)
    for i in range(n):
        prev=m[i]+torch.nn.functional.softplus(torch.nn.functional.pad(prev[:-1],(1,0),value=-torch.inf));rows.append(prev)
    w=torch.stack(rows).masked_fill(torch.ones(n,n,dtype=torch.bool).triu(1),-torch.inf)
    probs=torch.cat([torch.zeros(n,1,dtype=torch.float64),w],-1).softmax(-1)[:,1:]
    o=(probs*s)@value
    actual=torch.autograd.grad((o*g).sum(),[zq,zk])
    with torch.no_grad():
        t=probs*(s*(g@value.T)-(g*o).sum(-1,keepdim=True))
        for i in range(n-1,0,-1):t[i-1,:-1]+=t[i,1:]*torch.where(w[i-1,:-1]>20,1.,w[i-1,:-1].sigmoid())
        a=t@pk;dq=a-logq.exp()*a.sum(-1,keepdim=True)
        b=t.T@logq;dk=pk*(b-(pk*b).sum(-1,keepdim=True))
        for found,expected in zip([dq,dk],actual):torch.testing.assert_close(found,expected,atol=1e-10,rtol=1e-9)
    print('FP64 independent-logit derivative check passed',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--panel',required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.28);torch.backends.cuda.matmul.allow_tf32=False
    verify_derivative()
    ck=Path(a.checkpoint);meta=json.loads(ck.with_name(ck.name.replace('model_','meta_')).with_suffix('.json').read_text())
    with torch.device('meta'):model=build_model(ModelSpec.from_dict(meta['model_spec']))
    model.to_empty(device='cuda');model.init_weights();state=torch.load(ck,map_location='cpu',weights_only=True)
    model.load_state_dict({k.removeprefix('_orig_mod.'):v for k,v in state.items()},assign=True);del state
    model.cuda().eval();model.eval_hard=True;model.mxfp8_ce=None;model.mxfp8_ce_norm=None
    from flash_mxfp8 import refresh_fp8_modules
    refresh_fp8_modules(model)
    tok=AutoTokenizer.from_pretrained('/home/cicuvc/downstream-eval/tokenizer',local_files_only=True)
    bos=tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    tokens=[tok.encode(r['text'],add_special_tokens=False) for r in pq.read_table('/home/cicuvc/downstream-eval/data/pg19/test.parquet').to_pylist()[:2]]
    panel=json.loads(Path(a.panel).read_text())['records'];layers=[i for i,l in enumerate(model.layers) if not getattr(l.attn,'is_pure_mixer',False)]
    count=0;current=None;mode='hard';collect=False;change=None;local_layer=None
    labels={};hard_inputs={};inputs={};outputs={}
    original_voc=compiler.voc_forward;original_emb=interp.emb_fwd_true
    def emb(*args,**kwargs):
        r=list(original_emb(*args,**kwargs))
        if collect:labels[current]=(r[7].detach().cpu().clone(),r[6].detach().cpu().clone())
        else:
            r[7]=labels[current][0].to(r[7].device);r[6]=labels[current][1].to(r[6].device)
            if change is not None and current==change['layer']:
                idx=7 if change['side']=='q' else 6;r[idx]=r[idx].clone();r[idx][0,change['head'],change['position']]=change['symbol']
        return tuple(r)
    def voc(*args,**kwargs):
        nonlocal count,current
        current=layers[count];count+=1;args=list(args)
        if current in [6,9]:
            if collect:hard_inputs[current]=[x.detach().cpu().clone() if torch.is_tensor(x) else x for x in args]
            if mode=='all_soft' or (mode=='local_soft' and current==local_layer):
                if mode=='local_soft':
                    for x,y in zip(args[:8],hard_inputs[current][:8]):torch.testing.assert_close(x.detach().cpu(),y,atol=0,rtol=0)
                inputs[current]=[x.detach().cpu().clone() if torch.is_tensor(x) else x for x in args]
        if mode=='all_soft' or (mode=='local_soft' and current==local_layer):args[10]=torch.zeros_like(args[10])
        r=original_voc(*args,**kwargs)
        if current in [6,9]:outputs[current]=r[0]
        return r
    compiler.voc_forward=voc;interp.emb_fwd_true=emb
    def clear():
        outputs.clear();torch.cuda.synchronize()
        for m in model.modules():
            if type(m).__module__.startswith('flash_mxfp8.'):
                for name in ['_forward_states','_forward_plans','_backward_states','_backward_plans','_states']:
                    value=getattr(m,name,None)
                    if isinstance(value,dict):value.clear()
        gc.collect()
    rng=model.rng_counter.clone();records=[];base_records=[]
    for n,doc in [(2048,0),(2048,1),(8192,0)]:
        model.config.sequence_len=n;clear();nodes=[x for x in panel if x['context']==n and x['document']==doc]
        t=tokens[doc];ids=torch.tensor([[bos]+t[:n-1]],device='cuda');y=torch.tensor([t[:n]],device='cuda');cu=torch.tensor([0,n],device='cuda',dtype=torch.int32)
        def hard_eval(intervention=None):
            nonlocal count,mode,change
            count=0;mode='hard';change=intervention
            with torch.no_grad():v=model(ids,y,cu_seqlens=cu,loss_reduction='none').detach().double().cpu()
            outputs.clear();return v
        collect=True;base=hard_eval();collect=False;assert torch.equal(base,hard_eval())
        costs={}
        for x in nodes:
            h=x['head'];side=x['side'];pos=x['position'];layer=x['layer'];original=int(labels[layer][0 if side=='q' else 1][0,h,pos])
            symbols=sorted({x['original_symbol'],*[c['symbol'] for c in x['candidates']],original})
            rows=[]
            for symbol in symbols:
                val=base if symbol==original else hard_eval(dict(layer=layer,head=h,side=side,position=pos,symbol=symbol))
                delta=val-base;assert float(delta[:pos].abs().max())==0
                rows.append(dict(symbol=symbol,current_original=symbol==original,in_fixed_panel=symbol in {x['original_symbol'],*[c['symbol'] for c in x['candidates']]},delta_sum=float(delta.sum()),delta_mean=float(delta.mean())))
            costs[(layer,h,side,pos)]=(original,rows)
        predictions={};losses={}
        for which,selected in [('all_soft',[6,9]),('local_soft_6',[6]),('local_soft_9',[9])]:
            clear();inputs.clear();count=0;change=None;mode='all_soft' if which=='all_soft' else 'local_soft';local_layer=selected[0]
            with torch.enable_grad():
                loss=model(ids,y,cu_seqlens=cu);snap={i:outputs[i].detach().cpu() for i in selected}
                gradients=torch.autograd.grad(loss,[outputs[i] for i in selected]);grads={i:g.detach().cpu() for i,g in zip(selected,gradients)}
            losses[which]=float(loss.detach());del loss,gradients;clear()
            for layer in selected:
                layernodes=[x for x in nodes if x['layer']==layer]
                pred=dense_preferences(inputs[layer],grads[layer],layernodes,snap[layer])
                for key,value in pred.items():predictions[(which,layer,*key)]=value
            del grads,snap
        for x in nodes:
            layer=x['layer'];h=x['head'];side=x['side'];pos=x['position'];original,rows=costs[(layer,h,side,pos)]
            for r in rows:
                for which in ['all_soft',f'local_soft_{layer}']:
                    pred=predictions[(which,layer,h,side,pos)];symbol=r['symbol']
                    r[which]=dict(probability=pred['probability'][symbol],prob_direction=pred['prob_gradient'][symbol]-pred['prob_gradient'][original],
                        logit_direction=pred['logit_gradient'][symbol]-pred['logit_gradient'][original],logit_gradient_norm=pred['logit_gradient_norm'],oracle_output_relative_error=pred['oracle_output_relative_error'])
            records.append(dict(context=n,document=doc,layer=layer,head=h,side=side,position=pos,selection=x['selection'],original_symbol=original,final_original_symbol=x['original_symbol'],candidates=rows))
        assert torch.equal(model.rng_counter,rng)
        base_records.append(dict(context=n,document=doc,hard_loss=float(base.mean()),**losses))
        (out/'results.json').write_text(json.dumps(dict(checkpoint=str(ck),model_spec=meta['model_spec'],step=meta.get('step'),panel=a.panel,baselines=base_records,records=records,
            protocol='Fixed final-checkpoint candidate panel plus current original; all other hard labels frozen. Compare FP64 soft probability-transfer and logit-margin derivatives to actual hard coordinate costs. all_soft and local_soft (target soft, others hard) backgrounds; local inputs exactly match hard baseline. Soft gradient is at operator symbol distributions, not an optimizer update.'),indent=2)+'\n')
        print(n,doc,'DONE',losses,flush=True)
    compiler.voc_forward=original_voc;interp.emb_fwd_true=original_emb
    (out/'complete').write_text('ok\n')


if __name__=='__main__':main()
