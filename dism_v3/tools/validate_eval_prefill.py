"""Offline paired checkpoint validation of downstream SAM/quad model prefill.

Loads the existing downstream adapter and local PG19 text; no downloads or
full task rerun. Reports numerical drift rather than asserting bitwise model
identity across differently rounded hard cores and discrete later routing.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import time
from types import SimpleNamespace

import torch
import pyarrow.parquet as pq


@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--suite',type=Path,default=Path.home()/'downstream-eval')
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--lengths',type=int,nargs='+',default=[256,2048,8192])
    p.add_argument('--precision',choices=('tf32x3','bf16'),default='tf32x3')
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    spec=importlib.util.spec_from_file_location('eval_adapter',args.suite/'adapter_layout_bounded.py')
    adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
    lm=adapter.NanochatLM(str(args.checkpoint),str(args.suite/'tokenizer'),args.batch_size,
                         max(args.lengths),hard_backend='sam',prefill_precision=args.precision)
    import flash_dism.kernels.interp_true as interp
    original=interp.emb_fwd_true
    recorded=[]
    def capture(*a,**kw):
        result=original(*a,**kw)
        recorded.append(tuple(x.clone() for x in result[-2:]))
        return result
    result=dict(checkpoint=str(args.checkpoint),precision=args.precision,batch=args.batch_size,cases=[])
    try:
        lm.smoke()
        batch=next(pq.ParquetFile(args.suite/'data/pg19/test.parquet').iter_batches(batch_size=4,columns=['text']))
        books=[lm.tok_encode(text,add_special_tokens=False) for text in batch.column(0).to_pylist()]
        for length in args.lengths:
            sequences=[[lm.prefix_token_id]+books[i%len(books)][:length-1] for i in range(args.batch_size)]
            x,cu=lm._inputs(sequences)
            outputs={};routes={};times={}
            for backend in ('quadratic','sam'):
                lm.hard_backend=backend
                recorded.clear();interp.emb_fwd_true=capture
                hidden=lm._features(x,cu)
                interp.emb_fwd_true=original
                routes[backend]=list(recorded)
                outputs[backend]=hidden.clone()
                samples=[]
                for _ in range(args.repeats):
                    torch.cuda.synchronize();start=time.perf_counter()
                    lm._features(x,cu)
                    torch.cuda.synchronize();samples.append((time.perf_counter()-start)*1000)
                times[backend]=samples
            assert len(routes['quadratic']) == len(routes['sam']) > 0
            a,b=outputs['quadratic'],outputs['sam']
            target=torch.cat((x[:,1:],torch.full_like(x[:,:1],-1)),1)
            losses={}
            for backend,hidden in outputs.items():
                losses[backend]=adapter.fused_cross_entropy_unreduced(hidden.reshape(1,-1,hidden.shape[-1]),lm.weight,target.reshape(1,-1),ignore_index=-1,softcap=lm.model.config.softcap).view_as(target)
            selected=torch.linspace(0,x.shape[1]-1,32,device='cuda').long()
            logits=[torch.nn.functional.linear(h[:,selected].reshape(-1,h.shape[-1]).float(),lm.weight.float()) for h in (a,b)]
            logits=[lm.model.config.softcap*torch.tanh(z/lm.model.config.softcap) for z in logits]
            delta=(a.float()-b.float())
            row=dict(length=length,forward_ms=times,hidden_relative_l2=float(delta.norm()/a.float().norm()),
                hidden_max_abs=float(delta.abs().max()),sampled_logit_max_abs=float((logits[0]-logits[1]).abs().max()),
                sampled_greedy_agreement=float((logits[0].argmax(-1)==logits[1].argmax(-1)).float().mean()),
                mean_nll={k:float(v[:,:-1].double().mean()) for k,v in losses.items()},
                mean_abs_token_nll_delta=float((losses['sam']-losses['quadratic'])[:,:-1].abs().mean()),
                route_disagreement=[{name:float((a!=b).float().mean()) for name,a,b in zip(('key','query'),left,right)}
                                    for left,right in zip(routes['quadratic'],routes['sam'])])
            result['cases'].append(row);print(json.dumps(row),flush=True)
            args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2))
            del outputs,routes,logits,losses,a,b,hidden
        req=[SimpleNamespace(args=(text,dict(max_gen_toks=8,until=['\n']))) for text in
             ('The capital of France is','The solar system consists of','Once upon a time','The scientific method')]
        result['generation']={}
        result['short_likelihood']={}
        for backend in ('quadratic','sam'):
            lm.hard_backend=backend
            result['generation'][backend]=lm.generate_until(req)
            toks=lm.tok_encode('The capital of France is Paris.')
            result['short_likelihood'][backend]=lm._loglikelihood_tokens([(None,toks[:4],toks[4:])])[0]
        print(json.dumps({k:result[k] for k in ('generation','short_likelihood')}),flush=True)
        args.output.write_text(json.dumps(result,indent=2))
    finally:
        interp.emb_fwd_true=original
        lm.close()


if __name__=='__main__':main()
