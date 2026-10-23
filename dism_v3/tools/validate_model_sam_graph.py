"""Real checkpoint: eager versus CUDA Graph cached decoding, including rebuilds."""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--length',type=int,default=511)
    p.add_argument('--steps',type=int,default=20)
    p.add_argument('--interval',type=int,default=8)
    p.add_argument('--generation',action='store_true')
    p.add_argument('--shared-prime',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    sys.path.insert(0,'/home/cicuvc/downstream-eval')
    from adapter_layout_bounded import NanochatLM
    from flash_dism.inference import HardDismPrefill
    from nanochat.models.dism.inference import SamDecodeSession,SamGraphDecodeSession
    import pyarrow.parquet as pq
    checkpoint=json.load(open('dism_v3/build/prefill_planner_compare/inputs/manifest.json'))['checkpoint']
    context=max(2048,((a.length+a.steps+255)//256)*256)
    lm=NanochatLM(checkpoint,'/home/cicuvc/downstream-eval/tokenizer',batch_size=1,context=context)
    book=pq.read_table('/home/cicuvc/downstream-eval/data/pg19/test.parquet',columns=['text']).column('text')[10].as_py()
    ids=lm.tok_encode(book)[:a.length+a.steps]
    tokens=torch.tensor([ids],device='cuda')
    rows=[]
    with torch.inference_mode(),HardDismPrefill(workers=16) as prefill:
        eager=SamDecodeSession(lm.model,prefill,context,rebuild_interval=a.interval)
        graph=SamGraphDecodeSession(lm.model,prefill,context,rebuild_interval=a.interval)
        for i in range(a.steps+1):
            current=tokens[:,:a.length] if i==0 else tokens[:,a.length+i-1:a.length+i]
            outputs=[]; times=[]
            for session in (eager,graph):
                torch.cuda.synchronize();start=time.perf_counter()
                if i==0 and session is graph and a.shared_prime:
                    # Diagnostic-only clone of a complete, identical eager prime.
                    # Isolates graph replay from nondeterministic SAM prefill.
                    from flash_dism.inference import HardDismDecoder
                    graph.stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(graph.stream):
                        for src,dst in zip(eager.states,graph.session.states):
                            dst['conv']={k:v.clone() for k,v in src['conv'].items()}
                            dst['gdn']=src['gdn'].clone()
                            d=src['dism']
                            if d is None: continue
                            snap=d.core.snapshot; n=eager.position
                            clone=HardDismDecoder(d.batch,d.heads,d.r,d.dv,d.capacity,
                                snap[9][:d.heads].clone(),device=d.device,planner_backend='gpu',rebuild_interval=a.interval)
                            labels=snap[7][:n].reshape(n,3,d.batch,d.heads)
                            iq=labels[:,1].permute(1,2,0).contiguous()
                            ik=labels[:,0].permute(1,2,0).contiguous()
                            sk=snap[0][:,:n].reshape(d.batch,d.heads,n,d.r).transpose(1,2).contiguous()
                            v=snap[1][:,:n].reshape(d.batch,d.heads,n,d.dv).transpose(1,2).contiguous()
                            clone.prime(iq,ik,sk,v)
                            for index,(old,new) in enumerate(zip(snap,clone.core.snapshot)):
                                assert old.shape==new.shape
                                if index in (0,1): new[:,:n].copy_(old[:,:n])
                                else: new.copy_(old)
                            dst['dism']=clone
                        graph.session.position=eager.position
                        graph.session.batch=eager.batch
                    torch.cuda.current_stream().wait_stream(graph.stream)
                    outputs.append(outputs[0].clone())
                else:
                    outputs.append(session.features(current))
                torch.cuda.synchronize();times.append((time.perf_counter()-start)*1000)
            x,y=outputs
            assert torch.isfinite(x).all() and torch.isfinite(y).all()
            row=dict(step=i,eager_ms=times[0],graph_ms=times[1],captures=graph.captures,
                equal=bool(torch.equal(x,y)),max_abs=float((x.float()-y.float()).abs().max()),
                rel_l2=float((x.float()-y.float()).norm()/x.float().norm()),
                token_equal=bool(torch.equal(lm._greedy(x[:,-1]),lm._greedy(y[:,-1]))))
            print(json.dumps(row),flush=True);rows.append(row)
        with torch.cuda.stream(graph.stream):
            for state in graph.session.states:
                if state['dism'] is not None: assert state['dism'].core.check_status()==graph.position
        eager.close();graph.close()
        if a.generation:
            from types import SimpleNamespace
            requests=[SimpleNamespace(args=(prompt,dict(max_gen_toks=16,until=['\n']))) for prompt in ('The capital of France is','Once upon a time there was')]
            lm.prefill=prefill;lm.decode_backend='sam'
            lm.decode_cuda_graph=False
            expected=lm.generate_until(requests)
            lm.decode_cuda_graph=True
            actual=lm.generate_until(requests)
            print(json.dumps(dict(generation_eager=expected,generation_graph=actual,equal=expected==actual)),flush=True)

    ordinary=[r for j,r in enumerate(rows) if j>2 and r['captures']==rows[j-1]['captures']]
    result=dict(shared_prime=a.shared_prime,checkpoint=checkpoint,prompt=a.length,steps=a.steps,interval=a.interval,rows=rows,
        ordinary_eager_ms=statistics.median(r['eager_ms'] for r in ordinary),
        ordinary_graph_ms=statistics.median(r['graph_ms'] for r in ordinary))
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','checkpoint')}),flush=True)

if __name__=='__main__': main()
