"""Compare event/quadratic on immutable real-checkpoint hard-backward inputs."""
import argparse,json,gc
from pathlib import Path
import torch
from diagnose_layer_layout import ROOT,load,batches,Probe
from bench_graph_hard_routes import (GraphPlannerSlot,PackedTrainingGraph,NativeTrainingPlanner,
    cu,VarlenLayout,upload_plan,GraphCompactBackward,backward_hard_recompute)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint-dir',type=Path,required=True)
    ap.add_argument('--batch-dir',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if not args.output.resolve().is_relative_to(ROOT/'dism_v3/build'):
        raise ValueError('output must be under dism_v3/build')
    if args.output.exists():raise FileExistsError(args.output)
    torch.set_num_threads(1)
    model,meta=load(args.checkpoint_dir,22888);model.train()
    batch=batches(args.batch_dir,22888,4)[0]
    probe=Probe(model);probe.mask(4,.95,93)
    retained={}; original=GraphPlannerSlot.backward
    def retain(self,*tensors,**kwargs):
        if torch.cuda.is_current_stream_capturing():retained[self]=tuple(x.detach() for x in tensors)
        return original(self,*tensors,**kwargs)
    GraphPlannerSlot.backward=retain
    try:graph=PackedTrainingGraph(model,*batch,planner=True)
    finally:GraphPlannerSlot.backward=original
    snapshots=[]
    with graph:
        model.zero_grad(set_to_none=True)
        graph.forward_backward(*batch);graph.drain()
        with torch.cuda.stream(graph.stream):
            for layer,slot in zip(probe.layers,graph.runtime.slots):
                snapshots.append(dict(layer=layer,inputs=tuple(x.clone() for x in retained[slot]),
                    hosts=tuple(x.clone() for x in slot.hosts),documents=slot.documents.copy(),
                    heads=slot.heads,device=slot.device,budget=slot.budget))
        graph.stream.synchronize()
    probe.close(); del graph,retained,probe,model;gc.collect()
    native=NativeTrainingPlanner(16);stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream());results=[]
    with torch.cuda.stream(stream):
        for snap in snapshots:
            inputs=snap['inputs'];labels,hard,tau=snap['hosts']
            table=cu.make_graph_layout(batch[2].cpu(),batch[0].numel(),2048,2048)
            layout=VarlenLayout.for_graph(table,batch[0].numel())
            host=native.prepare(labels,tau,hard=hard,documents=snap['documents'],compact=True,chunk_size=64)
            payload=upload_plan(host,device=snap['device'],stream=stream,budget_bytes=snap['budget'])
            descriptor=payload.descriptor.to(snap['device'],non_blocking=True)
            choice=torch.ones(1,dtype=torch.int32,device=snap['device'])
            consumer=GraphCompactBackward(descriptor,choice)
            stream.synchronize()
            reference={k:v.clone().float() for k,v in backward_hard_recompute(*inputs,layout=layout).items()}
            out=consumer.backward(*[x.reshape(-1,x.shape[-1]) for x in inputs[:4]],heads=snap['heads'])
            metrics={}
            for k,ref in reference.items():
                a=out[k].float().flatten();b=ref.flatten()
                metrics[k]=dict(relative_l2=((a-b).norm()/b.norm().clamp_min(1e-30)).item(),
                    cosine=torch.nn.functional.cosine_similarity(a,b,dim=0).item(),max_abs=(a-b).abs().max().item())
            row=dict(layer=snap['layer'],statistics=host.statistics,metrics=metrics)
            valid=batch[1].cpu()!=-1
            symbols={}
            distributions=[]
            for side,name in enumerate(('q','k')):
                per_head=[]; probs=[]
                for h in range(snap['heads']):
                    ids=labels[side,:,h,:][valid].long()
                    counts=torch.bincount(ids,minlength=meta['model_spec']['config']['qk_vocab_size']).float()
                    prob=counts/counts.sum();probs.append(prob)
                    entropy=-(prob*prob.clamp_min(1e-30).log()).sum()
                    per_head.append(dict(active=int((counts>0).sum()),effective=entropy.exp().item(),max_share=prob.max().item()))
                symbols[name]=per_head;distributions.append(probs)
            symbols['qk_collision']=[(q*k).sum().item() for q,k in zip(*distributions)]
            row['symbols']=symbols;row['tau']=tau.tolist()
            results.append(row);print(json.dumps(row),flush=True)
            args.output.write_text(json.dumps(results,indent=2)+'\n')
            del payload,consumer,out,reference
    stream.synchronize()

if __name__=='__main__':main()
