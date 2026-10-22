"""Compare hard backward routes on identical inputs copied from a scratch model.

The model uses real synthetic optimizer warmup, then probes each packing shape.
Retention prevents captured inputs from being recycled; snapshots are copied
before independent captures, with separate owned layouts. This modified model
is not timed as an end-to-end benchmark. Planning/upload are outside GPU timing.
No datasets or user checkpoints are read. Production launch defaults stay fixed.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import hashlib
import re
from types import SimpleNamespace

import torch
import cu_flash_dism as cu

from flash_dism.varlen import VarlenLayout
from flash_dism.inference.training_planner import NativeTrainingPlanner

from nanochat.models import ModelSpec, build_model, PretrainOptimizerConfig
from nanochat.cuda_graph import PackedTrainingGraph
from nanochat.document_loss import DocumentLossAccumulator
from flash_dism.planner_graph import GraphPlannerSlot, upload_plan
from flash_dism.kernels.planner_graph import GraphCompactBackward, sample_route
from flash_dism.kernels.hard_backward_fused import CompactFusedHardPrefillBackward
from flash_dism.hard_backward import backward_hard_recompute


def measure(fn, repeats, samples):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=torch.cuda.current_stream()):
        output = fn()
    for _ in range(5):
        graph.replay()
    times = []
    for _ in range(samples):
        start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        start.record()
        for _ in range(repeats):
            graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end)/repeats)
    return dict(median_ms=statistics.median(times), samples_ms=times), output


def kernel_resources(kernels, directory):
    result=[]
    for kernel in kernels:
        ptx=kernel.asm['ptx']
        key=hashlib.sha256(ptx.encode()).hexdigest()[:16]
        if directory is not None:
            directory.mkdir(exist_ok=True,parents=True)
            (directory/f'{key}.ptx').write_text(ptx)
            (directory/f'{key}.cubin').write_bytes(kernel.asm['cubin'])
        result.append(dict(name=kernel.name,key=key,registers=kernel.n_regs,
                           spills=kernel.n_spills,shared_bytes=kernel.metadata.shared,
                           warps=kernel.metadata.num_warps,
                           ptx_calls=len(re.findall(r'\bcall(?:\.|\s)',ptx))))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-spec', required=True)
    p.add_argument('--rows', type=int, default=8)
    p.add_argument('--training-step', type=int, default=22887)
    p.add_argument('--warmup-steps', type=int, default=8)
    p.add_argument('--repeats', type=int, default=32)
    p.add_argument('--samples', type=int, default=5)
    p.add_argument('--event-ctas',type=int,nargs='*',default=[],
                   help='additional diagnostic graph-event grids')
    p.add_argument('--case-layer',type=int,help='zero-based DISM layer to probe; default all')
    p.add_argument('--dump-kernels',action='store_true')
    p.add_argument('--schedule-ablations',action='store_true',
                   help='compare indirect/direct pointers and loop/one-task scheduling')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]/'build'
    if args.output.exists() or not args.output.resolve().is_relative_to(root):
        p.error('output must be a fresh path under dism_v3/build')
    if os.environ.get('DISM_HARD_PLANNER','0') != '0':
        p.error('use DISM_HARD_PLANNER=0; the model opts into graph planner explicitly')
    if min(args.rows,args.warmup_steps,args.repeats,args.samples) < 1:
        p.error('counts must be positive')
    if any(c < 1 for c in args.event_ctas):
        p.error('CTA counts must be positive')
    torch.set_num_threads(1)
    torch.manual_seed(1741)
    torch.set_float32_matmul_precision('high')
    spec = ModelSpec.from_dict(json.loads(Path(args.model_spec).read_text()))
    model = build_model(spec).cuda().train()
    model.init_weights()
    model.set_training_step(args.training_step)
    opt = model.setup_pretraining_optimizer(PretrainOptimizerConfig(.0008,.0008,.0008,.0008,.01))
    n, vocab = model.config.sequence_len, model.config.vocab_size
    if n < 512 or n % 512:
        p.error('requires a sequence length divisible by 512')
    batches=[]
    parts=([n],[256,n-256],[512]*(n//512),[256]*(n//256))
    for micro in range(8):
        ids=torch.randint(vocab,(args.rows,n),pin_memory=True)
        labels=torch.randint(vocab,ids.shape,pin_memory=True)
        labels[:,::17]=-1
        ends=[0]
        for _ in range(args.rows):
            for length in parts[micro%4]:
                ends.append(ends[-1]+length)
        batches.append((ids.cuda(),labels.cuda(),torch.tensor(ends,dtype=torch.int32)))
    retained={}
    original=GraphPlannerSlot.backward
    def retain(self,*tensors,**kwargs):
        if torch.cuda.is_current_stream_capturing():
            retained[self]=(tuple(x.detach() for x in tensors),kwargs)
        return original(self,*tensors,**kwargs)
    GraphPlannerSlot.backward=retain
    try:
        graph=PackedTrainingGraph(model,*batches[0],microbatches=8,planner=True)
    finally:
        GraphPlannerSlot.backward=original
    rows=[]
    snapshots=[]
    with graph:
        for _ in range(args.warmup_steps):
            acc=DocumentLossAccumulator(8,'cuda')
            for batch in batches:
                acc.add(*graph.forward_backward(*batch))
            graph.drain()
            acc.finalize(model.parameters())
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step()
            model.zero_grad(set_to_none=True)
        print('synthetic optimizer warmup complete',flush=True)
        layer_indices=[i for i,b in enumerate(graph.bindings) if b is not None]
        for pack,batch in enumerate(batches[:4]):
            model.zero_grad(set_to_none=True)
            graph.forward_backward(*batch)
            graph.drain()
            with torch.cuda.stream(graph.stream):
                for layer,slot in zip(layer_indices,graph.runtime.slots):
                    if args.case_layer is not None and layer != args.case_layer:
                        continue
                    inputs,_=retained[slot]
                    labels,hard,tau=slot.hosts
                    for i,x in enumerate(inputs[4:6]):
                        unpacked=x[0].reshape(slot.heads,slot.rows,n).permute(1,0,2)
                        torch.testing.assert_close(unpacked.cpu(),labels[i],atol=0,rtol=0)
                    torch.testing.assert_close(inputs[6].cpu(),hard,atol=0,rtol=0)
                    torch.testing.assert_close(inputs[7].cpu(),tau,atol=0,rtol=0)
                    snapshots.append(dict(pack=pack,layer=layer,
                        boundaries=batch[2].clone(),inputs=tuple(x.clone() for x in inputs),
                        hosts=tuple(x.clone() for x in slot.hosts),documents=slot.documents.copy(),
                        heads=slot.heads,device=slot.device,budget=slot.budget))
            graph.stream.synchronize()
            print(f'snapshotted pack {pack}',flush=True)
    # Finish every model replay before isolated captures. Each probe owns its
    # inputs and native layout; model graph storage is never reused by a probe.
    native=NativeTrainingPlanner(16)
    if not snapshots:
        p.error('case-layer did not select a DISM layer')
    kernel_dir=args.output.with_suffix('')/'kernels' if args.dump_kernels else None
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for snapshot in snapshots:
            pack,layer=snapshot['pack'],snapshot['layer']
            inputs=snapshot['inputs']
            labels,hard,tau=snapshot['hosts']
            heads,device,budget=(snapshot[k] for k in ('heads','device','budget'))
            table=cu.make_graph_layout(snapshot['boundaries'],args.rows*n,n,n)
            kwargs=dict(layout=VarlenLayout.for_graph(table,args.rows*n))
            flat=[x.reshape(-1,x.shape[-1]) for x in inputs[:4]]
            # This plan uses the exact sampled original-row mask and
            # labels/tau from the same captured forward as these inputs.
            host=native.prepare(labels,tau,hard=hard,documents=snapshot['documents'],
                                compact=True,chunk_size=64)
            payload=upload_plan(host,device=device,stream=stream,budget_bytes=budget)
            descriptor=payload.descriptor.to(device,non_blocking=True)
            choice=torch.ones(1,dtype=torch.int32,device=device)
            ready=torch.ones_like(choice)
            expected=torch.ones_like(choice)
            consumer=GraphCompactBackward(descriptor,choice)
            static=CompactFusedHardPrefillBackward.from_compact(payload.device)
            # The static consumer normally waits for its upload event.
            # Upload is already complete in this GPU-only comparison;
            # exclude that uncaptured external dependency, as in the
            # existing standalone hard-backward benchmark.
            stream.synchronize()
            static.ready=SimpleNamespace(wait=lambda stream:None)
            static.bucket_ready=None
            def event():
                return consumer.backward(*flat,heads=heads)
            def quadratic():
                return backward_hard_recompute(*inputs,**kwargs)
            def dual():
                sample_route[(1,)](ready,expected,choice,num_warps=1)
                a=event()
                b=backward_hard_recompute(*inputs,**kwargs,_graph_choice=choice)
                return {k:a[k].view_as(b[k])+b[k].float() for k in a}
            functions=dict(quadratic=quadratic,event=event,
                           static_event=lambda:static.backward(*flat),dual_event=dual,dual_quadratic=dual)
            probes={f'event_ctas_{c}':GraphCompactBackward(descriptor,choice,ctas=c)
                    for c in args.event_ctas}
            if args.schedule_ablations:
                from graph_event_schedule_probe import ScheduleProbe
                for direct in (False,True):
                    for grid,loop in (('512',True),('tasks',True),('tasks',False)):
                        name=f'probe_{"direct" if direct else "indirect"}_{grid}_{"loop" if loop else "one"}'
                        probes[name]=ScheduleProbe(payload,descriptor,choice,direct=direct,
                                                   loop=loop,task_grid=grid=='tasks')
                probes['probe_indirect_tasks_one_nonempty']=ScheduleProbe(
                    payload,descriptor,choice,loop=False,task_grid=True,skip_empty=True)
            probes['event_no_prefix']=GraphCompactBackward(descriptor,choice,prefix_tasks=False)
            functions.update({name:lambda probe=probe:probe.backward(*flat,heads=heads)
                              for name,probe in probes.items()})
            timings={}
            resources={}
            reference=None
            diagnostics={}
            for mode,fn in functions.items():
                ready.fill_(0 if mode=='dual_quadratic' else 1)
                choice.fill_(1)
                timings[mode],out=measure(fn,args.repeats,args.samples)
                owner=probes.get(mode,static if mode=='static_event' else consumer)
                if mode not in ('quadratic','dual_event','dual_quadratic'):
                    resources[mode]=kernel_resources(owner.last_kernels,kernel_dir)
                if reference is None:
                    reference={k:v.detach().clone() for k,v in out.items()}
                else:
                    diagnostics[mode]={}
                    for key,target in reference.items():
                        a,b=out[key].float().flatten(),target.float().flatten()
                        # Same cross-route relaxed pointwise check used
                        # by the existing model acceptance, never loosened.
                        torch.testing.assert_close(a,b,atol=.001,rtol=.03)
                        diagnostics[mode][key]=dict(max_abs=(a-b).abs().max().item(),
                            relative_l2=((a-b).norm()/b.norm().clamp_min(1e-30)).item(),
                            cosine=torch.nn.functional.cosine_similarity(a,b,dim=0).item())
            row=dict(pack=pack,documents=snapshot['boundaries'].numel()-1,layer=layer,
                     hard_rows=int(hard.sum()),total_rows=hard.numel(),
                     statistics=host.statistics,timings=timings,diagnostics=diagnostics,
                     kernel_resources=resources)
            rows.append(row)
            args.output.write_text(json.dumps(dict(config=vars(args)|{'output':str(args.output)},cases=rows),indent=2)+'\n')
            print(json.dumps(dict(pack=pack,layer=layer,ms={k:v['median_ms'] for k,v in timings.items()})),flush=True)
            del payload,static,consumer,reference,out
    # Each packing occurs twice in a measured eight-microbatch optimizer step.
    estimate={k:2*sum(r['timings'][k]['median_ms'] for r in rows) for k in functions}
    print(json.dumps(dict(estimated_gpu_ms_per_step=estimate,
                         scope='sum of isolated same-input GPU timings; excludes CPU planning/upload; not wall time')),flush=True)


if __name__=='__main__':
    main()
