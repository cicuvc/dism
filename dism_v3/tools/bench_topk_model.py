"""Interleaved scratch-model top-k graph optimizer throughput.

No datasets or checkpoints are accessed. Compare candidate counts, sampled
readout, mixer output variants, or planner off/on at the same K.
Timings include planner draining.
Both models use the same initialization and inputs.
"""
import argparse
import json
import math
from pathlib import Path
import statistics
import time
import torch
from nanochat.models import ModelSpec, build_model, PretrainOptimizerConfig
from nanochat.cuda_graph import PackedTrainingGraph
from nanochat.document_loss import DocumentLossAccumulator
from flash_dism.hard_training import take_route_log


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-spec',required=True)
    parser.add_argument('--warmup',type=int,default=4)
    parser.add_argument('--rounds',type=int,default=12)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--probability',type=float,default=.5)
    parser.add_argument('--packing',choices=['full','changing'],default='changing')
    parser.add_argument('--top1-planner',action='store_true')
    parser.add_argument('--compare-planner',action='store_true',help='Compare planner off/on at the same K')
    parser.add_argument('--compare-output',action='store_true',help='Compare original RMSNorm/SiLU with LayerNorm/sigmoid at the same K')
    parser.add_argument('--compare-fusion',action='store_true',help='Compare explicit/fused implementation of the same sigmoid mixer output')
    parser.add_argument('--disable-output-fusion',action='store_true',help='Use the earlier explicit sigmoid output implementation')
    parser.add_argument('--output-planner',action='store_true',help='Enable planner for both output/fusion comparison models')
    parser.add_argument('--topk',type=int,choices=(1,2,4),default=4)
    parser.add_argument('--planner-workers',type=int,default=16)
    parser.add_argument('--layer-layout',help='Explicit scratch-model layout override')
    parser.add_argument('--compare-sampled',action='store_true',help='Compare full top-four to sampled residual, both planner off')
    parser.add_argument('--single-mode',choices=['top1','top4','sampled'],help='One resident model for externally ordered repetitions')
    parser.add_argument('--rows',type=int,default=8)
    parser.add_argument('--accum',type=int,default=8)
    args=parser.parse_args()
    if args.compare_planner and (args.compare_output or args.compare_fusion or args.compare_sampled or args.single_mode or args.top1_planner):
        parser.error('--compare-planner is a separate paired comparison')
    if args.compare_output and (args.compare_fusion or args.compare_sampled or args.single_mode or args.top1_planner):
        parser.error('--compare-output is a separate paired comparison')
    if args.compare_fusion and (args.compare_sampled or args.single_mode or args.top1_planner):
        parser.error('--compare-fusion is a separate paired comparison')
    if args.compare_fusion and args.disable_output_fusion:
        parser.error('--compare-fusion sets fusion separately for each model')
    if args.output_planner and not (args.compare_output or args.compare_fusion):
        parser.error('--output-planner requires --compare-output or --compare-fusion')
    if args.compare_sampled and args.top1_planner:parser.error('sampled comparison does not use top-one/planner')
    if args.rows<1 or args.accum<1:parser.error('rows and accum must be positive')
    root=Path(__file__).resolve().parents[1]/'build'
    if args.output.exists() or not args.output.resolve().is_relative_to(root):
        parser.error('output must be a fresh path under dism_v3/build')
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('high')
    torch.manual_seed(921)
    spec=ModelSpec.from_dict(json.loads(Path(args.model_spec).read_text()))
    spec.config['hard_prob_fixed']=args.probability
    if args.compare_output and spec.config.get('mixer_gate_scales') is None:
        parser.error('--compare-output requires a spec with calibrated mixer_gate_scales')
    if args.compare_fusion and spec.config.get('mixer_output') not in ('layernorm_sigmoid','rmsnorm_sigmoid'):
        parser.error('--compare-fusion requires a sigmoid mixer spec')
    if args.layer_layout:spec.config['layer_layout']=args.layer_layout
    rows,length,accum=args.rows,spec.config['sequence_len'],args.accum
    batches=[]
    for i in range(accum):
        ids=torch.randint(spec.config['vocab_size'],(rows,length),pin_memory=True)
        labels=torch.randint_like(ids,high=spec.config['vocab_size'],pin_memory=True)
        labels[:,::17]=-1
        parts=[length] if args.packing=='full' else ([length],[256,length-256],[512]*(length//512),[256]*(length//256))[i%4]
        ends=[0]
        for _ in range(rows):
            for n in parts:ends.append(ends[-1]+n)
        batches.append((ids,labels,torch.tensor(ends,dtype=torch.int32,pin_memory=True)))
    ids_gpu=torch.empty_like(batches[0][0],device='cuda')
    labels_gpu=torch.empty_like(ids_gpu)
    cu_gpu=torch.empty(rows*length//256+1,device='cuda',dtype=torch.int32)

    def stage(batch):
        ids,labels,bounds=batch
        ids_gpu.copy_(ids,non_blocking=True)
        labels_gpu.copy_(labels,non_blocking=True)
        cu=cu_gpu[:bounds.numel()]
        cu.copy_(bounds,non_blocking=True)
        return cu

    runners={}
    modes=(('explicit','fused') if args.compare_fusion else
           ('rmsnorm_silu','layernorm_sigmoid') if args.compare_output else
           ('planner_off','planner_on') if args.compare_planner else
           (args.single_mode,) if args.single_mode else (('top4','sampled') if args.compare_sampled else ('top1','top4')))
    for mode in modes:
        torch.manual_seed(1741)
        selected_config={**spec.config,'hard_query_topk':args.topk if (args.compare_planner or args.compare_output or args.compare_fusion) else 1 if mode=='top1' else 4,'hard_query_sampled':mode=='sampled'}
        if args.compare_output:
            selected_config['mixer_output']=mode
            if mode=='rmsnorm_silu':selected_config['mixer_gate_scales']=None
        selected=ModelSpec(spec.architecture,spec.architecture_version,selected_config)
        model=build_model(selected).cuda().train()
        model.init_weights()
        if args.compare_fusion:
            for layer in model.layers:
                if not hasattr(layer.attn.norm,'fused_output'):
                    raise RuntimeError('sigmoid mixer fusion is not installed')
                layer.attn.norm.enable_fused_output=(mode=='fused')
        elif args.disable_output_fusion:
            for layer in model.layers:
                layer.attn.norm.enable_fused_output=False
        model.set_training_step(11444)
        opt=model.setup_pretraining_optimizer(PretrainOptimizerConfig(.0008,.0008,.0008,.0008,.01))
        graph=PackedTrainingGraph(model,ids_gpu,labels_gpu,stage(batches[0]),microbatches=accum,planner=(args.output_planner or mode=='planner_on' or (mode=='top1' and args.top1_planner)),
                                  planner_workers=args.planner_workers)
        runners[mode]=(model,opt,graph)
        print(json.dumps(dict(mode=mode,parameters=sum(p.numel() for p in model.parameters()),model_spec=selected.to_dict())),flush=True)
    print(json.dumps(dict(ready=True,reserved_mib=torch.cuda.memory_reserved()/2**20)),flush=True)

    def step(mode):
        model,opt,graph=runners[mode]
        model.set_training_step(11444)
        counts=DocumentLossAccumulator(accum,'cuda')
        for batch in batches:
            loss,count=graph.forward_backward(ids_gpu,labels_gpu,stage(batch))
            counts.add(loss,count)
        graph.drain()
        loss=counts.finalize(model.parameters())
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.).item()
        if not math.isfinite(norm):raise FloatingPointError(norm)
        opt.step()
        model.zero_grad(set_to_none=True)
        value=loss.item()
        if not math.isfinite(value):raise FloatingPointError(value)
        return value

    for _ in range(args.warmup):
        for mode in runners:step(mode)
    torch.cuda.synchronize()
    times={mode:[] for mode in runners}
    losses={mode:[] for mode in runners}
    routes={mode:[] for mode in runners}
    reasons={mode:[] for mode in runners}
    take_route_log()
    torch.cuda.reset_peak_memory_stats()
    for index in range(args.rounds):
        order=list(runners) if index%2==0 else list(reversed(runners))
        for mode in order:
            start=time.perf_counter()
            loss=step(mode)
            torch.cuda.synchronize()
            elapsed=time.perf_counter()-start
            times[mode].append(elapsed)
            losses[mode].append(loss)
            routes[mode].extend(take_route_log())
            reasons[mode].append(runners[mode][2].last_planner_reasons)
            print(json.dumps(dict(round=index,mode=mode,seconds=elapsed,loss=loss)),flush=True)
    tokens=rows*length*accum
    layer_event_fraction={}
    for mode,decisions in routes.items():
        count=sum(not getattr(layer.attn,'is_pure_mixer',False) for layer in runners[mode][0].layers)
        groups=[decisions[index::count] for index in range(count)] if decisions else []
        layer_event_fraction[mode]=[group.count('e')/len(group) for group in groups]
    report=dict(model_spec=args.model_spec,effective_config=spec.config,
                runner_specs={mode:model.model_spec.to_dict() for mode,(model,_,_) in runners.items()},
                topk=args.topk if (args.compare_planner or args.compare_output or args.compare_fusion) else None,
                compare_fusion=args.compare_fusion,
                disable_output_fusion=args.disable_output_fusion,
                compare_output=args.compare_output,output_planner=args.output_planner,
                compare_planner=args.compare_planner,planner_workers=args.planner_workers,
                peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                routes={k:dict(event=v.count('e'),quadratic=v.count('q')) for k,v in routes.items()},
                layer_event_fraction=layer_event_fraction,
                last_microbatch_reasons=reasons,probability=args.probability,packing=args.packing,rows=rows,accum=accum,top1_planner=args.top1_planner,steps_per_backend=args.rounds,tokens_per_step=tokens,
                seconds=times,losses=losses,
                mean_ms={k:statistics.mean(v)*1000 for k,v in times.items()},
                median_ms={k:statistics.median(v)*1000 for k,v in times.items()},
                tokens_per_second={k:tokens/statistics.mean(v) for k,v in times.items()})
    if not args.single_mode:
        first,second=runners
        report.update(throughput_gain=statistics.mean(times[first])/statistics.mean(times[second])-1,
                      paired_time_ratios=[b/a for a,b in zip(times[first],times[second])])
    if not args.compare_fusion and not args.compare_output and not args.compare_planner and not args.compare_sampled and not args.single_mode:
        report.update(top4_throughput_drop=1-statistics.mean(times['top1'])/statistics.mean(times['top4']),
                      paired_top4_time_ratios=report['paired_time_ratios'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    for _,_,graph in runners.values():graph.close()


if __name__=='__main__':main()
