"""Interleave two scratch-model CUDA Graph training loops on one idle GPU.

The only backend difference is the explicit fused-dQ TMA interpolation option.
Both optimizers advance synthetic models; no checkpoints or datasets are used.
"""
import argparse
from functools import partial
import json
import math
from pathlib import Path
import statistics
import time
import torch
from nanochat.models import ModelSpec, build_model, PretrainOptimizerConfig
from nanochat.cuda_graph import PackedTrainingGraph
from nanochat.document_loss import DocumentLossAccumulator
from flash_dism.kernels import interp_bwd_split


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-spec',required=True)
    parser.add_argument('--warmup',type=int,default=4)
    parser.add_argument('--rounds',type=int,default=12)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]/'build'
    if args.output.exists() or not args.output.resolve().is_relative_to(root):
        parser.error('output must be a fresh path under dism_v3/build')
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('high')
    torch.manual_seed(921)
    spec=ModelSpec.from_dict(json.loads(Path(args.model_spec).read_text()))
    rows,length,accum=8,spec.config['sequence_len'],8
    batches=[]
    for i in range(accum):
        ids=torch.randint(spec.config['vocab_size'],(rows,length),pin_memory=True)
        labels=torch.randint_like(ids,high=spec.config['vocab_size'],pin_memory=True)
        labels[:,::17]=-1
        parts=([length],[256,length-256],[512]*(length//512),[256]*(length//256))[i%4]
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
    original=interp_bwd_split.emb_bwd_split
    for mode in ('split','tma'):
        torch.manual_seed(1741)
        model=build_model(spec).cuda().train()
        model.init_weights()
        model.set_training_step(11444)
        opt=model.setup_pretraining_optimizer(PretrainOptimizerConfig(.0008,.0008,.0008,.0008,.01))
        interp_bwd_split.emb_bwd_split=partial(original,fuse_dq_tma=(mode=='tma'))
        graph=PackedTrainingGraph(model,ids_gpu,labels_gpu,stage(batches[0]),microbatches=accum)
        runners[mode]=(model,opt,graph)
    interp_bwd_split.emb_bwd_split=original
    print(json.dumps(dict(ready=True,reserved_mib=torch.cuda.memory_reserved()/2**20)),flush=True)

    def step(mode):
        model,opt,graph=runners[mode]
        model.set_training_step(11444)
        counts=DocumentLossAccumulator(accum,'cuda')
        for batch in batches:
            loss,count=graph.forward_backward(ids_gpu,labels_gpu,stage(batch))
            counts.add(loss,count)
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
    for index in range(args.rounds):
        order=list(runners) if index%2==0 else list(reversed(runners))
        for mode in order:
            start=time.perf_counter()
            loss=step(mode)
            torch.cuda.synchronize()
            elapsed=time.perf_counter()-start
            times[mode].append(elapsed)
            losses[mode].append(loss)
            print(json.dumps(dict(round=index,mode=mode,seconds=elapsed,loss=loss)),flush=True)
    tokens=rows*length*accum
    report=dict(model_spec=args.model_spec,steps_per_backend=args.rounds,tokens_per_step=tokens,
                seconds=times,losses=losses,
                mean_ms={k:statistics.mean(v)*1000 for k,v in times.items()},
                median_ms={k:statistics.median(v)*1000 for k,v in times.items()},
                tokens_per_second={k:tokens/statistics.mean(v) for k,v in times.items()},
                paired_speedups=[a/b for a,b in zip(times['split'],times['tma'])])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
