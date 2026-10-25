"""Bounded synthetic optimizer steps for estimating the 3B-token run.

Uses a newly initialized model, no corpus, network, checkpoint or background job.
Includes document loss accumulation, clipping, MXFP8 AdamW and enabled RMS probes.
"""
import argparse
import json
import math
from pathlib import Path
import statistics
import time

import torch
from nanochat.models import ModelSpec,PretrainOptimizerConfig,build_model
from nanochat.document_loss import DocumentLossAccumulator
from nanochat.metrics import register_residual_rms_probe,snapshot_parameters,update_rms_max_per_group


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='nanochat/configs/dism_v4_123m_ggggggdddddd_slimpajama_3b_rmsnorm_h05.json')
    p.add_argument('--batch',type=int,default=8)
    p.add_argument('--accum',type=int,default=8)
    p.add_argument('--steps',type=int,default=5)
    p.add_argument('--warmup',type=int,default=2)
    p.add_argument('--precision',choices=['high','highest'],default='high')
    p.add_argument('--output',required=True)
    args=p.parse_args()
    torch.set_num_threads(1);torch.set_float32_matmul_precision(args.precision);torch.manual_seed(42)
    spec=ModelSpec.from_dict(json.loads(Path(args.config).read_text()))
    model=build_model(spec).cuda();model.init_weights();model.train()
    opt=model.setup_pretraining_optimizer(PretrainOptimizerConfig(.0008,.0008,.0008,.0008,.01))
    handles,buffer=register_residual_rms_probe(model)
    n=model.config.sequence_len
    # Fixed synthetic batches, but distinct tokens across accumulated micros.
    x=torch.randint(model.config.vocab_size,(args.accum,args.batch,n),device='cuda')
    y=torch.randint(model.config.vocab_size,x.shape,device='cuda')
    cu=torch.arange(0,args.batch*n+1,1024,device='cuda',dtype=torch.int32)
    print(json.dumps(dict(parameters=sum(p.numel() for p in model.parameters()),
                          spec=spec.to_dict(),args=vars(args))),flush=True)
    def step(index):
        model.set_training_step(index)
        opt.zero_grad(set_to_none=True)
        doc=DocumentLossAccumulator(args.accum,torch.device('cuda'))
        for j in range(args.accum):
            loss,count=model(x[j],y[j],cu_seqlens=cu,loss_reduction='document_sum')
            doc.add(loss,count)
            (loss/args.accum).backward()
        loss=doc.finalize(model.parameters())
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        assert bool(torch.isfinite(norm))
        before=snapshot_parameters(model)
        opt.step()
        updates=update_rms_max_per_group(model,before)
        assert all(math.isfinite(value) for value in updates.values())
        opt.zero_grad(set_to_none=True)
        return float(loss),float(norm)
    rows=[]
    try:
        for i in range(args.warmup+args.steps):
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
            loss,norm=step(i)
            torch.cuda.synchronize();elapsed=time.perf_counter()-start
            row=dict(step=i,warmup=i<args.warmup,seconds=elapsed,loss=loss,grad_norm=norm,
                     tokens_per_second=args.batch*args.accum*n/elapsed,
                     peak_allocated_bytes=torch.cuda.max_memory_allocated())
            print(json.dumps(row),flush=True)
            if i>=args.warmup:rows.append(row)
    finally:
        for handle in handles:handle.remove()
    seconds=statistics.median([r['seconds'] for r in rows])
    result=dict(args=vars(args),spec=spec.to_dict(),results=rows,median_seconds=seconds,
                tokens_per_second=args.batch*args.accum*n/seconds,
                hours_for_3b=3e9/(args.batch*args.accum*n/seconds)/3600,
                parameter_count=sum(p.numel() for p in model.parameters()))
    print(json.dumps({k:v for k,v in result.items() if k not in ('spec','results','args')}),flush=True)
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
