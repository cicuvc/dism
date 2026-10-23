"""CUDA Graph A/B for final RMSNorm + chunked CE, including backward.

Synthetic resident inputs; no checkpoint or dataset access. The baseline uses
PyTorch RMSNorm and the CE's existing direct quantizer, with identical weights.
"""
import argparse
import json
import statistics
from pathlib import Path
import torch
from flash_mxfp8 import RMSNormMXFP8, MXFP8FusedLinearCrossEntropy
from flash_mxfp8.tk_silu_bwd import SiluBwdQuantizeLibrary


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--rows',type=int,default=16384)
    ap.add_argument('--width',type=int,default=768)
    ap.add_argument('--vocab',type=int,default=32000)
    ap.add_argument('--chunk-size',type=int,default=1024)
    ap.add_argument('--repeats',type=int,default=5)
    ap.add_argument('--iterations',type=int,default=50)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();root=Path(__file__).resolve().parents[1]
    if args.output.exists() or not args.output.resolve().is_relative_to(root/'build'):
        ap.error('fresh output under this worktree dism_v3/build required')
    torch.set_num_threads(1);torch.manual_seed(53)
    x=torch.randn(args.rows,args.width,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    factor=torch.ones(args.width,device='cuda',requires_grad=True)
    dy=torch.randn_like(x)
    weight=(torch.randn(args.vocab,args.width,device='cuda')*.02).bfloat16().requires_grad_()
    targets=torch.randint(args.vocab,(args.rows,),device='cuda')
    targets[::17]=-100
    norm=RMSNormMXFP8(chunk_size=args.chunk_size)
    ce={mode:MXFP8FusedLinearCrossEntropy(chunk_size=args.chunk_size,softcap=30.,balance='token')
        for mode in ('separate','fused')}
    quantizers=[SiluBwdQuantizeLibrary().plan_direct(x[start:start+args.chunk_size])
                for start in range(0,args.rows,args.chunk_size)]
    def operation(mode,part):
        for p in (x,factor,weight):p.grad=None
        # Match DismLM: the final norm runs inside BF16 autocast (RMSNorm
        # itself has FP32 autocast policy), then CE explicitly disables AMP.
        with torch.autocast('cuda',dtype=torch.bfloat16):
            if mode=='separate':
                y=torch.nn.functional.rms_norm(x,(args.width,),factor,1e-6).bfloat16();qs=None
            else:y,qs=norm(x,factor)
        if part in ('norm_quant','norm_quant_backward'):
            if mode=='separate':
                for start,q in zip(range(0,args.rows,args.chunk_size),quantizers):
                    q.rebind(y[start:start+args.chunk_size]);q();q.x=None
            if part=='norm_quant_backward':y.backward(dy)
            return y
        with torch.autocast('cuda',enabled=False):
            loss=ce[mode](y,targets,weight,quantized_input=qs)
        loss.backward()
        return loss
    def call(mode,part):
        with torch.set_grad_enabled(part != 'norm_quant'):
            return operation(mode,part)
    graphs={}
    parts=('norm_quant','norm_quant_backward','norm_ce_backward')
    for part in parts:
        for mode in ('separate','fused'):
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(4):call(mode,part)
            torch.cuda.current_stream().wait_stream(stream)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):call(mode,part)
            graphs[part,mode]=graph
    samples=[]
    for part in parts:
        for repeat in range(args.repeats):
            for mode in (('separate','fused') if repeat%2==0 else ('fused','separate')):
                graph=graphs[part,mode]
                for _ in range(3):graph.replay()
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(args.iterations):graph.replay()
                end.record();end.synchronize()
                samples.append(dict(part=part,mode=mode,repeat=repeat,ms=start.elapsed_time(end)/args.iterations))
        medians={mode:statistics.median(s['ms'] for s in samples if s['part']==part and s['mode']==mode)
                 for mode in ('separate','fused')}
        print(part,medians,flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},samples=samples),indent=2)+'\n')

if __name__=='__main__':main()
