"""Compare runtime versus static-true varlen phases; no model/checkpoint writes."""
import argparse
import json
import statistics
from pathlib import Path
import torch
import cu_flash_dism as cu
from flash_dism.varlen import VarlenLayout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tokens', type=int, default=32768)
    ap.add_argument('--length', type=int, default=2048)
    ap.add_argument('--repeats', type=int, default=20)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    assert args.tokens % args.length == 0 and args.length % 256 == 0
    torch.manual_seed(813)
    n,h,d,r = args.tokens,6,64,32
    layout = VarlenLayout.from_cu_seqlens(torch.arange(0,n+1,args.length,dtype=torch.int32),n)
    vectors = [(torch.randn(1,n,h,c,device='cuda')*.2).bfloat16() for c in (d,d,r,r,d)]
    lse = [torch.rand(1,n,h,device='cuda')+5 for _ in range(2)]
    labels = [torch.randint(512,(1,h,n),device='cuda') for _ in range(2)]
    raw = (*vectors,*lse,*labels,torch.ones(1,h,device='cuda',dtype=torch.bool),
           torch.zeros(1,h,device='cuda',dtype=torch.bool),torch.ones(h,device='cuda'))
    operands = list(cu.prepare_operands(raw,True))
    a,b = cu.varlen_summary(operands,layout.table)
    boundary = cu.varlen_chunk(a,b,layout.table,h)
    operands.append(boundary)
    out,norm,vertical = cu.varlen_forward(operands,layout.table,True,False)
    do = torch.randn_like(out)
    delta = cu.varlen_delta(out,do,layout.table)
    bargs = (operands,vertical,do.squeeze(0),norm,delta)
    first = cu.varlen_backward_summary(*bargs,layout.table,False)
    reverse = cu.varlen_backward_chunk(first[3],first[4],layout.table,h)
    functions = {
        'F1': lambda flag: cu.varlen_summary(operands,layout.table,flag),
        'F3': lambda flag: cu.varlen_forward(operands,layout.table,True,False,flag),
        'B1': lambda flag: cu.varlen_backward_summary(*bargs,layout.table,False,flag),
        'B3': lambda flag: cu.varlen_backward_qk(*bargs,reverse,layout.table,False,flag),
    }
    report = {'tokens':n,'document_length':args.length,'heads':h,'R':r,'D':d,'DV':d,
              'gpu':torch.cuda.get_device_name(),'phases':{}}
    for name, fn in functions.items():
        for _ in range(3):
            fn(False); fn(True)
        times = {False:[],True:[]}
        for i in range(args.repeats):
            for flag in ((False,True) if i%2 == 0 else (True,False)):
                torch.cuda.synchronize()
                start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record()
                result = fn(flag)
                end.record(); end.synchronize()
                times[flag].append(start.elapsed_time(end))
                del result
        row = {str(k):statistics.median(v) for k,v in times.items()}
        report['phases'][name] = row
        print(name,row,flush=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
