"""Standalone fixed-length hard CUDA backward. Never loads training state."""
import argparse
import json
import time

import numpy as np
import torch

from bench_training import measure, profile_kernels, gpu_status
from flash_dism.hard_forward import forward_hard
from flash_dism.hard_backward import backward_hard
from flash_dism.summary import summarize_hard, hard_chunk_scan


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--batch',type=int,default=16)
    parser.add_argument('--tokens',type=int,default=2048)
    parser.add_argument('--patterns',nargs='+',default=['random','equal'])
    parser.add_argument('--profile',action='store_true')
    parser.add_argument('--events',action='store_true')
    parser.add_argument('--soft-reference',action='store_true')
    parser.add_argument('--ctas',type=int,default=0)
    args=parser.parse_args()
    b,n,h,r,dv=args.batch,args.tokens,6,32,64
    torch.manual_seed(417)
    rows=[]
    before=gpu_status()
    for pattern in args.patterns:
        sq,sk=[(torch.randn(b,n,h,r,device='cuda')*.3).bfloat16() for _ in range(2)]
        v,do=[torch.randn(b,n,h,dv,device='cuda').bfloat16() for _ in range(2)]
        iq,ik=[torch.randint(512,(b,h,n),device='cuda',dtype=torch.int32) for _ in range(2)]
        if pattern=='equal': iq.zero_(); ik.zero_()
        hard=torch.ones(b,h,device='cuda',dtype=torch.bool)
        tau=torch.full((h,),.5,device='cuda')
        _,norm,state=forward_hard(sq,sk,v,iq,ik,hard,tau,save_state=True)
        def backward():
            return backward_hard(sq,sk,v,do,iq,ik,hard,norm,state,ctas=args.ctas)
        def recompute_boundary_backward():
            summary=summarize_hard(iq,ik,hard)
            temporary=dict(boundary=hard_chunk_scan(summary),log_weights=state['log_weights'])
            return backward_hard(sq,sk,v,do,iq,ik,hard,norm,temporary,ctas=args.ctas)
        actual=backward()
        torch.cuda.synchronize()
        row=dict(pattern=pattern,backward=measure(backward,5,10,5),
                 recompute_boundary_backward=measure(recompute_boundary_backward,5,10,5),
                 boundary_bytes=state['boundary'].numel()*4,dtau=actual[3].tolist())
        if args.profile: row['profile']=profile_kernels(backward,iterations=10)
        if args.soft_reference:
            import cu_flash_dism as cu
            qk=torch.zeros(b,n,h,64,device='cuda',dtype=torch.bfloat16)
            lse=torch.zeros(b,n,h,device='cuda')
            direction=torch.ones(b,h,device='cuda',dtype=torch.bool)
            raw=(qk,qk,sq,sk,v,lse,lse,iq,ik,direction,torch.zeros_like(hard),tau)
            output,soft_norm,soft_state=cu.core_forward(raw,None,0,True,False,True)
            backend=cu.get_config(r,64,dv)
            delta=backend.backward_delta(output,do)
            fn=lambda: backend.backward_summary(soft_state['operands'],soft_state['vertical'],
                                                 do,soft_norm,delta,n,0,False)
            row['soft_b1_timing_only']=measure(fn,5,10,5)
            row['soft_full_timing_only']=measure(lambda: cu.core_backward(soft_state['operands'],
                soft_state['vertical'],output,soft_norm,do,None,0,False,False),5,10,5)
        if args.events:
            from flash_dism.inference.training_planner import NativeTrainingPlanner
            from flash_dism.kernels.hard_backward_scan import HardPrefillBackward
            from flash_dism.kernels.hard_backward_fused import FusedHardPrefillBackward
            labels=np.stack([iq.cpu().numpy(),ik.cpu().numpy()])
            start=time.perf_counter()
            host=NativeTrainingPlanner(16).prepare(labels,tau.cpu().numpy().astype(np.float64),chunk_size=64)
            row['planner_ms']=(time.perf_counter()-start)*1e3
            device=host.upload(); device.wait()
            base=HardPrefillBackward.from_native(device,mma_precision='bf16')
            plan=FusedHardPrefillBackward.from_plan(base,host.data['offsets'].numpy(),host_lengths=host.data['lengths'].numpy())
            flat=[x.reshape(-1,x.shape[-1]) for x in (sq,sk,v,do)]
            event_fn=lambda: plan.backward(*flat)
            expected=event_fn()
            row['events_gpu']=measure(event_fn,5,10,5)
            row['event_dtau']=expected['rtau'].tolist()
            row['relative_l2_vs_event']={name:float((a.float().flatten()-expected[name].float().flatten()).norm()/
                                                        expected[name].float().norm().clamp_min(1e-8))
                for name,a in zip(('sq_vec','sk_vec','v','rtau'),actual)}
        rows.append(row)
    print(json.dumps(dict(shape=dict(b=b,n=n,h=h,r=r,dv=dv),ctas=args.ctas,
                         gpu_before=before,gpu_after=gpu_status(),cases=rows,
                         scope='fixed rows; allocation/zero/LUT/reduction included; no production end-to-end claim'),indent=2))


if __name__=='__main__': main()
