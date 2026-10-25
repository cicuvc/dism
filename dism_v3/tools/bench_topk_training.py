"""CUDA graph timing of top-four core against native quadratic top-one."""
import argparse,json
from pathlib import Path
import torch
from flash_dism.kernels import topk_hard as top4
from flash_dism.hard_forward import forward_hard
from flash_dism.hard_backward import backward_hard_recompute


def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',type=int,default=4)
    p.add_argument('--length',type=int,default=2048);p.add_argument('--output',required=True)
    a=p.parse_args();torch.set_num_threads(2);torch.manual_seed(7)
    b,n,h,r,d,c=a.batch,a.length,12,32,64,512
    sq=torch.randn(b,n,h,r,device='cuda',dtype=torch.bfloat16)
    sk=torch.randn_like(sq);v=torch.randn(b,n,h,d,device='cuda',dtype=torch.bfloat16);do=torch.randn_like(v)
    logits=torch.randn(b,h,n,c,device='cuda');vals,ids=logits.topk(4,-1);ids=ids.int().contiguous()
    pi=vals.softmax(-1).contiguous();del logits
    iq=ids[:,:,:,0].contiguous();ik=torch.randint(c,(b,h,n),device='cuda',dtype=torch.int32)
    tau=torch.linspace(.1,1.,h,device='cuda');seg=torch.zeros(b,n,device='cuda',dtype=torch.int32)
    rows=[]
    for probability in (.5,1.):
        hard=torch.ones(b,h,device='cuda',dtype=torch.bool)
        if probability==.5:hard[:,1::2]=False
        for mode in ('top1','top4'):
            def step():
                if mode=='top1':
                    out=forward_hard(sq,sk,v,iq,ik,hard,tau)[0]
                    grads=backward_hard_recompute(sq,sk,v,do,iq,ik,hard,tau)
                else:
                    out,state=top4.forward(sq,sk,v,iq,ik,ids,pi,tau,hard,seg)
                    grads=top4.backward(sq,sk,v,do,iq,ik,ids,pi,hard,seg,state)
                return out,grads
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):step()
            torch.cuda.current_stream().wait_stream(stream)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):result=step()
            for _ in range(3):graph.replay()
            times=[]
            for _ in range(5):
                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                begin.record()
                for _ in range(20):graph.replay()
                end.record();end.synchronize();times.append(begin.elapsed_time(end)/20)
            row=dict(mode=mode,hard_probability=probability,ms=sorted(times)[2],samples=times)
            rows.append(row);print(json.dumps(row),flush=True)
            del result,graph
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(dict(config=vars(a),results=rows),indent=2)+'\n')


if __name__=='__main__':main()
