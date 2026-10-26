"""CUDA Graph stage timing of full top-four and sampled control kernels."""
import argparse,json
from pathlib import Path
import torch
from flash_dism.kernels import topk_hard as full, sampled_hard as sampled


def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--length',type=int,default=2048);a=p.parse_args()
 torch.set_num_threads(2);torch.manual_seed(81)
 b,n,h,d,r,dv,c=4,a.length,12,64,32,64,512
 q=torch.randn(b,n,h,d,device='cuda',dtype=torch.bfloat16);eq=torch.randn(h,c,d,device='cuda')
 sq=torch.randn(b,n,h,r,device='cuda',dtype=torch.bfloat16);sk=torch.randn_like(sq)
 v=torch.randn(b,n,h,dv,device='cuda',dtype=torch.bfloat16);do=torch.randn_like(v)
 iq=torch.randint(c,(b,h,n),device='cuda',dtype=torch.int32);ik=torch.randint_like(iq,c)
 hard=torch.ones(b,h,device='cuda',dtype=torch.bool);tau=torch.linspace(0,.5,h,device='cuda')
 seg=torch.zeros(b,n,device='cuda',dtype=torch.int32);counter=torch.tensor(37,device='cuda',dtype=torch.int64)
 ids,pi=full.select(q,eq,iq,hard,1.);draw=sampled.sample(pi,counter,1729)
 def measure(fn):
  stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(stream):
   for _ in range(3):fn()
  torch.cuda.current_stream().wait_stream(stream)
  graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph):ret=fn()
  times=[]
  for _ in range(3):
   begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
   begin.record()
   for _ in range(30):graph.replay()
   end.record();end.synchronize();times.append(begin.elapsed_time(end)/30)
  return times
 report={}
 for name,k in [('top4',full),('sampled',sampled)]:
  extra=(draw,) if name=='sampled' else ()
  out,state=k.forward(sq,sk,v,iq,ik,ids,pi,tau,hard,seg,*extra)
  bound,log,dlog,den,lam=state
  if name=='sampled':
   prefix=k.prefix_counts(ik,hard,c)
   norm=lambda:k._normalizers[(n//32,b*h)](iq,ik,seg,hard,bound,ids,draw,log,dlog,den,lam,prefix,n,h,c,num_warps=4,num_stages=1)
   fwd=lambda:k._forward[(n//32,b*h)](sq,sk,v,iq,ik,seg,hard,bound,ids,pi,draw,log,dlog,den,lam,out,n,h,r,dv,num_warps=4,num_stages=1)
  else:
   norm=lambda:k._normalizers[(n//32,b*h)](iq,ik,seg,hard,bound,ids,log,dlog,den,lam,n,h,BK=32,num_warps=4,num_stages=1)
   fwd=lambda:k._forward[(n//32,b*h)](sq,sk,v,iq,ik,seg,hard,bound,ids,pi,log,den,lam,out,n,h,r,dv,num_warps=4,num_stages=1)
  stages=dict(normalizers=norm,readout=fwd,
   backward=lambda:k.backward(sq,sk,v,do,iq,ik,ids,pi,hard,seg,state,*extra),
   full_forward=lambda:k.forward(sq,sk,v,iq,ik,ids,pi,tau,hard,seg,*extra))
  if name=='sampled':stages['prefix_counts']=lambda:k.prefix_counts(ik,hard,c)
  if name=='sampled':
   dsq=torch.zeros_like(sq,dtype=torch.float32);dsk=torch.zeros_like(sk,dtype=torch.float32)
   dvout=torch.zeros_like(v,dtype=torch.float32);dt=torch.zeros_like(tau);u=torch.zeros_like(pi)
   def backward_variant(bk,warps):
    dsq.zero_();dt.zero_();u.zero_()
    k._backward[(n//bk,b*h)](sq,sk,v,do,iq,ik,seg,hard,bound,ids,pi,draw,log,dlog,den,lam,
                            dsq,dsk,dvout,dt,u,n,h,r,dv,BK=bk,num_warps=warps,num_stages=1)
    return u-pi*u.sum(-1,keepdim=True)
   for bk,warps in [(32,8),(64,4),(64,8)]:
    stages[f'backward_k{bk}_w{warps}']=lambda bk=bk,warps=warps:backward_variant(bk,warps)
  report[name]={stage:measure(fn) for stage,fn in stages.items()}
  if name=='sampled':
   report[name]['backward_resources']=[dict(key=str(key),registers=c.n_regs,spills=c.n_spills)
      for key,c in k._backward.device_caches[0][0].items()]
  print(name,report[name],flush=True)
 a.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
