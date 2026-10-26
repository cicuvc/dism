"""Top-k analytic event backward, mixed rows and packed documents."""
import runpy
from pathlib import Path

import numpy as np
import pytest
import torch

from flash_dism.kernels import topk_hard
from flash_dism.kernels.topk_event import TopkEventBackward
from flash_dism.topk_planner import NativeTopkPlanner,upload_topk

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA')


@pytest.mark.parametrize('topk',[1,2,4])
@pytest.mark.parametrize('repeated',[False,True])
def test_event_gradients(topk,repeated):
    torch.manual_seed(551)
    b,n,h,r,d,c=2,512,2,32,64,8
    sq=(torch.randn(b,n,h,r,device='cuda')*.2).bfloat16()
    sk=(torch.randn_like(sq.float())*.2).bfloat16();v=torch.randn(b,n,h,d,device='cuda').bfloat16()
    z=torch.randn(b,h,n,c,device='cuda')
    vals,ids=z.topk(topk,-1);ids=ids.int().contiguous();pi=vals.softmax(-1).contiguous()
    iq=ids[...,0].contiguous();ik=torch.randint(c,(b,h,n),device='cuda',dtype=torch.int32)
    if repeated:iq.zero_();ik.zero_();ids[:]=torch.arange(topk,device='cuda')
    tau=torch.tensor([0.,.1],device='cuda');hard=torch.tensor([[True,False],[True,True]],device='cuda')
    seg=torch.zeros(b,n,device='cuda',dtype=torch.int32);seg[:,256:]=1
    out,state=topk_hard.forward(sq,sk,v,iq,ik,ids,pi,tau,hard,seg)
    do=torch.randn_like(v)
    expected=topk_hard.backward(sq,sk,v,do,iq,ik,ids,pi,hard,seg,state)
    native=NativeTopkPlanner(2)
    docs=np.array([(i,s,s+256) for i in range(b) for s in (0,256)],dtype=np.int64)
    raw=native.prepare(torch.stack((iq,ik)).cpu(),tau.cpu(),ids.cpu(),hard.cpu(),docs,2**28)
    stream=torch.cuda.current_stream()
    payload=upload_topk(raw,device=sq.device,stream=stream,budget=2**29)
    descriptor=payload.descriptor.to(sq.device);choice=torch.ones(1,device=sq.device,dtype=torch.int32)
    engine=TopkEventBackward(descriptor,choice)
    found=engine.backward(sq,sk,v,do,pi)
    for name,x,y in zip(('sq','sk','v','tau','router'),found,expected):
        x=x.double().flatten();y=y.double().flatten()
        if y.norm()<1e-8:assert x.norm()<1e-8;continue
        rel=(x-y).norm()/y.norm();cos=torch.nn.functional.cosine_similarity(x,y,dim=0)
        print(topk,repeated,name,rel.item(),cos.item(),flush=True)
        assert torch.isfinite(x).all() and rel<.025 and cos>.999,(name,rel,cos)
    # Independent FP64 recurrence; hard masks are per original row/head.
    if not repeated:
        oracle=runpy.run_path(str(Path(__file__).parents[1]/'tools/check_topk_training.py'))['oracle']
        inputs=[t.double().requires_grad_() for t in (sq,sk,v,tau,vals)]
        ref=oracle(*inputs[:3],iq,ik,ids,inputs[4],inputs[3],seg)
        ref=ref*hard[:,None,:,None]
        grads=torch.autograd.grad((ref*do).sum(),inputs)
        for name,x,y in zip(('sq','sk','v','tau','router'),found,grads):
            x=x.double().flatten();y=y.double().flatten()
            if y.norm()<1e-8:assert x.norm()<1e-8;continue
            rel=(x-y).norm()/y.norm();cos=torch.nn.functional.cosine_similarity(x,y,dim=0)
            assert torch.isfinite(x).all() and rel<.025 and cos>.999,(name,rel,cos)
    # A stale descriptor must never be dereferenced by either branch.
    choice.zero_();descriptor.zero_()
    gated=engine.backward(sq,sk,v,do,pi)
    for x in gated:assert torch.count_nonzero(x)==0


@pytest.mark.parametrize('topk',[1,2,4])
def test_empty_ready_plan(topk):
    b,n,h=1,256,2
    labels=torch.zeros((2,b,h,n),dtype=torch.int32)
    ids=torch.zeros((b,h,n,topk),dtype=torch.int32)
    # Soft rows intentionally contain duplicate candidate padding: no hard job
    # may inspect or expand those candidates.
    raw=NativeTopkPlanner(2).prepare(labels,torch.zeros(h),ids,torch.zeros(b,h,dtype=torch.bool),
                                    np.array([[0,0,n]],dtype=np.int64),2**20)
    payload=upload_topk(raw,device=torch.device('cuda',0),stream=torch.cuda.current_stream(),budget=2**21)
    engine=TopkEventBackward(payload.descriptor.cuda(),torch.ones(1,device='cuda',dtype=torch.int32))
    sq=torch.randn(b,n,h,32,device='cuda',dtype=torch.bfloat16);sk=torch.randn_like(sq)
    v=torch.randn(b,n,h,64,device='cuda',dtype=torch.bfloat16);do=torch.randn_like(v)
    pi=torch.randn(b,h,n,topk,device='cuda').softmax(-1)
    engine.backward(sq,sk,v,do,pi);torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):got=engine.backward(sq,sk,v,do,pi)
    for _ in range(2):
        for x in got:x.fill_(float('nan'))
        graph.replay();torch.cuda.synchronize()
        for x in got:assert torch.count_nonzero(x)==0
