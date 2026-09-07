"""Production delta preprocessing and independent natural-log gradient contract."""
import itertools
from pathlib import Path
import re
import subprocess
import pytest
import torch
from torch.utils.cpp_extension import CUDA_HOME
from dism_v2.backward import delta,_extension


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")
@pytest.mark.parametrize("dv",(32,64,128))
@pytest.mark.parametrize("n",(1,7,8,9,65,139,513))
def test_delta(dv,n):
    if torch.cuda.get_device_capability()!=(12,0): pytest.skip("sm120a only")
    g=torch.Generator(device="cuda").manual_seed(101+dv+n)
    out=torch.randn((2,2,n,dv),device="cuda",dtype=torch.bfloat16,generator=g)
    dout=torch.randn(out.shape,device="cuda",dtype=torch.bfloat16,generator=g)
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    before=torch.cuda.get_rng_state()
    with torch.cuda.stream(stream): result=delta(dout,out)
    torch.cuda.current_stream().wait_stream(stream)
    assert result.dtype==torch.float32 and result.shape==out.shape[:-1]
    torch.testing.assert_close(result.double(),(dout.double()*out.double()).sum(-1),atol=2e-5,rtol=2e-6)
    assert torch.equal(before,torch.cuda.get_rng_state())
    torch.testing.assert_close(delta(torch.zeros_like(dout),out),torch.zeros_like(result),atol=0,rtol=0)
    with pytest.raises(RuntimeError,match="BF16"): delta(dout.float(),out)
    with pytest.raises(RuntimeError,match="contiguous"): delta(dout.transpose(0,1),out)


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")
def test_backward_codegen():
    if torch.cuda.get_device_capability()!=(12,0): pytest.skip("sm120a only")
    tool=str(Path(CUDA_HOME)/"bin/cuobjdump")
    sass=subprocess.check_output([tool,"--dump-sass",_extension().__file__],text=True)
    assert "UTMALDG.5D" in sass
    assert not re.search(r"\b(?:CALL|LDL|STL)(?:\.|\s)",sass)
    assert not re.search(r"\b(?:ATOM|RED)(?:\.|\s)",sass)
    resources=subprocess.check_output([tool,"--dump-resource-usage",_extension().__file__],text=True)
    sizes=re.findall(r"(?:STACK|LOCAL):(\d+)",resources)
    assert sizes and all(int(x)==0 for x in sizes)


@pytest.mark.parametrize("d,dv",itertools.product((32,64,128),repeat=2))
@pytest.mark.parametrize("column_lse",(False,True))
@pytest.mark.parametrize("mode",("soft","mixed","hard"))
def test_natural_gradient_contract(d,dv,column_lse,mode):
    # CPU FP64 test: differentiate the reference recurrence, without replacing
    # -inf breaks by finite sentinels or passing through undefined log(0).
    torch.manual_seed(303+d+dv)
    h,n=2,9
    a=torch.randn(h,n,d,dtype=torch.float64,requires_grad=True)
    b=torch.randn_like(a,requires_grad=True)
    v=torch.randn(h,n,dv,dtype=torch.float64,requires_grad=True)
    lse=torch.full((h,n),3.,dtype=torch.float64,requires_grad=True)
    tau=torch.tensor([-.3,.4],dtype=torch.float64,requires_grad=True)
    hard=torch.zeros(h,n,dtype=torch.bool)
    if mode=="mixed": hard[:,::3]=True
    if mode=="hard": hard[:]=True
    labels=torch.arange(n)%3
    match=labels[:,None]==labels[None,:]
    soft=a@b.transpose(-1,-2)*d**-.5-lse.unsqueeze(-2 if column_lse else -1)+tau[:,None,None]
    m=torch.where(hard[:,:,None],torch.where(match,tau[:,None,None],-torch.inf),soft)
    m=m.masked_fill(torch.arange(n)[None,:]>torch.arange(n)[:,None],-torch.inf)
    rows=[]
    previous=torch.full((h,n),-torch.inf,dtype=torch.float64)
    for i in range(n):
        shifted=torch.nn.functional.pad(previous[:,:-1],(1,0),value=-torch.inf)
        previous=m[:,i,:]+torch.nn.functional.softplus(shifted)
        rows.append(previous)
    w=torch.stack(rows,dim=1)
    p=torch.softmax(torch.cat((torch.zeros(h,n,1,dtype=torch.float64),w),-1),-1)[...,1:]
    out=p@v
    dout=torch.randn_like(out)
    expected=torch.autograd.grad((out*dout).sum(),(a,b,v,lse,tau),allow_unused=True)
    with torch.no_grad():
        e=p*((dout@v.transpose(-1,-2))-(dout*out).sum(-1,keepdim=True))
        gradient=torch.zeros_like(e)
        for i in range(n-1,-1,-1):
            successor=torch.nn.functional.pad(gradient[:,i+1,1:],(0,1)) if i+1<n else torch.zeros(h,n,dtype=torch.float64)
            gradient[:,i]=e[:,i]+torch.sigmoid(w[:,i])*successor
        soft_gradient=gradient*(~hard[:,:,None])
        actual=(soft_gradient@b*d**-.5,soft_gradient.transpose(-1,-2)@a*d**-.5,
            p.transpose(-1,-2)@dout,-soft_gradient.sum(-2 if column_lse else -1),gradient.sum((-2,-1)))
        for x,y,operand in zip(actual,expected,(a,b,v,lse,tau)):
            torch.testing.assert_close(x,torch.zeros_like(operand) if y is None else y,atol=2e-12,rtol=2e-12)
