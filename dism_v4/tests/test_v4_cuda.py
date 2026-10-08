"""CUDA v4 against independent FP64 CPU forward/backward, including row gates."""
import pytest
import torch
from flash_dism.forward import forward_core
from flash_dism.backward import backward_core,dism_core
from flash_dism.varlen import forward_varlen,backward_varlen,dism_core_varlen
from flash_dism.reference.dism_v4_ref import dism_ref,dism_ref_backward
from gradient_acceptance import assert_gradient,GradientTolerance

torch.set_num_threads(1)

def sample(n=256,mode='mixed',gate='vary',heads=2,d=64,r=32,dv=64):
    gen=torch.Generator().manual_seed(551+n)
    def rand(d):return (.2*torch.randn(1,n,heads,d,generator=gen)).bfloat16()
    q,k=rand(d),rand(d);sq,sk=rand(r),rand(r);v=rand(dv)
    lq,lk=[torch.full((1,n,heads),4.) for _ in range(2)]
    iq,ik=[torch.randint(4,(1,heads,n),generator=gen) for _ in range(2)]
    direction=(torch.arange(heads)%2==0).unsqueeze(0)
    hard=torch.rand(1,heads,n,generator=gen)<{'soft':0.,'mixed':.5,'hard':1.}[mode]
    tau=torch.linspace(.1,2.,heads)
    delta=torch.rand(1,heads,n,generator=gen)*2
    if gate=='zero':delta.zero_()
    elif gate=='reset':delta.fill_(torch.inf)
    elif gate=='boundaries':
        delta[:,:,::16]=torch.inf
        delta[:,:,1::32]=0.
    return [q,k,sq,sk,v,lq,lk,iq,ik,direction,hard,tau],delta


def oracle(x,delta,upstream):
    q,k,sq,sk,v,lq,lk,iq,ik,direction,hard,tau=[a.double() if a.is_floating_point() else a for a in x]
    args=(q,k,sq,sk,lq,lk,iq,ik,direction,hard,delta.double(),v,tau)
    return dism_ref(*args),dism_ref_backward(*args,upstream.double())


@pytest.mark.parametrize('mode',['soft','mixed','hard'])
@pytest.mark.parametrize('gate',['zero','vary','reset','boundaries'])
@pytest.mark.parametrize('packed',[False,True])
def test_core_oracle(mode,gate,packed):
    x,delta=sample(mode=mode,gate=gate)
    gpu=[a.cuda() for a in x]
    fn=forward_varlen if packed else forward_core
    opts={'cu_seqlens':torch.tensor([0,256],device='cuda',dtype=torch.int32)} if packed else {'ctas':1}
    out,_,state=fn(*gpu,gate_delta=delta.cuda(),save_state=True,fp32_output=True,**opts)
    torch.manual_seed(661);do=torch.randn_like(x[4])
    grad=(backward_varlen if packed else backward_core)(state,do.cuda(),fp32_output=True)
    expected,reference=oracle(x,delta,do)
    torch.testing.assert_close(out.cpu().double(),expected,atol=.003,rtol=.03)
    for name,a in grad.items():
        key='delta' if name=='gate_delta' else name
        assert_gradient(a.cpu(),reference[key],name)
    if gate=='zero':
        old=fn(*gpu,save_state=True,fp32_output=True,**opts)[0]
        torch.testing.assert_close(out,old,atol=0,rtol=0)
    if gate=='reset':assert torch.count_nonzero(grad['gate_delta'])==0


def sliced(x,start,length):
    return [a[:,start:start+length].contiguous() if i<7 else
            a[:,:,start:start+length].contiguous() if i in (7,8,10) else a for i,a in enumerate(x)]


@pytest.mark.parametrize('lengths',[[256,512,0,256],[2048],[0,0]])
@pytest.mark.parametrize('mode',['soft','mixed','hard'])
def test_packed_boundaries_and_autograd(lengths,mode):
    n=sum(lengths);x,delta=sample(n,mode,'boundaries')
    indices=(0,1,2,3,4,5,6,11)
    gpu=[a.cuda().requires_grad_(i in indices) for i,a in enumerate(x)]
    dg=delta.cuda().requires_grad_()
    cu=torch.tensor([0,*torch.tensor(lengths).cumsum(0).tolist()],device='cuda',dtype=torch.int32)
    actual=dism_core_varlen(*gpu,gate_delta=dg,cu_seqlens=cu)
    torch.manual_seed(62);do=torch.randn_like(x[4])
    gradients=torch.autograd.grad(actual,[gpu[i] for i in indices]+[dg],do.cuda())
    expected=[];refs={};start=0
    for length in lengths:
        if length:
            out,ref=oracle(sliced(x,start,length),delta[:,:,start:start+length],do[:,start:start+length])
            expected.append(out)
            for key,value in ref.items():refs.setdefault(key,[]).append(value)
        start+=length
    names=('q_vec','k_vec','sq_vec','sk_vec','v','q_lse','k_lse','rtau','delta')
    if n==0:
        assert actual.numel()==0
        for grad in gradients:assert torch.count_nonzero(grad)==0
        return
    expected=torch.cat(expected,1)
    torch.testing.assert_close(actual.cpu().double(),expected,atol=.003,rtol=.03)
    for name,a in zip(names,gradients):
        values=refs[name]
        ref=sum(values) if name=='rtau' else torch.cat(values,2 if name=='delta' else 1)
        assert_gradient(a.cpu(),ref,name)


def readout_rounding_gate_effect(delta, output_error, upstream):
    """Exact linear effect of saved-output rounding for all-match, tau=2.

    This independent CPU recurrence isolates the existing BF16 readout error
    without weakening the tolerance on the CUDA reverse scan itself.
    """
    delta=delta.double().cpu();n=delta.shape[-1]
    previous=torch.full_like(delta,-torch.inf);rows=[];alphas=[]
    for i in range(n):
        shifted=torch.nn.functional.pad(previous[...,:-1],(1,0),value=-torch.inf)-delta[:,:,i,None]
        alphas.append(torch.where(shifted>20,1.,shifted.sigmoid()))
        previous=2.+torch.nn.functional.softplus(shifted);rows.append(previous)
    w=torch.stack(rows,-2).masked_fill(~torch.ones(n,n,dtype=torch.bool).tril(),-torch.inf)
    maximum=w.amax(-1,keepdim=True).clamp_min(0.)
    ew=(w-maximum).exp();p=ew/(ew.sum(-1,keepdim=True)+(-maximum).exp())
    correction=(output_error.double()*upstream.double().cpu()).sum(-1).transpose(1,2)
    total=-p*correction.unsqueeze(-1);effect=torch.zeros_like(delta)
    for i in range(n-1,-1,-1):
        propagated=total[:,:,i]*alphas[i]
        effect[:,:,i]=-propagated.sum(-1)
        if i:total[:,:,i-1,:-1]+=propagated[...,1:]
    return effect,total.sum((0,2,3))


@pytest.mark.parametrize('n',[512,2048])
def test_persistent_tasks_and_contiguous_matches(n):
    x,delta=sample(n,'hard','vary')
    # Long positive recurrences amplify missing gates and row-index mistakes.
    x[7].zero_();x[8].zero_();x[-1].fill_(2.)
    delta[:,:,::32]=torch.inf
    delta[:,:,1::32]=0.
    gpu=[torch.cat([a,a],0).cuda() if i!=11 else a.cuda() for i,a in enumerate(x)]
    dg=torch.cat([delta,delta*.25],0).cuda()
    out,_,state=forward_core(*gpu,gate_delta=dg,save_state=True,fp32_output=True,ctas=1)
    torch.manual_seed(951);do=torch.randn_like(out).bfloat16()
    gradients=backward_core(state,do,fp32_output=True,ctas=1)
    outputs=[];refs=[]
    for b in range(2):
        o,g=oracle(x,dg[b:b+1].cpu(),do[b:b+1].cpu());outputs.append(o);refs.append(g)
    torch.testing.assert_close(out.cpu().double(),torch.cat(outputs,0),atol=.003,rtol=.03)
    rounding=readout_rounding_gate_effect(dg,out.cpu().double()-torch.cat(outputs,0),do)
    for name,a in gradients.items():
        key='delta' if name=='gate_delta' else name
        ref=refs[0][key]+refs[1][key] if key=='rtau' else torch.cat([z[key] for z in refs],0)
        if key=='delta':
            effect=rounding[0]
            # Account explicitly for the saved-output term; require the usual
            # strict reverse-scan accuracy and cap the independently measured
            # rounding contribution (5% of the ideal gradient norm).
            assert effect.norm() <= .05*ref.norm()
            assert_gradient(a.cpu(),ref+effect,name)
        elif key=='rtau':
            assert_gradient(a.cpu(),ref+rounding[1],name)
        else:
            assert_gradient(a.cpu(),ref,name)


def test_decode_chunks():
    from flash_dism.reference.dism_decode_ref import dism_decode_ref
    x,delta=sample(33,'mixed','boundaries')
    torch.manual_seed(17);do=torch.randn_like(x[4]);expected,_=oracle(x,delta,do)
    parts=[];cache=None;offset=0
    for length in (1,7,25):
        q,k,sq,sk,v,lq,lk,iq,ik,direction,hard,tau=[a.double() if a.is_floating_point() else a for a in sliced(x,offset,length)]
        out,cache=dism_decode_ref(q,k,sq,sk,lq,lk,iq,ik,direction,hard,v,tau,
            cache=cache,gate_delta=delta[:,:,offset:offset+length].double())
        parts.append(out);offset+=length
    torch.testing.assert_close(torch.cat(parts,1),expected,atol=1e-12,rtol=1e-12)


@pytest.mark.parametrize('r',[16,32])
@pytest.mark.parametrize('d',[32,64])
@pytest.mark.parametrize('dv',[32,64])
@pytest.mark.parametrize('packed',[False,True])
def test_all_dimensions(r,d,dv,packed):
    x,delta=sample(gate='boundaries',d=d,r=r,dv=dv)
    fn=forward_varlen if packed else forward_core
    opts={'cu_seqlens':torch.tensor([0,256],device='cuda',dtype=torch.int32)} if packed else {'ctas':1}
    out,_,state=fn(*[a.cuda() for a in x],gate_delta=delta.cuda(),save_state=True,fp32_output=True,**opts)
    torch.manual_seed(831);do=torch.randn_like(x[4])
    gradients=(backward_varlen if packed else backward_core)(state,do.cuda(),fp32_output=True)
    expected,reference=oracle(x,delta,do)
    torch.testing.assert_close(out.cpu().double(),expected,atol=.003,rtol=.03)
    for name,gradient in gradients.items():assert_gradient(gradient.cpu(),reference['delta' if name=='gate_delta' else name],name)
