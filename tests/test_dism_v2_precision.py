"""Precision regression: same-input core, unquantized interpolation, long chains.

Run with --junitxml=... to retain per-case maximum absolute error/RMSE/relative
L2 in the precision property. No production code or oracle semantics modified.
"""
import itertools
import json
import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from dism_v2.core import forward, forward_interpolated
from dism_v2.dism_ref import interpolation_ref, voc_dism_ref

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def row_mask(state):
    # Independent CPU integer implementation, not the device helper.
    result = []
    for row in range(math.prod(state.shape)):
        mask = 2**32-1
        x,y,z,w = (state.offset//4)&mask, state.offset//2**34, row&mask, row>>32
        k0,k1 = state.seed&mask, state.seed>>32
        for _ in range(10):
            p,q = 0xD2511F53*x, 0xCD9E8D57*z
            x,y,z,w = (q>>32)^y^k0, q&mask, (p>>32)^w^k1, p&mask
            k0,k1 = (k0+0x9E3779B9)&mask, (k1+0xBB67AE85)&mask
        p32 = np.float32(state.hard_prob)
        result.append((x>>8)*2**-24 < p32)
    return torch.tensor(result,device="cuda").reshape(*state.shape,1)


def metrics(actual, expected, *, rowwise=False):
    a,b = actual.double(),expected.double()
    assert torch.equal(torch.isneginf(a),torch.isneginf(b))
    assert not torch.isnan(a).any() and not torch.isposinf(a).any()
    finite = torch.isfinite(b)
    delta = a[finite]-b[finite]
    if not delta.numel():
        return dict(max_abs=0.,rmse=0.,relative_l2=0.)
    result = dict(max_abs=delta.abs().max().item(),rmse=delta.square().mean().sqrt().item(),
                  relative_l2=(delta.norm()/b[finite].norm().clamp_min(1e-30)).item())
    if rowwise:
        assert finite.all() and torch.isfinite(a).all()
        def cosine(x,y):
            nx,ny = x.norm(dim=-1),y.norm(dim=-1)
            # Both zero outputs match exactly. One zero output has no direction
            # agreement. Do not let a fixed epsilon distort small finite rows.
            denominator = nx*ny
            similarity = (x*y).sum(-1)/torch.where(denominator>0,denominator,torch.ones_like(denominator))
            return torch.where((nx==0)&(ny==0),torch.ones_like(similarity),similarity).clamp(-1,1)
        rows = cosine(a,b).flatten()
        result.update(cosine=cosine(a.flatten(),b.flatten()).item(),
                      cosine_row_mean=rows.mean().item(),cosine_row_min=rows.min().item(),
                      cosine_row_p01=torch.quantile(rows,.01).item())
    return result


@pytest.fixture(autouse=True)
def full_precision_torch():
    # The oracle must not accidentally use TF32 when a caller enables it.
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = old


def test_cosine_metrics():
    a = torch.tensor([[1.,0.],[0.,0.],[0.,1.]])
    b = torch.tensor([[1.,0.],[0.,0.],[0.,-1.]])
    result = metrics(a,b,rowwise=True)
    assert result["cosine"] == pytest.approx(0.)
    assert result["cosine_row_mean"] == pytest.approx(1/3)
    assert result["cosine_row_min"] == -1.
    assert result["cosine_row_p01"] == pytest.approx(-.96)
    assert metrics(a[:1]*0,b[:1],rowwise=True)["cosine"] == 0.
    # Tiny but nonzero vectors should not be mistaken for zero by an epsilon.
    assert metrics(a[:1]*1e-20,b[:1]*1e-20,rowwise=True)["cosine"] == pytest.approx(1.)


RANDOM_CASES = [
    (1025,d,dv,direction,p,tau)
    for (d,dv),direction,p,tau in itertools.product(
        itertools.product((32,64,128),repeat=2),
        ("q_from_k","k_from_q"),(0.,0.37),(-8.,8.))
] + [(n,64,128,"random",p,tau)
     for n in (2049,4097) for p,tau in ((0.,0.),(0.,8.),(0.37,8.),(1.,-8.))
] + [(8193,64,128,"random",0.,8.)] + [
    (1025,d,dv,direction,p,math.log(d))
    for (d,dv),direction,p in itertools.product(
        itertools.product((32,64,128),repeat=2),("q_from_k","k_from_q"),(0.,.37))
] + [(n,64,128,"random",p,math.log(64)) for n in (2049,4097,8193) for p in (0.,.37)]


@pytest.mark.parametrize("n,d,dv,direction,probability,tau_value",RANDOM_CASES)
@torch.no_grad()
def test_random_precision(n,d,dv,direction,probability,tau_value,record_property):
    g = torch.Generator(device="cuda").manual_seed(1000+n+d+dv)
    def rand(shape):
        return torch.randn(shape,device="cuda",dtype=torch.bfloat16,generator=g)
    q,k,v = rand((1,1,n,d)),rand((1,1,n,d)),rand((1,1,n,dv))
    qvoc,kvoc = rand((1,31,d)),rand((1,31,d))
    tau = torch.tensor([tau_value],device="cuda")
    scale = d**-0.5
    fp = interpolation_ref(q,k,qvoc,kvoc,scale)
    bf = replace(fp,q_from_k=fp.q_from_k.bfloat16().contiguous(),
        k_from_q=fp.k_from_q.bfloat16().contiguous(),q_lse=fp.q_lse.contiguous(),
        k_lse=fp.k_lse.contiguous(),q_index=fp.q_index.contiguous(),k_index=fp.k_index.contiguous())
    actual,l2,state = forward_interpolated(q,k,v,tau,bf,sm_scale=scale,
        direction=direction,hard_prob=probability,generator=g,return_rng_state=True)
    mask = row_mask(state)
    report = dict(n=n,d=d,dv=dv,direction=state.direction,hard_prob=probability,tau=tau_value,
                  within_tau_bound=tau_value<=math.log(d))
    refs = []
    for name,interp in (("same_bf16_inputs",bf),("fp32_interpolation",fp)):
        # FP32 values keep the oracle PV and output in FP32: we measure actual
        # kernel BF16 weight/output rounding rather than matching its rounding.
        expected,aux = voc_dism_ref(q,k,v.float(),tau,qvoc,kvoc,
            hard_prob=probability,sm_scale=scale,direction=state.direction,
            interpolation=interp,hard_mask=mask,return_aux=True)
        normalizer = torch.logaddexp(torch.logsumexp(aux["scores"],-1),torch.zeros_like(l2))/math.log(2)
        report[name] = dict(output=metrics(actual,expected,rowwise=True),l2=metrics(l2,normalizer))
        refs.append((expected,normalizer))
        del aux
    report["interpolation_quantization"] = dict(output=metrics(refs[0][0],refs[1][0],rowwise=True),
                                                l2=metrics(refs[0][1],refs[1][1]))
    # Record each criterion before asserting, so an earlier failure cannot
    # hide the status of the other oracle. Full FP32 interpolation currently
    # exposes a precision gap; do not silently relax this to make CI green.
    checks = dict(core_output=(actual.float(),refs[0][0],.008,.012),
                  core_l2=(l2,refs[0][1],2e-5,2e-5),
                  fp32_output=(actual.float(),refs[1][0],.008,.012))
    failures = []
    for name,(a,b,atol,rtol) in checks.items():
        report[name+"_passes"] = bool(torch.isclose(a,b,atol=atol,rtol=rtol).all())
        if not report[name+"_passes"]:
            failures.append(name)
    record_property("precision",json.dumps(report))
    assert not failures, f"Failed precision criteria: {failures}; metrics={json.dumps(report)}"


def constant_chain_reference(v, tau):
    """FP64 closed form of the reference recurrence for all-matching labels.

    exp(W[i,j])=sum(exp(t*tau),t=1..j+1), j<=i. Independent of scan/chunks.
    Stable CPU prefix softmax includes the zero-score/zero-value fallback.
    """
    value = v.double().cpu().numpy()[0,0]
    count = np.arange(1,len(value)+1,dtype=np.float64)
    if tau == 0:
        w = np.log(count)
    elif tau > 0:
        w = count*tau+np.log(-np.expm1(-count*tau))-np.log(-np.expm1(-tau))
    else:
        w = tau+np.log(-np.expm1(count*tau))-np.log(-np.expm1(tau))
    out = np.empty_like(value)
    l2 = np.empty(len(value))
    maximum,denominator,numerator = 0.,1.,np.zeros(value.shape[-1])
    for i in range(len(value)):
        m = max(maximum,w[i])
        alpha,beta = np.exp(maximum-m),np.exp(w[i]-m)
        numerator = numerator*alpha+value[i]*beta
        denominator = denominator*alpha+beta
        maximum = m
        out[i] = numerator/denominator
        l2[i] = (m+np.log(denominator))/np.log(2)
    def tensor(x): return torch.from_numpy(x).to(v.device)
    return tensor(out)[None,None],tensor(l2)[None,None],tensor(w/np.log(2))


@pytest.mark.parametrize("tau_value",(-16.,-1.,-0.001,0.,0.001,1.,math.log(32),16.))
@pytest.mark.parametrize("n",(1025,8193))
@torch.no_grad()
def test_long_matching_chain(n,tau_value,record_property):
    g = torch.Generator(device="cuda").manual_seed(901)
    q = torch.zeros((1,1,n,32),device="cuda",dtype=torch.bfloat16)
    v = torch.randn((1,1,n,64),device="cuda",dtype=torch.bfloat16,generator=g)
    labels = torch.zeros((1,1,n),device="cuda",dtype=torch.long)
    lse = torch.zeros_like(labels,dtype=torch.float32)
    tau = torch.tensor([tau_value],device="cuda")
    actual,l2,summary,boundary = forward(q,q,v,lse,tau,labels,labels,
        sm_scale=1.,direction="q_from_k",hard_prob=1.,return_debug=True)
    expected,expected_l2,w = constant_chain_reference(v,float(tau.item()))
    # Only complete checkpoints here; partial-tail identities are covered by
    # the independent double recurrence in test_dism_v2_core.py.
    rows = torch.arange(n//32,device="cuda")*32+31
    cols = torch.arange(boundary.shape[-1],device="cuda")
    target = w[cols.clamp_max(n-1)][None,:].expand(len(rows),-1).clone()
    target.masked_fill_(cols[None,:]>rows[:,None],-torch.inf)
    report = dict(n=n,d=32,dv=64,tau=tau_value,within_tau_bound=tau_value<=math.log(32),
                  output=metrics(actual,expected,rowwise=True),l2=metrics(l2,expected_l2),
                  boundary=metrics(boundary[0,0,:n//32],target))
    record_property("precision",json.dumps(report))
    torch.testing.assert_close(actual.double(),expected,atol=.008,rtol=.012)
    torch.testing.assert_close(l2.double(),expected_l2,atol=2e-5,rtol=2e-5)
    torch.testing.assert_close(boundary[0,0,:n//32].double(),target,atol=3e-5,rtol=3e-5)


@torch.no_grad()
def test_chain_closed_form_matches_reference():
    # Anchor the long-sequence analytic oracle to dism_ref.py at small N.
    from dism_v2.dism_ref import dism_recurrence, normalize_with_zero_fallback
    v = torch.randn((1,1,65,32),device="cuda",dtype=torch.float64)
    for tau in (-16.,-.001,0.,.001,16.):
        scores = dism_recurrence(torch.full((1,1,65,65),tau,device="cuda",dtype=torch.float64))
        expected = normalize_with_zero_fallback(scores,v)
        out,l2,_ = constant_chain_reference(v,tau)
        # Reference softplus uses its documented large-positive linear branch.
        torch.testing.assert_close(out,expected,atol=2e-7,rtol=2e-7)
        norm = torch.logaddexp(torch.logsumexp(scores,-1),torch.zeros_like(l2))/math.log(2)
        torch.testing.assert_close(l2,norm,atol=2e-7,rtol=2e-7)
