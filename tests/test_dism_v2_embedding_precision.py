"""Actual emb_kernel forward -> Dism core, with separated precision oracles."""
import itertools
import json
import math
from dataclasses import replace

import pytest
import torch

from dism_v2.core import forward_interpolated
from dism_v2.dism_ref import InterpolationResult, interpolation_ref, voc_dism_ref
from dism_v2.emb_kernel import EmbInterpFunction, emb_fwd
from test_dism_v2_precision import metrics, row_mask, full_precision_torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

CASES = [(1025,d,dv,31,direction,p,1,1)
    for (d,dv),direction,p in itertools.product(itertools.product((32,64,128),repeat=2),
        ("q_from_k","k_from_q"),(0.,.37,1.))]
CASES += [(n,64,128,31,"random",p,1,1)
    for n in (2049,4097,8193) for p in (0.,.37)]
CASES += [(257,64,32,vocab,"q_from_k",.37,2,2) for vocab in (64,65,129,257)]


def embedding_fp32_store(q,k,qvoc,kvoc,scale):
    """Diagnostic invocation of the unchanged Triton function, FP32 outputs.

    The softmax weights are STILL BF16 before tl.dot. Only final output
    storage rounding is removed. This is not a proposed production interface.
    """
    batch,heads,n,d = q.shape
    out = [torch.empty(q.shape,device=q.device,dtype=torch.float32) for _ in range(2)]
    out += [torch.empty(q.shape[:3],device=q.device,dtype=torch.float32) for _ in range(4)]
    out += [torch.empty(q.shape[:3],device=q.device,dtype=torch.int32) for _ in range(2)]
    emb_fwd[(batch*heads,(n+63)//64)](
        q,k,qvoc,kvoc,*out,batch,heads,n,scale,
        *q.stride()[:3],*k.stride()[:3],*qvoc.stride()[:2],*kvoc.stride()[:2],
        *out[0].stride()[:3],*out[2].stride()[:2],d,qvoc.shape[1],64,64)
    return InterpolationResult(*out)


@pytest.mark.parametrize("n,d,dv,vocab,direction,probability,batch,heads",CASES)
@torch.no_grad()
def test_embedding_precision(n,d,dv,vocab,direction,probability,batch,heads,record_property):
    # Match the previous precision suite's random input for the long V=31 case.
    g = torch.Generator(device="cuda").manual_seed(1000+n+d+dv)
    def rand(shape): return torch.randn(shape,device="cuda",dtype=torch.bfloat16,generator=g)
    q,k,v = rand((batch,heads,n,d)),rand((batch,heads,n,d)),rand((batch,heads,n,dv))
    qvoc,kvoc = rand((heads,vocab,d)),rand((heads,vocab,d))
    tau = torch.full((heads,),math.log(d),device="cuda")
    scale = d**-.5
    fp = interpolation_ref(q,k,qvoc,kvoc,scale)
    bf = replace(fp,q_from_k=fp.q_from_k.bfloat16().contiguous(),
        k_from_q=fp.k_from_q.bfloat16().contiguous(),q_lse=fp.q_lse.contiguous(),
        k_lse=fp.k_lse.contiguous(),q_index=fp.q_index.contiguous(),k_index=fp.k_index.contiguous())
    # Import the v2 module explicitly, avoiding the reference's legacy absolute
    # `from emb_kernel import ...`. Preserve its actual eight-value ordering.
    emb = InterpolationResult(*EmbInterpFunction.apply(q,k,qvoc,kvoc,scale))
    assert emb.q_from_k.dtype == emb.k_from_q.dtype == torch.bfloat16
    # Current CUDA core accepts int64 labels. This conversion is lossless and
    # does not replace the embedding kernel's chosen labels with oracle labels.
    emb = replace(emb,q_index=emb.q_index.long(),k_index=emb.k_index.long())
    emb_float = embedding_fp32_store(q,k,qvoc,kvoc,scale)
    report = dict(n=n,d=d,dv=dv,vocab=vocab,batch=batch,heads=heads,tau=math.log(d),
        hard_prob=probability,within_tau_bound=True)
    for name in ("q_from_k","k_from_q","q_lse","k_lse","q_top_prob","k_top_prob"):
        report[name] = metrics(getattr(emb,name),getattr(fp,name),rowwise=name.endswith("from_k") or name.endswith("from_q"))
    for name in ("q_from_k","k_from_q"):
        report["pre_store_"+name] = metrics(getattr(emb_float,name),getattr(fp,name),rowwise=True)
    report["q_label_mismatches"] = int((emb.q_index!=fp.q_index).sum())
    report["k_label_mismatches"] = int((emb.k_index!=fp.k_index).sum())
    actual,l2,state = forward_interpolated(q,k,v,tau,emb,sm_scale=scale,
        direction=direction,hard_prob=probability,generator=g,return_rng_state=True)
    report["direction"] = state.direction
    mask = row_mask(state)
    references = {}
    for name,interp in (("emb_inputs",emb),("torch_bf16_inputs",bf),("torch_fp32_inputs",fp),
                        ("emb_fp32_store",emb_float)):
        expected,aux = voc_dism_ref(q,k,v.float(),tau,qvoc,kvoc,hard_prob=probability,
            sm_scale=scale,direction=state.direction,interpolation=interp,hard_mask=mask,return_aux=True)
        norm = torch.logaddexp(torch.logsumexp(aux["scores"],-1),torch.zeros_like(l2))/math.log(2)
        references[name] = (expected,norm)
        report["cuda_vs_"+name] = dict(output=metrics(actual,expected,rowwise=True),l2=metrics(l2,norm))
        del aux
    for name in ("torch_bf16_inputs","torch_fp32_inputs","emb_fp32_store"):
        report["emb_reference_vs_"+name] = dict(
            output=metrics(references["emb_inputs"][0],references[name][0],rowwise=True),
            l2=metrics(references["emb_inputs"][1],references[name][1]))
    report["pre_store_reference_vs_torch_fp32_inputs"] = dict(
        output=metrics(references["emb_fp32_store"][0],references["torch_fp32_inputs"][0],rowwise=True),
        l2=metrics(references["emb_fp32_store"][1],references["torch_fp32_inputs"][1]))
    checks = dict(core_output=(actual.float(),references["emb_inputs"][0],.008,.012),
                  core_l2=(l2,references["emb_inputs"][1],2e-5,2e-5),
                  fp32_output=(actual.float(),references["torch_fp32_inputs"][0],.008,.012),
                  q_lse=(emb.q_lse,fp.q_lse,2e-5,2e-5),
                  k_lse=(emb.k_lse,fp.k_lse,2e-5,2e-5))
    failures = []
    for name,(a,b,atol,rtol) in checks.items():
        report[name+"_passes"] = bool(torch.isclose(a,b,atol=atol,rtol=rtol).all())
        if not report[name+"_passes"]: failures.append(name)
    record_property("embedding_precision",json.dumps(report))
    assert not failures, f"Failed precision criteria: {failures}; metrics={json.dumps(report)}"
