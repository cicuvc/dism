"""Inference-only top2 readout ablation of an unchanged top4 checkpoint.

Retain argmax histories and top4 kernel geometry, zero slots 2/3 and renormalize
slots 0/1. This measures quality, not a specialized top2 kernel's performance.
All overrides are process-local; the checkpoint and production code stay intact.
Remaining CLI arguments are forwarded to the frozen BOS-corrected evaluator.
"""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import runpy
import sys

import torch
from flash_dism.kernels import topk_hard


def top2_probabilities(pi):
    out = torch.zeros_like(pi)
    two = pi[..., :2]
    out[..., :2] = two / two.sum(-1, keepdim=True).clamp_min(torch.finfo(pi.dtype).tiny)
    return out


@contextmanager
def top2_readout():
    original = topk_hard.select
    calls = {'count': 0}

    def select(*args, **kwargs):
        if torch.is_grad_enabled():
            raise RuntimeError('Top2 ablation is inference-only; no surrogate backward override')
        ids, pi = original(*args, **kwargs)
        calls['count'] += 1
        return ids, top2_probabilities(pi)

    topk_hard.select = select
    try:
        yield calls
    finally:
        topk_hard.select = original


def self_test():
    from check_topk_training import oracle
    torch.manual_seed(176); torch.set_num_threads(1)
    b,n,h,d,r,c = 1,512,2,32,16,8
    q=torch.randn(b,n,h,d,device='cuda',dtype=torch.bfloat16)*.2
    eq=torch.randn(h,c,d,device='cuda')*.2
    logits=torch.einsum('bnhd,hcd->bhnc',q.double(),eq.bfloat16().double())
    iq=logits.argmax(-1).int().contiguous()
    hard=torch.ones(b,h,device='cuda',dtype=torch.bool)
    original=topk_hard.select
    with torch.inference_mode(),top2_readout() as calls:
        ids,pi=topk_hard.select(q,eq,iq,hard,1.)
    assert calls['count']==1 and topk_hard.select is original
    assert torch.equal(ids[...,0],iq) and not pi[...,2:].count_nonzero()
    expected=logits.gather(-1,ids[...,:2].long()).softmax(-1)
    torch.testing.assert_close(pi[...,:2].double(),expected,atol=2e-6,rtol=2e-5)
    assert not top2_probabilities(torch.zeros_like(pi)).count_nonzero()
    sq=torch.randn(b,n,h,r,device='cuda',dtype=torch.bfloat16)*.2
    sk=torch.randn_like(sq)*.2;v=torch.randn(b,n,h,d,device='cuda',dtype=torch.bfloat16)
    tau=torch.tensor([0.,.1],device='cuda');seg=torch.zeros(b,n,device='cuda',dtype=torch.int32);seg[:,256:]=1
    for same in (False,True):
        ik=torch.randint(c,(b,h,n),device='cuda',dtype=torch.int32)
        labels=iq.clone();candidates=ids.clone()
        if same:
            labels.zero_();ik.zero_();candidates[:]=torch.arange(4,device='cuda')
        with torch.inference_mode():
            actual,_=topk_hard.forward(sq,sk,v,labels,ik,candidates,pi,tau,hard,seg)
            z=pi.double().log()
            reference=oracle(sq.double(),sk.double(),v.double(),labels,ik,candidates,z,tau.double(),seg)
            relative=float((actual.double()-reference).norm()/reference.norm())
            cosine=float(torch.nn.functional.cosine_similarity(actual.double().flatten(),reference.flatten(),dim=0))
        assert bool(torch.isfinite(actual).all()) and relative<.025 and cosine>.999
        print('TOP2_REFERENCE_PASS',dict(repeated=same,relative_l2=relative,cosine=cosine),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__,add_help=False)
    parser.add_argument('--eval-root',type=Path,default=Path('/home/cicuvc/downstream-eval'))
    parser.add_argument('--self-test',action='store_true')
    args,remaining=parser.parse_known_args()
    if args.self_test:
        self_test();return
    checkpoint=Path(remaining[remaining.index('--checkpoint')+1])
    output=Path(remaining[remaining.index('--output')+1])
    metadata=json.loads(checkpoint.with_name(checkpoint.name.replace('model_','meta_')).with_suffix('.json').read_text())
    config=metadata['model_spec']['config']
    assert config.get('hard_query_topk')==4 and not config.get('hard_query_sampled',False)
    adapter=args.eval_root/'adapter_bos_yarn_20261025.py'
    sys.path.insert(0,str(args.eval_root));sys.argv=[str(adapter),*remaining]
    with top2_readout() as calls:
        runpy.run_path(str(adapter),run_name='__main__')
    assert calls['count']>0,'No top4 selectors intercepted; refusing mislabeled results'
    protocol=output/'protocol.json';data=json.loads(protocol.read_text())
    data.update(checkpoint_topk=4,evaluation_topk=2,selector_override_calls=calls['count'],
                readout_ablation='Keep top2, renormalize within top2, zero remaining slots; argmax histories unchanged',
                kernel_geometry='Original four-slot kernels; quality ablation only, no speedup claim')
    protocol.write_text(json.dumps(data,indent=2)+'\n')
    print('TOP2_ABLATION_COMPLETE',calls,flush=True)


if __name__=='__main__':main()
