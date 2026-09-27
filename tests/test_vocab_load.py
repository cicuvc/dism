import pytest
import torch
from dism_v2.eval_vocab_load import load_metrics, alignment


def test_uniform_load():
    c=torch.ones(512,dtype=torch.long)*100
    r=load_metrics(c)
    assert r['unused']==0 and r['below_1pct_uniform']==0
    assert r['effective_vocab']==pytest.approx(512)
    assert r['gini']==pytest.approx(0)
    assert alignment(c,c)['overlap']==1
    assert alignment(c,c)['js_nats']==pytest.approx(0)


def test_disjoint_collapsed_load():
    q=torch.tensor([100,0,0,0]);k=q.flip(0)
    r=load_metrics(q)
    assert r['unused']==3 and r['effective_vocab']==1
    assert r['top1_mass']==1 and r['labels_for_90pct']==1
    a=alignment(q,k)
    assert a['overlap']==0 and a['q_mass_on_unused_k']==1
    assert a['js_nats']==pytest.approx(0.69314718056)
