import math
import torch
from experiments.embedding_margin_3k import metrics, norm_stats


def test_uniform_margin():
    values,_=metrics(torch.zeros(2,512))
    torch.testing.assert_close(values[:,0],torch.full((2,),1/512))
    assert torch.count_nonzero(values[:,2:4])==0
    torch.testing.assert_close(values[:,4],torch.full((2,),math.log(512)))
    torch.testing.assert_close(values[:,5],torch.full((2,),512.))


def test_margin_probability_ratio_and_shift():
    logits=torch.tensor([[1.,3.,2.,-2.]])
    values,ids=metrics(logits)
    assert ids.item()==1
    torch.testing.assert_close(values[:,0]/values[:,1],values[:,3].exp())
    shifted,_=metrics(logits+10)
    torch.testing.assert_close(shifted,values,atol=2e-6,rtol=2e-6)
    assert norm_stats(torch.tensor([1.,2.,3.]))['mean']==2.
