import torch
from dism_v2.eval_vocab_large import causal_label_coverage


def test_coverage_matches_causal_bruteforce():
    torch.manual_seed(5)
    q=torch.randint(9,(2,3,65))
    k=torch.randint(9,(2,3,65))
    expected=torch.stack([(q[:,:,i,None]==k[:,:,:i+1]).any(-1) for i in range(65)],-1)
    torch.testing.assert_close(causal_label_coverage(q,k,9),expected)
