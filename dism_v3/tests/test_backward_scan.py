import pytest
import torch
import cu_flash_dism


@pytest.mark.parametrize('kind', ['random', 'identity', 'zero', 'signed'])
@pytest.mark.parametrize('boundary', ['zero', 'random'])
def test_reverse_affine_early_boundary(kind, boundary):
    torch.manual_seed(81)
    pairs = torch.randn(16,32,2, dtype=torch.float64)
    pairs[...,0] = torch.rand(16,32,dtype=torch.float64)
    if kind == 'identity':
        pairs[...,0] = 1
        pairs[...,1] = 0
    elif kind == 'zero':
        pairs[...,0] = 0
    elif kind == 'signed':
        pairs[...,0] -= .5
    bottom = torch.randn(32,dtype=torch.float64)
    right = torch.randn(16,dtype=torch.float64)
    if boundary == 'zero':
        bottom.zero_()
        right.zero_()
    # Include corner [16,32], separately encoded in right[-1].
    oracle = torch.zeros(17,33,2,dtype=torch.float64)
    oracle[...,0] = 1
    oracle[16,:32,1] = bottom
    oracle[1:,32,1] = right
    for i in range(15,-1,-1):
        for j in range(31,-1,-1):
            a,b = pairs[i,j]
            x,y = oracle[i+1,j+1]
            oracle[i,j,0] = a*x
            oracle[i,j,1] = a*y+b
    result,edge = cu_flash_dism.backward_scan_probe(
        pairs.float().cuda(),bottom.float().cuda(),right.float().cuda())
    torch.testing.assert_close(result.cpu(),oracle[:16,:32].float(),atol=2e-6,rtol=2e-5)
    torch.testing.assert_close(edge.cpu(),oracle[0,:32].float(),atol=2e-6,rtol=2e-5)
