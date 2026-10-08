import pytest
import torch
import torch.nn.functional as F
import cu_flash_dism


@pytest.mark.parametrize('columns', [64,128,320])
@pytest.mark.parametrize('initial', ['empty', 'random'])
@pytest.mark.parametrize('masked', [False, True])
@pytest.mark.parametrize('exact', [False, True])
def test_forward_prescan_postscan(columns, initial, masked, exact):
    if not exact and cu_flash_dism.summary_lse_mode() != 0:
        pytest.skip('forward scan approximation probe currently supports tanh only')
    torch.manual_seed(198)
    scores = torch.randn(16, columns, device='cuda') * .2 - .1
    top = torch.randn(columns, device='cuda') * 2
    if initial == 'empty':
        top.fill_(-1e6)
    if masked:
        scores.masked_fill_(torch.rand_like(scores) < .25, -1e6)
    got, early_bottom = cu_flash_dism.forward_scan_probe(scores, top, exact)
    state = top.double()
    reference = []
    for row in scores.double():
        incoming = F.pad(state[:-1], (1, 0), value=-1e6)
        state = row + torch.logaddexp2(incoming, torch.zeros_like(incoming))
        reference.append(state)
    expected = torch.stack(reference).float()
    reachable = expected > -1e5
    assert (got[~reachable] < -1e5).all()
    tolerance = 2e-5 if exact else .03
    torch.testing.assert_close(got[reachable], expected[reachable], atol=tolerance, rtol=2e-5)
    bottom_mask = reachable[-1]
    assert (early_bottom[~bottom_mask] < -1e5).all()
    torch.testing.assert_close(early_bottom[bottom_mask], expected[-1][bottom_mask],
                               atol=tolerance, rtol=2e-5)
    if exact:
        torch.testing.assert_close(early_bottom[bottom_mask], got[-1][bottom_mask],
                                   atol=2e-5, rtol=2e-6)
