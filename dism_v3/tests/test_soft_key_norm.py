import torch
from torch.nn import functional as F
from flash_dism import DismAttention, DismConfig


def test_soft_key_only_norm_and_gradient():
    module = DismAttention(64, 2, readout_dim=32, soft_k_l2_norm=True)
    x = torch.randn(2, 5, 64, requires_grad=True)
    q = module._activate_readout(x)
    k = module._activate_readout(x, is_key=True)
    expected_q = F.silu(x).unflatten(-1, (2, 32))
    expected_k = F.normalize(expected_q, p=2, dim=-1, eps=1e-6)
    torch.testing.assert_close(q, expected_q)
    torch.testing.assert_close(k, expected_k)
    torch.testing.assert_close(k.norm(dim=-1), torch.ones_like(k[..., 0]))
    weight = torch.randn_like(k)
    actual = torch.autograd.grad((k*weight).sum(), x, retain_graph=True)[0]
    expected = torch.autograd.grad((expected_k*weight).sum(), x)[0]
    torch.testing.assert_close(actual, expected)
    default = DismAttention(64, 2, readout_dim=32)
    torch.testing.assert_close(default._activate_readout(x, is_key=True), expected_q)
    config = DismConfig(soft_k_l2_norm=True)
    assert DismConfig.from_dict(config.to_dict()).soft_k_l2_norm
