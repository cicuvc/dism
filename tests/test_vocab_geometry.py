import pytest
import torch
from torch import nn

from dism_v2.check_vocab_geometry import correlation, effective, surgery, transformed_vocab


def test_correlation_and_effective():
    assert correlation(torch.arange(4), torch.arange(4)*3) == pytest.approx(1)
    assert correlation(torch.ones(4), torch.arange(4)) is None
    assert effective(torch.tensor([1, 1, 1, 1])) == pytest.approx(4)
    assert effective(torch.tensor([0, 10, 0, 0])) == pytest.approx(1)


def test_norm_matched_silu():
    e = torch.randn(2, 16, 8)
    changed = transformed_vocab(e, 'embedding_silu_norm')
    torch.testing.assert_close(changed.norm(dim=-1), e.norm(dim=-1))
    assert not torch.equal(changed, e)
    torch.testing.assert_close(transformed_vocab(e, 'embedding_silu'), torch.nn.functional.silu(e))


def test_surgery_restores_on_exception():
    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_voc = nn.Parameter(torch.randn(2, 16, 8))
            self.k_voc = nn.Parameter(torch.randn(2, 16, 8))
            self.qd_conv, self.kd_conv = nn.Identity(), nn.Identity()
            self.qd_conv.activation = self.kd_conv.activation = 'swish'

        def expanded_vocabularies(self, dtype):
            return self.q_voc.to(dtype), self.k_voc.to(dtype)

    model = nn.Module()
    block = nn.Module()
    block.dism = Attention()
    model.blocks = nn.ModuleList([block])
    original = block.dism.q_voc.detach().clone()
    for mode in ('embedding_silu', 'no_qk_silu'):
        with pytest.raises(RuntimeError):
            with surgery(model, mode, None):
                if mode == 'embedding_silu':
                    q, k = block.dism.expanded_vocabularies(torch.bfloat16)
                    torch.testing.assert_close(q, torch.nn.functional.silu(original).bfloat16())
                    assert k.dtype == torch.bfloat16
                else:
                    assert block.dism.qd_conv.activation is None
                raise RuntimeError('test cleanup')
        assert block.dism.qd_conv.activation == 'swish'
        torch.testing.assert_close(block.dism.expanded_vocabularies(torch.float32)[0], original)
        torch.testing.assert_close(block.dism.q_voc, original)
