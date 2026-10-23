"""State ownership and numerical checks for the explicit SAM model session."""
import pytest
import torch
from nanochat.models.dism.inference import SamDecodeSession

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA')


@pytest.mark.parametrize('length', [1, 3, 17, 256])
def test_convolution_history(length):
    from nanochat.models.dism.conv import LinearShortConvSiLU
    torch.manual_seed(1)
    conv = LinearShortConvSiLU(32, 32, 4).cuda()
    torch.nn.init.normal_(conv.weight, std=.02)
    torch.nn.init.normal_(conv.conv.weight, std=.1)
    x = torch.randn(2, length+5, 32, device='cuda', dtype=torch.bfloat16)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        reference = conv(x)
        state = {}
        output = [SamDecodeSession._conv(conv, x[:, :length], state, 'v')]
        for i in range(length, length+5):
            output.append(SamDecodeSession._conv(conv, x[:, i:i+1], state, 'v'))
        torch.testing.assert_close(torch.cat(output, 1), reference, rtol=0, atol=0)
        assert state['v'].shape[1] == 3


@pytest.mark.parametrize('length', [17, 256])
def test_session_state_and_rebuild(length):
    from nanochat.models import build_model, ModelSpec
    from flash_dism.inference import HardDismPrefill
    torch.manual_seed(11)
    model = build_model(ModelSpec('dism_gdn_hybrid', 1, dict(
        sequence_len=512, vocab_size=128, n_layer=2, n_head=2, n_embd=64,
        intermediate_size=128, head_dim=32, value_dim=32, readout_dim=16,
        qk_vocab_size=32, mxfp8=False, mxfp8_projections=False))).cuda()
    model.init_weights(); model.eval()
    before = {k: v.clone() for k,v in model.state_dict().items()}
    tokens = torch.randint(128, (1,length+5), device='cuda')
    with torch.inference_mode(), HardDismPrefill(workers=2) as prefill:
        session = SamDecodeSession(model, prefill, length+5, rebuild_interval=2)
        parts = [session.features(tokens[:,:length])]
        with pytest.raises(ValueError, match='batch'):
            session.features(tokens.expand(2,-1)[:, :1])
        with pytest.raises(ValueError, match='one token'):
            session.features(tokens[:, :2])
        for i in range(length,length+5):
            parts.append(session.features(tokens[:,i:i+1]))
        actual = torch.cat(parts,1)
        full = SamDecodeSession(model, prefill, length+5)
        expected = full.features(tokens)
        # Same tolerance as module CUDA cached-decoding regression.
        torch.testing.assert_close(actual, expected, atol=.005, rtol=.02)
        assert session.position == length+5
        for state in session.states:
            assert state['gdn'] is not None
            assert all(t.shape[1] == 3 for t in state['conv'].values())
            if state['dism'] is not None:
                assert state['dism'].core.check_status() == length+5
        with pytest.raises(ValueError, match='capacity'):
            session.features(tokens[:,:1])
        session.close()
        with pytest.raises(ValueError, match='closed'):
            session.features(tokens[:,:1])
        full.close()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)
