"""LM contract: untied weights, decay, packing resume, causal SWA and real CUDA gradients."""
import copy

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from torch.nn import functional as F

from dism_v2.lm_data import PackedStream, split_files
from dism_v2.lm_model import DecoderLM, LMConfig, SlidingAttention, parameter_groups, hard_probability, learning_rate
from fla.modules.fused_cross_entropy import FusedCrossEntropyLoss


def test_model_parameters_and_decay():
    model = DecoderLM()
    assert 49_000_000 < sum(p.numel() for p in model.parameters()) < 51_000_000
    assert model.embedding.weight is not model.lm_head.weight
    assert model.embedding.weight.data_ptr() != model.lm_head.weight.data_ptr()
    decay, no_decay = parameter_groups(model, .1)
    ids = {id(p) for p in no_decay['params']}
    for name, p in model.named_parameters():
        assert (id(p) in ids) == (p.ndim <= 1 or getattr(p, '_no_weight_decay', False)), name
    assert sum(p.numel() for g in (decay, no_decay) for p in g['params']) == sum(p.numel() for p in model.parameters())


def test_schedules():
    assert hard_probability(0, 30000) == 0
    assert hard_probability(29999, 30000) == 1
    assert hard_probability(30010, 30000) == 1
    assert 0 < hard_probability(1000, 30000) < 1
    assert learning_rate(999, 30000, 1000, 3e-4) == 3e-4
    assert learning_rate(29999, 30000, 1000, 3e-4) == pytest.approx(3e-5)


class TinyTokenizer:
    eos_token_id = 99
    def __call__(self, texts, **kwargs):
        return {'input_ids': [[int(x) for x in text.split()] for text in texts]}


def test_stream_resume_and_split(tmp_path):
    # Multiple parquet batches, row groups, shards, and corpus epochs.
    for name in ('train-0', 'train-1', 'validation-0'):
        pq.write_table(pa.table({'text': ['1 2 3 4 5'] * 151}), tmp_path / (name + '.parquet'), row_group_size=73)
    train, val = split_files(tmp_path)
    assert len(train) == 2 and len(val) == 1
    assert not set(train) & set(val)
    stream = PackedStream(train, TinyTokenizer(), context=7, batch_size=3)
    for _ in range(42):
        x, y = stream.next_batch()
        torch.testing.assert_close(x[:, 1:], y[:, :-1])
    resumed = PackedStream(train, TinyTokenizer(), context=7, batch_size=3)
    resumed.load_state_dict(copy.deepcopy(stream.state_dict()))
    for _ in range(160):
        for a, b in zip(stream.next_batch(), resumed.next_batch()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert stream.epoch > 0
    finite = PackedStream(val, TinyTokenizer(), context=7, batch_size=3, repeat=False)
    for _ in range(37):
        finite.next_batch()
    with pytest.raises(StopIteration):
        finite.next_batch()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA')
def test_swa_matches_causal_window():
    torch.manual_seed(99)
    c = LMConfig(context=256)
    model = SlidingAttention(c).cuda()
    x = torch.randn(1, 193, 256, device='cuda', requires_grad=True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        result = model(x)
        q, k, v = model.qkv(x).reshape(1, 193, 3, 4, 64).unbind(2)
        q, k = model.rope(q).transpose(1, 2), model.rope(k).transpose(1, 2)
        i = torch.arange(193, device='cuda')
        mask = (i[:, None] >= i[None, :]) & (i[:, None] - i[None, :] < 128)
        scores = (q.float() @ k.float().transpose(-1, -2)) / 8
        weights = scores.masked_fill(~mask, -torch.inf).softmax(-1)
        expected = model.out((weights.to(v.dtype) @ v.transpose(1, 2)).transpose(1, 2).reshape(1, 193, 256))
    torch.testing.assert_close(result, expected, atol=.004, rtol=.03)
    result.float().square().mean().backward()
    assert torch.isfinite(x.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA')
def test_fused_ce_backward():
    torch.manual_seed(11)
    x = torch.randn(19, 50257, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    y = torch.randint(50257, (19,), device='cuda')
    ref = x.detach().float().requires_grad_()
    expected = F.cross_entropy(ref, y)
    actual = FusedCrossEntropyLoss(inplace_backward=True)(x, y)
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
    expected.backward(); actual.backward()
    torch.testing.assert_close(x.grad.float(), ref.grad, atol=3e-5, rtol=.01)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA')
def test_real_lm_backward_and_rng_replay():
    torch.manual_seed(81)
    model = DecoderLM(LMConfig(layers=1, context=256, softcap=30.)).cuda()
    x = torch.randint(50257, (1, 129), device='cuda')
    y = torch.randint(50257, x.shape, device='cuda')
    generator = torch.Generator(device='cuda').manual_seed(37)
    rng = generator.get_state()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss = model(x, y, .37, generator)
        generator.set_state(rng)
        replay = model(x, y, .37, generator)
    torch.testing.assert_close(loss, replay, rtol=0, atol=0)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    assert model.embedding.weight.grad.abs().sum() > 0
    assert model.lm_head.weight.grad.abs().sum() > 0
    assert model.blocks[0].swa.qkv.weight.grad.abs().sum() > 0
    assert model.blocks[0].dism.q_voc.grad.abs().sum() > 0
