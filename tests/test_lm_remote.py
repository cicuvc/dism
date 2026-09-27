import copy

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from dism_v2.lm_data import PackedStream
from dism_v2.lm_token_stream import IndexedBatches, RemotePackedStream


class Tokenizer:
    eos_token_id = 99
    def __call__(self, texts, **kwargs):
        return {'input_ids': [[int(x) for x in text.split()] for text in texts]}


def test_indexed_stream_replay_and_restart(tmp_path):
    file = tmp_path / 'train.parquet'
    pq.write_table(pa.table({'text': ['1 2 3 4 5 6'] * 73}), file, row_group_size=19)
    def stream(batch):
        return PackedStream([str(file)], Tokenizer(), context=7, batch_size=batch, seed=777)
    indexed = IndexedBatches(stream(6), tmp_path / 'state', 'identity', cache_size=3)
    oracle = stream(2)
    expected = []
    for index in range(107):
        blocks = []
        for _ in range(3):
            x, y = oracle.next_batch()
            blocks.append(torch.cat((x, y[:, -1:]), 1))
        payload = torch.cat(blocks).numpy().astype('<i4').tobytes()
        expected.append(payload)
        assert indexed.get(index)[0] == payload
    assert indexed.get(0)[0] == expected[0]
    assert indexed.get(103)[0] == expected[103]
    restarted = IndexedBatches(stream(6), tmp_path / 'state', 'identity', cache_size=3)
    assert restarted.get(105)[0] == expected[105]
    assert restarted.index == 106


def test_remote_microbatch_resume(monkeypatch):
    from dism_v2 import lm_token_stream as module
    def fetch(url, secret, path):
        index = int(path.rsplit('/', 1)[-1])
        return (np.arange(48, dtype='<i4') + index * 48).tobytes(), index // 3
    monkeypatch.setattr(module, 'fetch', fetch)
    meta = dict(batch=6, context=7, steps=30, eval_batches=10, identity='test')
    first = RemotePackedStream('unused', 'secret', 'train', meta, 2)
    resumed = RemotePackedStream('unused', 'secret', 'train', meta, 2)
    try:
        for _ in range(4):
            first.next_batch()
        state = copy.deepcopy(first.state_dict())
        resumed.load_state_dict(state)
        for _ in range(11):
            for left, right in zip(first.next_batch(), resumed.next_batch()):
                torch.testing.assert_close(left, right, atol=0, rtol=0)
        bad = dict(state, identity='changed')
        with pytest.raises(ValueError):
            resumed.load_state_dict(bad)
    finally:
        first.close()
        resumed.close()


def test_swa_parameter_match():
    from dism_v2.lm_model import LMConfig, matched_swa_config, parameter_count, DecoderLM
    ref = LMConfig(softcap=30.)
    config = matched_swa_config(ref)
    assert config.ffn_hidden == 1728
    assert parameter_count(ref) == 49_678_876
    assert parameter_count(config) == 49_641_856
    assert abs(parameter_count(config) / parameter_count(ref) - 1) < .001
    with torch.device('meta'):
        model = DecoderLM(config)
    assert all(block.dism is None for block in model.blocks)
    assert model.embedding.weight is not model.lm_head.weight


def test_position_moments():
    from dism_v2.eval_lm_positions import PositionMoments
    result = PositionMoments(3)
    result.update(torch.tensor([[1., 2., 3.], [3., 4., 5.]]))
    stats = result.result()
    assert stats['mean'] == [2., 3., 4.]
    assert stats['stderr'] == [1., 1., 1.]
    assert stats['count'] == [2, 2, 2]


def test_full_attention_config_and_dispatch(monkeypatch):
    from dataclasses import replace
    from dism_v2 import lm_model as module
    swa = module.matched_swa_config(module.LMConfig(softcap=30.))
    full = replace(swa, architecture='full_attention', window=-1)
    assert module.parameter_count(full) == module.parameter_count(swa) == 49_641_856
    calls = []
    def attention(q, k, v, **kwargs):
        calls.append(kwargs)
        return v
    monkeypatch.setattr(module, 'flash_attn_func', attention)
    model = module.SlidingAttention(full)
    result = model(torch.randn(1, 129, 256))
    assert result.shape == (1, 129, 256)
    assert calls == [dict(causal=True, window_size=(-1, -1), dropout_p=0.)]
    with pytest.raises(ValueError, match='window=-1'):
        module.DecoderLM(replace(full, window=128))
