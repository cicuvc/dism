"""Local synthetic NIAH tests: state isolation, failures and paired scoring."""
import copy
import math
import pickle
import random
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from nanochat.niah_eval import ShortNIAH, build_suite


class Tokenizer:
    def encode(self, text, prepend=None, **kwargs):
        if isinstance(text, list):
            return [[prepend] + self.encode(t) for t in text]
        return list(text.encode('ascii'))

    def get_bos_token_id(self):
        return 0


class Model(torch.nn.Module):
    aligned_segment_ends = True

    def __init__(self, failure=None):
        super().__init__()
        self.norm = torch.nn.Identity()
        self.lm_head = torch.nn.Linear(4, 128, bias=False)
        torch.nn.init.zeros_(self.lm_head.weight)
        self.config = SimpleNamespace(sequence_len=256, softcap=0.)
        self.eval_hard = False
        self.register_buffer('rng_counter', torch.tensor(37))
        self.register_buffer('qb_price', torch.arange(4.))
        self.failure, self.calls = failure, 0

    def forward(self, x, cu_seqlens):
        assert not self.training and self.eval_hard and not torch.is_grad_enabled()
        assert x.shape[1] <= self.config.sequence_len
        self.calls += 1
        random.random(); np.random.rand(); torch.rand(1, device=x.device)
        if self.failure == 'buffer':
            self.rng_counter.add_(1)
        if self.failure == 'runtime':
            raise RuntimeError('injected failure')
        return self.norm(torch.zeros((*x.shape, 4), device=x.device))


@pytest.fixture
def data(tmp_path):
    path = tmp_path / 'pg19.parquet'
    pq.write_table(pa.table({'text': ['abcdefghij ' * 1000]}), path)
    return path


def rng_state():
    return (pickle.dumps(random.getstate()), pickle.dumps(np.random.get_state()),
            torch.get_rng_state().clone(),
            [s.clone() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_initialized() else [])


def assert_rng(before):
    after = rng_state()
    assert before[:2] == after[:2]
    assert torch.equal(before[2], after[2])
    assert all(torch.equal(a, b) for a, b in zip(before[3], after[3]))


@pytest.mark.parametrize('failure', [None, 'runtime', 'buffer'])
def test_state_and_failure_restoration(data, failure):
    model = Model(failure).train()
    model.norm.eval()  # Preserve mixed module modes, too.
    model.lm_head.weight.grad = torch.ones_like(model.lm_head.weight)
    before = copy.deepcopy(model.state_dict())
    versions = [b._version for b in model.buffers()]
    rng = rng_state()
    probe = ShortNIAH(data, contexts=[512], depths=[.5], trials=1)
    if failure:
        with pytest.raises(RuntimeError, match='injected failure|mutated model buffers'):
            probe.evaluate(model, Tokenizer())
    else:
        result = probe.evaluate(model, Tokenizer())
        assert result['status'] == 'ok'
        assert result['metrics']['niah/present_nll'] == pytest.approx(math.log(128))
        assert result['metrics']['niah/gain'] == 0
        assert model.calls == 2
        assert versions == [b._version for b in model.buffers()]
    assert_rng(rng)
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, before[name])
    assert model.training and not model.norm.training and model.lm_head.training
    assert not model.eval_hard and model.config.sequence_len == 256
    assert torch.all(model.lm_head.weight.grad == 1)
    assert not model.norm._forward_hooks


def test_missing_data_retries_and_warns_once(tmp_path, data, caplog):
    path = tmp_path / 'later.parquet'
    probe = ShortNIAH(path, contexts=[256], depths=[.5], trials=1)
    model, tokenizer = Model(), Tokenizer()
    rng = rng_state()
    for _ in range(2):
        assert probe.evaluate_distributed(model, tokenizer)['metrics'] == {'niah/skipped': 1}
    assert len(caplog.records) == 1 and model.calls == 0
    assert_rng(rng)
    path.write_bytes(data.read_bytes())
    assert probe.evaluate(model, tokenizer)['status'] == 'ok'
    path.unlink()
    assert probe.evaluate(model, tokenizer)['status'] == 'skipped'


@pytest.mark.parametrize('contents', [b'broken parquet', None])
def test_bad_data_skips(tmp_path, contents):
    path = tmp_path / 'bad.parquet'
    if contents is None:
        pq.write_table(pa.table({'text': ['too short']}), path)
    else:
        path.write_bytes(contents)
    assert ShortNIAH(path, contexts=[256]).evaluate(Model(), Tokenizer())['status'] == 'skipped'


def test_paired_inputs_and_private_rng(data):
    before = rng_state()
    suite = build_suite(Tokenizer(), data, [512], [.05, .95], 2, 123)
    assert suite == build_suite(Tokenizer(), data, [512], [.05, .95], 2, 123)
    assert_rng(before)
    for cell in suite['cells']:
        for sample in cell['samples']:
            start, target = sample['target_start'], sample['target_tokens']
            assert sample['present'][start:start+len(target)] == target
            assert sample['absent'][start:start+len(target)] == target
            assert len(sample['present']) == len(sample['absent']) == 512
            assert sample['present'] != sample['absent']


def test_live_loader_and_prefetched_batch_untouched(data, tmp_path):
    from nanochat.dataloader import tokenizing_distributed_data_loader_with_state_bos_bestfit
    pq.write_table(pa.table({'text': ['abcdefgh ' * 10] * 20}), tmp_path / 'train.parquet')
    pq.write_table(pa.table({'text': ['validation']}), tmp_path / 'val.parquet')
    kwargs = dict(tokenizer=Tokenizer(), B=1, T=256, split='train', device='cpu',
                  data_dir=str(tmp_path), seq_align=256, tokenizer_batch_size=2,
                  tokenizer_threads=1, buffer_size=4, aligned_segment_ends=True, num_workers=0)
    loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(**kwargs)
    batch = next(loader)
    saved = copy.deepcopy(loader.state_dict())
    tensors = [t.clone() for t in batch[:4]]
    ShortNIAH(data, contexts=[256], depths=[.5], trials=1).evaluate(Model(), Tokenizer())
    def same(a, b):
        if isinstance(a, torch.Tensor):
            assert torch.equal(a, b)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for k in a: same(a[k], b[k])
        elif isinstance(a, (tuple, list)):
            assert len(a) == len(b)
            for x, y in zip(a, b): same(x, y)
        else:
            assert a == b
    same(saved, loader.state_dict())
    same(tensors, list(batch[:4]))
    restored = tokenizing_distributed_data_loader_with_state_bos_bestfit(**kwargs, resume_state_dict=saved)
    same(next(loader)[:4], next(restored)[:4])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_real_dism_qb_state(data):
    from nanochat.models import build_model
    from test_qb_training import spec
    model = build_model(spec()).cuda().train()
    model.init_weights()
    model.set_training_step(1000)
    model.rng_counter.fill_(37)
    model.layers[1].attn.qb.q_price.fill_(.1)
    before = copy.deepcopy(model.state_dict())
    versions = [b._version for b in model.buffers()]
    rng = rng_state()
    result = ShortNIAH(data, contexts=[512], depths=[.5], trials=1).evaluate(model, Tokenizer())
    assert result['status'] == 'ok'
    assert_rng(rng)
    assert model.training and model.config.sequence_len == 256
    assert versions == [b._version for b in model.buffers()]
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, before[name]), name


def _distributed_worker(rank, rendezvous, data_path):
    import torch.distributed as dist
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2)
    try:
        for mode in ('ok', 'missing', 'error'):
            model = Model('runtime' if mode == 'error' else None)
            path = data_path + '.missing' if mode == 'missing' else data_path
            probe = ShortNIAH(path, contexts=[256], depths=[.5], trials=1)
            rng = rng_state()
            if mode == 'error':
                with pytest.raises(RuntimeError, match='injected failure'):
                    probe.evaluate_distributed(model, Tokenizer())
            else:
                result = probe.evaluate_distributed(model, Tokenizer())
                gathered = [None, None]
                dist.all_gather_object(gathered, result)
                assert gathered[0] == gathered[1]
                assert result['status'] == ('ok' if mode == 'ok' else 'skipped')
            assert_rng(rng)
            if rank == 1 or mode == 'missing':
                assert model.calls == 0
            assert int(model.rng_counter) == 37 and model.training
    finally:
        dist.destroy_process_group()


def test_distributed_rank_zero_and_error_broadcast(data, tmp_path):
    torch.multiprocessing.spawn(_distributed_worker,
        args=((tmp_path / 'rendezvous').as_uri(), str(data)), nprocs=2, join=True)
