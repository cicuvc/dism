import io
from types import SimpleNamespace

import pytest
import torch

from nanochat.data_prefetch import StatefulPrefetchLoader


class Source:
    """Recycles all input buffers, like the packed data loader."""
    def __init__(self, cursor=0):
        self.cursor = cursor
        self.buffers = [torch.empty(2, 16, dtype=torch.long) for _ in range(4)]

    def __next__(self):
        for t in self.buffers:
            t.fill_(self.cursor)
        self.cursor += 1
        return (*self.buffers, self.state_dict())

    def state_dict(self):
        return {'cursor': self.cursor}

    progress_state = state_dict


def make_loader(device, state=None):
    tokenizer = SimpleNamespace(pool=SimpleNamespace(shutdown=lambda **kw: None))
    return StatefulPrefetchLoader(
        Source(state['source']['cursor'] if state else 0), tokenizer,
        device, state=state, depth=3)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_owned_buffers_and_serialized_resume(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA required for pinned staging')
    if device == 'cuda':
        torch.cuda.init()
        cuda_rng = torch.cuda.get_rng_state().clone()
    rng = torch.get_rng_state().clone()
    loader = make_loader(device)
    restored = None
    try:
        first = next(loader)
        assert first[4] == {'cursor': 1}
        with loader.condition:
            assert loader.condition.wait_for(lambda: len(loader.pending) == 3, timeout=10)
            if device == 'cuda':
                assert all(t.is_pinned() for b in loader.pending for t in b[:4])
        state = loader.state_dict()
        assert state['progress'] == {'cursor': 1}
        # Exercise serialization and pageable map_location restoration, not just
        # an in-memory queue copy.
        stream = io.BytesIO()
        torch.save(state, stream)
        stream.seek(0)
        state = torch.load(stream, map_location='cpu', weights_only=False)
        restored = make_loader(device, state)
        for i in range(1, 10):
            expected, actual = next(loader), next(restored)
            for x, y in zip(expected[:4], actual[:4]):
                assert torch.equal(x, y)
                assert (y == i).all().item()
            assert expected[4] == actual[4] == {'cursor': i + 1}
        # Later producer fills must not mutate an earlier returned batch.
        assert all((t == 0).all().item() for t in first[:4])
        assert torch.equal(rng, torch.get_rng_state())
        if device == 'cuda':
            assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
    finally:
        loader.close()
        if restored is not None:
            restored.close()
