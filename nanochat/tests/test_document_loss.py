"""Step normalization, clipping and distributed reduction independent of CUDA."""
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from nanochat.document_loss import DocumentLossAccumulator


def documents():
    # Unequal token counts, one ignored-only and one empty document.
    return [torch.arange(n, dtype=torch.float64)*.13 + offset
            for n, offset in [(1, .2), (7, .8), (0, 0), (3, 1.1), (2, 0), (9, -.4)]]


def numerator(p, ids):
    losses = [(p[0]*documents()[i] + p[1]).square().mean()
              for i in ids if i not in (2, 4)]
    # Graph-connected zero keeps empty-rank behavior identical to CE.
    return sum(losses, p.sum()*0), torch.tensor(len(losses))


@pytest.mark.parametrize('partitions', [[[0, 1, 2, 3, 4, 5]], [[0], [1, 2], [3, 4, 5]],
                                      [[2, 4], [0, 1, 3, 5]]])
@pytest.mark.parametrize('scale', [1., 128.])
def test_accumulation_matches_whole_step_and_clipping(partitions, scale):
    p = torch.nn.Parameter(torch.tensor([.7, -.1], dtype=torch.float64))
    ref = torch.nn.Parameter(p.detach().clone())
    expected_sum, count = numerator(ref, range(6))
    expected = expected_sum/count
    expected.backward()
    step = DocumentLossAccumulator(len(partitions), 'cpu')
    for ids in partitions:
        loss, count = numerator(p, ids)
        step.add(loss, count)
        (loss*scale/len(partitions)).backward()
    actual = step.finalize([p])
    p.grad.div_(scale)  # same ordering as GradScaler.unscale_ in training
    torch.testing.assert_close(actual.double(), expected.detach(), rtol=1e-7, atol=1e-7)
    torch.testing.assert_close(p.grad, ref.grad, rtol=1e-12, atol=1e-12)
    torch.nn.utils.clip_grad_norm_([p], .1)
    torch.nn.utils.clip_grad_norm_([ref], .1)
    torch.testing.assert_close(p.grad, ref.grad, rtol=1e-12, atol=1e-12)
    with pytest.raises(RuntimeError):
        step.finalize([p])


def test_all_ignored_and_incomplete_step():
    p = torch.nn.Parameter(torch.tensor([.7, -.1]))
    step = DocumentLossAccumulator(2, 'cpu')
    with pytest.raises(RuntimeError):
        step.finalize([p])
    for ids in ([2], [4]):
        loss, count = numerator(p, ids)
        step.add(loss, count)
        (loss/2).backward()
    assert step.finalize([p]).item() == 0
    assert step.count.item() == 0
    assert torch.equal(p.grad, torch.zeros_like(p))


def _distributed_worker(rank, world, rendezvous, empty_rank):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=world,
                            timeout=timedelta(seconds=60))
    try:
        p = torch.nn.Parameter(torch.tensor([.7, -.1], dtype=torch.float64))
        extra = torch.nn.Parameter(torch.tensor([.3], dtype=torch.float64))
        unused = torch.nn.Parameter(torch.ones(1, dtype=torch.float64))
        ref = torch.nn.Parameter(p.detach().clone())
        expected_sum, expected_count = numerator(ref, range(6))
        (expected_sum/expected_count).backward()
        partitions = ([[0, 1, 3], [5]] if empty_rank else [[0], [1, 3]]) if rank == 0 else (
            [[2], [4]] if empty_rank else [[2, 4], [5]])
        step = DocumentLossAccumulator(2, 'cpu')
        for ids in partitions:
            loss, count = numerator(p, ids)
            step.add(loss, count)
            (loss/2).backward()
        # A separately unused-on-one-rank parameter must join the same collectives.
        if rank == 0:
            extra.grad = torch.ones_like(extra)
        actual = step.finalize([p, extra, unused])
        torch.testing.assert_close(actual.double(), expected_sum.detach()/expected_count,
                                   rtol=1e-7, atol=1e-7)
        torch.testing.assert_close(p.grad, ref.grad, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(extra.grad, torch.full_like(extra, .5))
        assert unused.grad is None
        torch.nn.utils.clip_grad_norm_([p], .1)
        torch.nn.utils.clip_grad_norm_([ref], .1)
        torch.testing.assert_close(p.grad, ref.grad, rtol=1e-12, atol=1e-12)
        # If an optimizer also does AVG, identical global gradients are unchanged.
        averaged = p.grad.clone()
        dist.all_reduce(averaged)
        averaged.div_(world)
        torch.testing.assert_close(averaged, p.grad)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('empty_rank', [False, True])
def test_two_rank_unequal_document_counts(tmp_path, empty_rank):
    mp.spawn(_distributed_worker,
             args=(2, 'file://' + str(tmp_path/'rendezvous'), empty_rank), nprocs=2, join=True)


@pytest.mark.parametrize('alpha', [0., .3, 1.])
@pytest.mark.parametrize('antithetic', [False, True])
def test_explicit_dual_path_counts_documents_once(alpha, antithetic):
    from types import SimpleNamespace
    from nanochat.dual_loss import dual_loss_backward
    p = torch.nn.Parameter(torch.tensor(.7, dtype=torch.float64))
    original = SimpleNamespace(rng_counter=torch.zeros((), dtype=torch.int64),
                               config=SimpleNamespace(direction_antithetic=antithetic))
    def model(x, y, *, hard_mode, direction_flip, paired_counter, loss_reduction, **kw):
        assert loss_reduction == 'document_sum'
        count = torch.tensor(y)
        return (p+x+int(hard_mode)+.1*int(direction_flip)).square()*count, count
    step = DocumentLossAccumulator(2, 'cpu')
    for x, count in ((.2, 2), (.9, 5)):
        dual_loss_backward(model, original, x, count, None, None, alpha, 2,
                           document_step=step)
    actual = step.finalize([p])
    assert original.rng_counter.item() == 2
    assert step.count.item() == 7
    q = p.detach().clone().requires_grad_()
    flips = (0, 1) if antithetic else (0,)
    expected = sum((q+x+hard+.1*flip).square()*count*weight/len(flips)
                   for x, count in ((.2, 2), (.9, 5))
                   for hard, weight in ((0, 1-alpha), (1, alpha)) for flip in flips)/7
    expected.backward()
    torch.testing.assert_close(p.grad, q.grad, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(actual.double(), expected.detach(), rtol=1e-7, atol=1e-7)
