import pytest
import torch
import torch.nn.functional as F

from nanochat.document_loss import DocumentLossAccumulator

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def oracle(hidden, weight, labels, bounds, cap):
    logits = hidden @ weight.T
    logits = cap*torch.tanh(logits/cap)
    losses = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.flatten(),
                             ignore_index=-1, reduction='none')
    parts = []
    for start, end in zip(bounds[:-1], bounds[1:]):
        valid = labels.flatten()[start:end] != -1
        if valid.any():
            parts.append(losses[start:end][valid].mean())
    return sum(parts, logits.sum()*0), len(parts)


@pytest.mark.parametrize('compiled', [False, True])
@pytest.mark.parametrize('balance', ['document', 'token'])
def test_sum_api_accumulation_with_empty_and_ignored_documents(compiled, balance):
    from flash_dism.kernels.fused_ce_chunked import fused_ce_chunked_sum
    torch.manual_seed(827)
    hidden = torch.randn(1, 29, 16, device='cuda')*.2
    weights = torch.randn(23, 16, device='cuda')*.2
    labels = torch.randint(23, (1, 29), device='cuda')
    labels[:, 4:9] = -1  # ignored-only document
    labels[:, 18] = -1   # partial document
    bounds = [0, 4, 4, 9, 11, 29]
    inputs = hidden.clone().requires_grad_()
    w = weights.clone().requires_grad_()
    expected, count = oracle(inputs, w, labels, bounds, 3.)
    if balance == 'token':
        logits = inputs @ w.T
        expected = F.cross_entropy((3.*torch.tanh(logits/3.)).flatten(0,1),
                                   labels.flatten(), ignore_index=-1, reduction='sum')
        count = (labels != -1).sum().item()
    (expected/count).backward()
    expected_grads = inputs.grad.clone(), w.grad.clone()
    # Unequal numbers of valid documents: [1, 0, 2].
    parts = [(0, 4, [0, 4, 4]), (4, 9, [0, 5]), (9, 29, [0, 2, 20])]
    inputs.grad = None
    w.grad = None
    fn = fused_ce_chunked_sum
    if compiled:
        fn = torch.compile(fn, fullgraph=True, dynamic=True)
    step = DocumentLossAccumulator(3, 'cuda')
    for start, end, local_bounds in parts:
        cu = torch.tensor(local_bounds, device='cuda', dtype=torch.int32)
        loss, n = fn(inputs[:, start:end].contiguous(), w, labels[:, start:end].contiguous(),
                     softcap=3., cu_seqlens=cu if balance == 'document' else None, chunk_rows=4)
        assert not n.requires_grad
        step.add(loss, n)
        (loss/3).backward()
    actual = step.finalize([inputs, w])
    assert step.count.item() == count
    assert count == (3 if balance == 'document' else 23)
    torch.testing.assert_close(actual, expected.detach()/count, rtol=2e-6, atol=2e-6)
    for actual_grad, expected_grad in zip((inputs.grad, w.grad), expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=2e-5, atol=1e-7)


@pytest.mark.parametrize('architecture', ['dism_hybrid', 'swa_control'])
def test_model_document_loss_and_token_metrics(architecture):
    from nanochat.models import build_model, ModelSpec
    config = dict(sequence_len=512, vocab_size=128, n_layer=1, n_head=2, n_embd=64,
                  intermediate_size=128, head_dim=32, value_dim=32, readout_dim=16,
                  qk_vocab_size=32, anneal_steps=3000, hard_prob_max=.95)
    torch.manual_seed(28)
    model = build_model(ModelSpec(architecture, 1, config)).cuda().eval()
    model.init_weights()
    ids = torch.randint(128, (2, 512), device='cuda')
    labels = torch.randint(128, ids.shape, device='cuda')
    labels[0, :256] = -1
    labels[1, :100] = -1
    cu = torch.tensor([0, 256, 512, 512, 1024], device='cuda', dtype=torch.int32)
    with torch.no_grad():
        mean = model(ids, labels, cu_seqlens=cu)
        total, count = model(ids, labels, cu_seqlens=cu, loss_reduction='document_sum')
        per_token = model(ids, labels, cu_seqlens=cu, loss_reduction='none').flatten()
    assert count.item() == 2
    torch.testing.assert_close(mean, total/count, rtol=1e-6, atol=1e-6)
    assert per_token.shape == (1024,)
    assert torch.count_nonzero(per_token[labels.flatten() == -1]) == 0
    expected = (per_token[256:512].mean() + per_token[612:1024].mean())/2
    torch.testing.assert_close(mean, expected, rtol=.002, atol=.005)


@pytest.mark.parametrize('architecture', ['dism_hybrid', 'swa_control'])
def test_model_token_loss(architecture):
    from nanochat.models import build_model, ModelSpec
    config = dict(sequence_len=512, vocab_size=128, n_layer=1, n_head=2, n_embd=64,
                  intermediate_size=128, head_dim=32, value_dim=32, readout_dim=16,
                  qk_vocab_size=32, loss_balance='token')
    model = build_model(ModelSpec(architecture, 1, config)).cuda().eval()
    model.init_weights()
    assert not model.document_balanced_loss
    assert model.count_normalized_loss and model.loss_sum_reduction == 'token_sum'
    ids = torch.randint(128, (1,512), device='cuda')
    labels = torch.randint(128, ids.shape, device='cuda')
    labels[:,:100] = -1
    cu = torch.tensor([0,256,512],device='cuda',dtype=torch.int32)
    with torch.no_grad():
        mean = model(ids,labels,cu_seqlens=cu)
        total,count = model(ids,labels,cu_seqlens=cu,loss_reduction='token_sum')
        per_token = model(ids,labels,cu_seqlens=cu,loss_reduction='none')
    assert count.item() == 412
    torch.testing.assert_close(mean,total/count,rtol=1e-6,atol=1e-6)
    torch.testing.assert_close(mean,per_token.flatten()[100:].mean(),rtol=.002,atol=.005)
