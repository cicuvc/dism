import pytest
import torch
from torch.nn import functional as F
from transformers import AutoConfig, AutoModelForCausalLM
from flash_dism import DismConfig, DismForCausalLM, DismSwaAttention
from flash_dism.modeling_dism import next_token_labels


def tiny_config(**kwargs):
    return DismConfig(vocab_size=128, hidden_size=32, num_hidden_layers=2,
                      num_heads=2, head_dim=32, value_dim=32, readout_dim=16,
                      qk_vocab_size=16, intermediate_size=64,
                      bos_token_id=1, eos_token_id=2, pad_token_id=None, **kwargs)


def test_label_boundaries():
    labels = torch.arange(8).reshape(1, 8)
    cu = torch.tensor([0, 3, 3, 8], dtype=torch.int32)
    torch.testing.assert_close(next_token_labels(labels, cu_seqlens=cu),
                              torch.tensor([[1, 2, -100, 4, 5, 6, 7, -100]]))
    mask = torch.tensor([[0, 0, 1, 1, 1, 1, 0, 0]])
    torch.testing.assert_close(next_token_labels(labels, mask),
                              torch.tensor([[-100, -100, 3, 4, 5, -100, -100, -100]]))


def test_cpu_cache_and_save_load(tmp_path):
    torch.manual_seed(10)
    model = DismForCausalLM(tiny_config()).eval()
    assert isinstance(model.model.layers[0].attn, DismSwaAttention)
    assert model.lm_head.weight is not model.get_input_embeddings().weight
    ids = torch.randint(0, 128, (1, 9))
    direction = torch.tensor([[True, False]])
    whole = model(ids, direction=direction, use_cache=False, output_hidden_states=True)
    first = model(ids[:, :5], direction=direction)
    last = model(ids[:, 5:], past_key_values=first.past_key_values)
    torch.testing.assert_close(torch.cat((first.logits, last.logits), 1), whole.logits, atol=2e-6, rtol=2e-5)
    assert last.past_key_values.get_seq_length(0) == last.past_key_values.get_seq_length(1) == 9
    assert len(whole.hidden_states) == 3
    model.save_pretrained(tmp_path)
    loaded = AutoModelForCausalLM.from_pretrained(tmp_path).eval()
    torch.testing.assert_close(loaded(ids, direction=direction, use_cache=False).logits, whole.logits)
    assert isinstance(AutoConfig.from_pretrained(tmp_path), DismConfig)
    no_decay = {id(p) for p in model.optimizer_param_groups(.01)[1]['params']}
    for layer in model.model.layers:
        assert id(layer.attn.q_vocab) in no_decay
        assert id(layer.attn.sq_proj.weight) not in no_decay
    loaded_no_decay = {id(p) for p in loaded.optimizer_param_groups(.01)[1]['params']}
    assert id(loaded.model.layers[0].attn.q_vocab) in loaded_no_decay


def test_cpu_loss_and_tuple_interface():
    model = DismForCausalLM(tiny_config(attention_type='dism')).eval()
    ids = torch.tensor([[1, 3, 4, 5, 6]])
    direction = torch.tensor([[True, False]])
    output = model(ids, labels=ids, direction=direction, use_cache=False)
    expected = F.cross_entropy(output.logits[:, :-1].flatten(0, 1), ids[:, 1:].flatten())
    torch.testing.assert_close(output.loss, expected)
    empty = model(ids, labels=torch.full_like(ids, -100), direction=direction,
                  use_cache=False, return_dict=False, return_logits=False)
    assert empty[0] == 0 and empty[1] is None
    embeds = model.get_input_embeddings()(ids)
    torch.testing.assert_close(model(inputs_embeds=embeds, direction=direction, use_cache=False).logits, output.logits)


def test_generate():
    model = DismForCausalLM(tiny_config()).eval()
    ids = torch.tensor([[1, 5, 7]])
    result = model.generate(ids, max_new_tokens=3, do_sample=False, eos_token_id=None,
                             pad_token_id=0)
    assert result.shape == (1, 6)
    sampled = model.generate(ids, max_new_tokens=2, do_sample=True, temperature=.8,
                              top_p=.9, eos_token_id=None, pad_token_id=0)
    assert sampled.shape == (1, 5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA')
@pytest.mark.parametrize('packed', [False, True])
def test_cuda_loss_gradients(packed):
    torch.manual_seed(31)
    model = DismForCausalLM(tiny_config()).cuda().train()
    length = 512 if packed else 256
    ids = torch.randint(0, 128, (1, length), device='cuda')
    options = dict(hard_prob=.5, hard_seed=7, direction=torch.tensor([[True, False]], device='cuda'))
    if packed:
        options['cu_seqlens'] = torch.tensor([0, 256, 512], device='cuda', dtype=torch.int32)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        output = model(ids, labels=ids, output_hidden_states=True, **options)
    assert output.logits is None and output.past_key_values is None
    assert torch.isfinite(output.loss)
    hidden = output.hidden_states[-1].detach().to(torch.bfloat16)
    logits = F.linear(hidden.float(), model.lm_head.weight.detach().to(torch.bfloat16).float())
    logits = 30 * (logits/30).tanh()
    targets = next_token_labels(ids, cu_seqlens=options.get('cu_seqlens'))
    expected = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
    torch.testing.assert_close(output.loss, expected, atol=.005, rtol=.002)
    output.loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    assert model.lm_head.weight.grad.abs().sum() > 0
    assert model.get_input_embeddings().weight.grad.abs().sum() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA')
def test_checkpoint_replay():
    torch.manual_seed(41)
    model = DismForCausalLM(tiny_config(attention_type='dism')).cuda().train()
    ids = torch.randint(0, 128, (1, 256), device='cuda')
    options = dict(hard_prob=.5, hard_seed=123, direction=torch.tensor([[True, False]], device='cuda'))
    with torch.autocast('cuda', dtype=torch.bfloat16):
        expected = model(ids, labels=ids, **options).loss
    expected.backward()
    gradients = {name: p.grad.clone() for name, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    model.gradient_checkpointing_enable()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        actual = model(ids, labels=ids, **options).loss
    actual.backward()
    torch.testing.assert_close(actual, expected)
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter.grad, gradients[name], atol=3e-4, rtol=.03)
