import copy

import torch
import pytest
import pyarrow as pa
import pyarrow.parquet as pq

import nanochat.dataloader as dataloader_module
import nanochat.flash_attention as fa_module
from nanochat.dataloader import (
    build_varlen_metadata,
    deterministic_permutation,
    max_varlen_segments,
    tokenizing_distributed_data_loader_with_state_bos_bestfit,
)
from nanochat.flash_attention import flash_attn
from nanochat.flash_attention import HAS_FA2
from nanochat.gpt import GPT, GPTConfig
from nanochat.common import COMPUTE_DTYPE


class _FakeTokenizer:
    bos = 99

    def get_bos_token_id(self):
        return self.bos

    def encode(self, texts, prepend=None, num_threads=1):
        lengths = {"long": 7, "medium": 5, "short": 4}
        return [[prepend] + [i + 1] * (lengths[text] - 1) for i, text in enumerate(texts)]


class _DeterministicTokenizer:
    bos = 127

    def get_bos_token_id(self):
        return self.bos

    def encode(self, texts, prepend=None, num_threads=1):
        return [[prepend] + [ord(c) % 97 for c in text] for text in texts]


def _clone_batch(batch):
    return tuple(item.clone() if isinstance(item, torch.Tensor) else dict(item) for item in batch)


def _write_shuffle_dataset(path):
    # The lexicographically final file is reserved for validation.
    for file_idx in range(2):
        texts = [f"f{file_idx}-r{row_group}-d{doc}" for row_group in range(3) for doc in range(4)]
        pq.write_table(
            pa.table({"text": texts}), path / f"train-{file_idx}.parquet", row_group_size=4,
        )
    pq.write_table(pa.table({"text": ["validation"]}), path / "zz-val.parquet")


def test_shuffle_permutation_has_versioned_golden_order():
    assert deterministic_permutation(10, 1337, 0, 1) == [1, 4, 0, 6, 2, 7, 3, 8, 9, 5]
    assert deterministic_permutation(10, 1337, 1, 1, 2, 3) == [4, 3, 5, 6, 1, 7, 8, 9, 2, 0]


def test_document_source_shuffle_is_seeded_and_epoch_dependent(tmp_path):
    _write_shuffle_dataset(tmp_path)

    def first_epoch(seed):
        source = dataloader_module._DocumentBatchIterator(
            "train", None, tokenizer_batch_size=100, data_dir=str(tmp_path),
            shuffle=True, shuffle_seed=seed,
        )
        batches = [next(source) for _ in range(len(source.row_groups))]
        return batches, source

    first, source = first_epoch(17)
    repeated, _ = first_epoch(17)
    changed, _ = first_epoch(18)
    assert first == repeated
    assert first != changed
    assert {position for _, position in first} == {
        (pq_idx, rg_idx, 1) for pq_idx, rg_idx in source.row_groups
    }

    second_epoch = [next(source) for _ in range(len(source.row_groups))]
    assert {position for _, position in second_epoch} == {
        (pq_idx, rg_idx, 2) for pq_idx, rg_idx in source.row_groups
    }
    assert [position[:2] for _, position in first] != [position[:2] for _, position in second_epoch]


def test_shuffled_document_source_exact_resume_mid_row_group(tmp_path):
    _write_shuffle_dataset(tmp_path)
    kwargs = dict(
        split="train", tokenizer_batch_size=3, data_dir=str(tmp_path),
        shuffle=True, shuffle_seed=29,
    )
    source = dataloader_module._DocumentBatchIterator(resume_state_dict=None, **kwargs)
    next(source)
    state = source.state_dict()
    expected = [next(source) for _ in range(12)]

    restored = dataloader_module._DocumentBatchIterator(
        resume_state_dict=state, **kwargs,
    )
    assert [next(restored) for _ in range(12)] == expected

    with pytest.raises(ValueError, match="shuffle"):
        dataloader_module._DocumentBatchIterator(
            "train", state, tokenizer_batch_size=3, data_dir=str(tmp_path),
            shuffle=False,
        )
    with pytest.raises(ValueError, match="shuffle_seed"):
        dataloader_module._DocumentBatchIterator(
            "train", state, tokenizer_batch_size=3, data_dir=str(tmp_path),
            shuffle=True, shuffle_seed=30,
        )


def test_shuffled_ddp_stream_assigns_each_epoch_unit_once(tmp_path, monkeypatch):
    texts = [f"r{row_group}-d{doc}" for row_group in range(5) for doc in range(4)]
    pq.write_table(pa.table({"text": texts}), tmp_path / "train.parquet", row_group_size=4)
    pq.write_table(pa.table({"text": ["validation"]}), tmp_path / "zz-val.parquet")
    assigned = []
    # Five row groups intentionally do not divide evenly across three ranks.
    # Four units per rank cover global positions 0..11: two full epochs plus two
    # units, without drops or duplicate (epoch, row-group) assignments.
    for rank in range(3):
        monkeypatch.setattr(
            dataloader_module, "get_dist_info",
            lambda rank=rank: (True, rank, rank, 3),
        )
        source = dataloader_module._DocumentBatchIterator(
            "train", None, tokenizer_batch_size=100, data_dir=str(tmp_path),
            shuffle=True, shuffle_seed=31,
        )
        assigned.extend(next(source)[1] for _ in range(4))

    assert len(assigned) == len(set(assigned)) == 12
    for epoch in (1, 2):
        assert {(pq_idx, rg_idx) for pq_idx, rg_idx, item_epoch in assigned if item_epoch == epoch} == {
            (0, rg_idx) for rg_idx in range(5)
        }


def test_legacy_source_checkpoint_resumes_without_enabling_shuffle(tmp_path):
    _write_shuffle_dataset(tmp_path)
    kwargs = dict(split="train", tokenizer_batch_size=3, data_dir=str(tmp_path))
    source = dataloader_module._DocumentBatchIterator(
        resume_state_dict=None, shuffle=False, **kwargs,
    )
    next(source)
    current = source.state_dict()
    legacy = {
        key: current[key]
        for key in (
            "split", "rank", "world_size", "tokenizer_batch_size", "manifest_hash",
            "pyarrow_version", "epoch", "pq_idx", "rg_idx", "doc_offset",
        )
    }
    legacy["version"] = 3
    expected = [next(source) for _ in range(8)]

    restored = dataloader_module._DocumentBatchIterator(
        resume_state_dict=legacy, shuffle=None, **kwargs,
    )
    assert restored.shuffle is False
    assert [next(restored) for _ in range(8)] == expected


def test_validation_source_is_never_implicitly_shuffled(tmp_path):
    _write_shuffle_dataset(tmp_path)
    source = dataloader_module._DocumentBatchIterator(
        "val", None, tokenizer_batch_size=100, data_dir=str(tmp_path),
        shuffle=None, shuffle_seed=999,
    )
    assert source.shuffle is False
    assert next(source)[0] == ["validation"]


def test_legacy_outer_loader_checkpoint_keeps_exact_order(tmp_path):
    _write_shuffle_dataset(tmp_path)
    kwargs = dict(
        tokenizer=_DeterministicTokenizer(), B=2, T=32, split="train",
        device="cpu", data_dir=str(tmp_path), seq_align=8,
        tokenizer_batch_size=3, buffer_size=5, shuffle=False,
    )
    loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(**kwargs)
    for _ in range(4):
        next(loader)
    legacy = copy.deepcopy(loader.state_dict())
    legacy["version"] = 2
    for key in ("shuffle", "shuffle_seed", "shuffle_algorithm", "row_group_manifest_hash"):
        legacy["config"].pop(key)
        legacy["source"].pop(key)
    legacy["source"].pop("stream_position")
    legacy["source"]["version"] = 3
    expected = [_clone_batch(next(loader)) for _ in range(6)]

    restored_kwargs = dict(kwargs)
    restored_kwargs.pop("shuffle")
    restored = tokenizing_distributed_data_loader_with_state_bos_bestfit(
        **restored_kwargs, resume_state_dict=legacy,
    )
    actual = [_clone_batch(next(restored)) for _ in range(6)]
    for expected_batch, actual_batch in zip(expected, actual):
        for expected_item, actual_item in zip(expected_batch[:4], actual_batch[:4]):
            torch.testing.assert_close(expected_item, actual_item, rtol=0, atol=0)
        assert expected_batch[4] == actual_batch[4]


def test_aligned_loader_masks_padding_and_cross_document_targets(monkeypatch):
    def fake_document_batches(*args, **kwargs):
        while True:
            yield ["long", "medium", "short"], (0, 0, 1)

    monkeypatch.setattr(dataloader_module, "_document_batches", fake_document_batches)
    loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
        _FakeTokenizer(), B=1, T=16, split="train", device="cpu",
        buffer_size=1, seq_align=8,
    )
    inputs, targets, cu_seqlens, segment_ids, _ = next(loader)

    assert inputs.shape == targets.shape == segment_ids.shape == (1, 16)
    assert cu_seqlens.shape == (max_varlen_segments(1, 16, 8) + 1,)
    # Documents occupy [0, 7) and [8, 13); starts are aligned to 8.
    assert cu_seqlens[:5].tolist() == [0, 7, 8, 13, 16]
    assert torch.all(cu_seqlens[5:] == 16)  # zero-length compile-stability padding
    assert targets[0, 0] >= 0
    assert targets[0, 6] == -1  # document -> alignment padding
    assert targets[0, 7] == -1  # alignment padding -> next document
    assert targets[0, 12] == -1  # document -> tail padding
    assert segment_ids[0, 0] != segment_ids[0, 8]


def test_stateful_loader_exact_resume_preserves_buffer_and_row_group_offset(tmp_path):
    # Two files are needed because the final sorted shard is reserved for validation.
    texts = [chr(97 + i % 20) * (2 + i % 13) for i in range(53)]
    pq.write_table(pa.table({"text": texts}), tmp_path / "train.parquet", row_group_size=7)
    pq.write_table(pa.table({"text": ["validation"]}), tmp_path / "val.parquet")
    kwargs = dict(
        tokenizer=_DeterministicTokenizer(), B=2, T=32, split="train",
        device="cpu", data_dir=str(tmp_path), seq_align=8,
        tokenizer_threads=4, tokenizer_batch_size=3, buffer_size=5,
    )
    loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(**kwargs)
    for _ in range(5):
        next(loader)

    # Serialize through torch.save/load, not merely an in-memory Python handoff.
    state_path = tmp_path / "loader_state.pt"
    torch.save(loader.state_dict(), state_path)
    expected = [_clone_batch(next(loader)) for _ in range(8)]

    restored_state = torch.load(state_path, map_location="cpu")
    restored = tokenizing_distributed_data_loader_with_state_bos_bestfit(
        **kwargs, resume_state_dict=restored_state,
    )
    actual = [_clone_batch(next(restored)) for _ in range(8)]
    for expected_batch, actual_batch in zip(expected, actual):
        for expected_item, actual_item in zip(expected_batch[:4], actual_batch[:4]):
            torch.testing.assert_close(expected_item, actual_item, rtol=0, atol=0)
        assert expected_batch[4] == actual_batch[4]


def test_sdpa_varlen_prevents_cross_document_attention():
    old_override = fa_module._override_impl
    try:
        fa_module._override_impl = "sdpa"
        fa_module._refresh_impl_flags()
        B, T, H, D = 1, 8, 2, 4
        segment_ids = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1]], dtype=torch.int32)
        cu_seqlens, _ = build_varlen_metadata(segment_ids, max_segments=6)
        q = torch.randn(B, T, H, D)
        k = torch.randn(B, T, H, D)
        v = torch.randn(B, T, H, D)

        y1 = flash_attn.flash_attn_varlen_func(
            q, k, v, cu_seqlens, T, segment_ids, causal=True, window_size=(T, 0)
        )
        k2, v2 = k.clone(), v.clone()
        k2[:, :3].mul_(100)
        v2[:, :3].add_(100)
        y2 = flash_attn.flash_attn_varlen_func(
            q, k2, v2, cu_seqlens, T, segment_ids, causal=True, window_size=(T, 0)
        )
        torch.testing.assert_close(y1[:, 3:], y2[:, 3:])
    finally:
        fa_module._override_impl = old_override
        fa_module._refresh_impl_flags()


def test_torch_compile_does_not_recompile_when_document_count_changes():
    old_override = fa_module._override_impl
    compile_count = 0

    def counting_backend(graph_module, example_inputs):
        nonlocal compile_count
        compile_count += 1
        return graph_module.forward

    try:
        fa_module._override_impl = "sdpa"
        fa_module._refresh_impl_flags()
        torch._dynamo.reset()
        B, T, H, D = 1, 8, 2, 4

        def forward(q, k, v, cu_seqlens, segment_ids):
            return flash_attn.flash_attn_varlen_func(
                q, k, v, cu_seqlens, T, segment_ids,
                causal=True, window_size=(T, 0),
            )

        compiled = torch.compile(forward, backend=counting_backend, dynamic=False, fullgraph=True)
        q = torch.randn(B, T, H, D)
        k = torch.randn(B, T, H, D)
        v = torch.randn(B, T, H, D)
        max_segments = 8

        ids_two_docs = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]], dtype=torch.int32)
        cu_two, _ = build_varlen_metadata(ids_two_docs, max_segments)
        compiled(q, k, v, cu_two, ids_two_docs)

        ids_four_docs = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3]], dtype=torch.int32)
        cu_four, _ = build_varlen_metadata(ids_four_docs, max_segments)
        compiled(q, k, v, cu_four, ids_four_docs)

        assert compile_count == 1
    finally:
        torch._dynamo.reset()
        fa_module._override_impl = old_override
        fa_module._refresh_impl_flags()


def test_compiled_gpt_does_not_recompile_when_document_count_changes():
    old_override = fa_module._override_impl
    compile_count = 0

    def counting_backend(graph_module, example_inputs):
        nonlocal compile_count
        compile_count += 1
        return graph_module.forward

    try:
        fa_module._override_impl = "sdpa"
        fa_module._refresh_impl_flags()
        torch._dynamo.reset()
        model = GPT(GPTConfig(
            sequence_len=8, vocab_size=128, n_layer=2,
            n_head=2, n_kv_head=2, n_embd=32, window_pattern="L",
        ))
        model.init_weights()
        compiled = torch.compile(
            model, backend=counting_backend, dynamic=False, fullgraph=True
        )
        inputs = torch.randint(0, 128, (1, 8))
        targets = torch.randint(0, 128, (1, 8))
        max_segments = 8

        ids_two_docs = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]], dtype=torch.int32)
        cu_two, _ = build_varlen_metadata(ids_two_docs, max_segments)
        loss = compiled(
            inputs, targets, cu_seqlens=cu_two, segment_ids=ids_two_docs
        )
        loss.backward()
        model.zero_grad(set_to_none=True)

        ids_four_docs = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3]], dtype=torch.int32)
        cu_four, _ = build_varlen_metadata(ids_four_docs, max_segments)
        loss = compiled(
            inputs, targets, cu_seqlens=cu_four, segment_ids=ids_four_docs
        )
        loss.backward()

        assert compile_count == 1
    finally:
        torch._dynamo.reset()
        fa_module._override_impl = old_override
        fa_module._refresh_impl_flags()


@pytest.mark.skipif(
    not HAS_FA2 or not torch.cuda.is_available()
    or COMPUTE_DTYPE not in (torch.float16, torch.bfloat16),
    reason="CUDA FA2 with fp16/bf16 required",
)
def test_compiled_gpt_fa2_does_not_recompile_when_document_count_changes():
    """Exercise the real FA2 autograd.Function, including backward, under Dynamo."""
    old_override = fa_module._override_impl
    compile_count = 0

    def counting_backend(graph_module, example_inputs):
        nonlocal compile_count
        compile_count += 1
        return graph_module.forward

    try:
        fa_module._override_impl = "fa2"
        fa_module._refresh_impl_flags()
        torch._dynamo.reset()
        B, T = 2, 64
        model = GPT(GPTConfig(
            sequence_len=T, vocab_size=128, n_layer=2,
            n_head=2, n_kv_head=2, n_embd=32, window_pattern="L",
        )).cuda()
        model.init_weights()
        compiled = torch.compile(
            model, backend=counting_backend, dynamic=False, fullgraph=True
        )
        inputs = torch.randint(0, 128, (B, T), device="cuda")
        targets = torch.randint(0, 128, (B, T), device="cuda")

        for segment_len in (32, 16):
            segment_ids = (
                torch.arange(B * T, device="cuda") // segment_len
            ).view(B, T).to(torch.int32)
            cu_seqlens, _ = build_varlen_metadata(segment_ids, max_segments=16)
            loss = compiled(
                inputs, targets,
                cu_seqlens=cu_seqlens, segment_ids=segment_ids,
            )
            loss.backward()
            model.zero_grad(set_to_none=True)

        assert compile_count == 1
    finally:
        torch._dynamo.reset()
        fa_module._override_impl = old_override
        fa_module._refresh_impl_flags()
