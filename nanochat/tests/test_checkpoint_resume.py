import sys
import types
import copy
import pytest

import torch
import torch.nn.functional as F
import pyarrow as pa
import pyarrow.parquet as pq

# checkpoint_manager imports tokenizer definitions, while these tests do not use
# tokenizer training. Keep the test runnable in minimal environments without rustbpe.
if "rustbpe" not in sys.modules:
    try:
        import rustbpe  # noqa: F401
    except ImportError:
        sys.modules["rustbpe"] = types.ModuleType("rustbpe")

from nanochat.checkpoint_manager import (
    build_model as build_checkpoint_model,
    load_checkpoint,
    load_rank_training_state,
    save_checkpoint,
)
from nanochat.models import ModelSpec, build_model
from nanochat.dataloader import tokenizing_distributed_data_loader_with_state_bos_bestfit


class _Tokenizer:
    def get_bos_token_id(self):
        return 63

    def encode(self, texts, prepend=None, num_threads=1):
        return [[prepend] + [ord(char) % 61 + 1 for char in text] for text in texts]

    def get_vocab_size(self):
        return 64


class _TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(64, 12)
        self.dropout = torch.nn.Dropout(0.2)
        self.head = torch.nn.Linear(12, 64)

    def forward(self, inputs, targets):
        logits = self.head(self.dropout(self.embedding(inputs)))
        return F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=-1)


def test_rank_training_state_checkpoint_roundtrip(tmp_path):
    model = {"weight": torch.arange(6, dtype=torch.float32)}
    optimizer = {"momentum": torch.arange(3, dtype=torch.float32)}
    rank_state = {
        "version": 1,
        "dataloader": {
            "version": 3,
            "doc_buffer": {
                "tokens": torch.tensor([1, 2, 3], dtype=torch.int32),
                "offsets": torch.tensor([0, 2, 3], dtype=torch.int64),
            },
        },
        "prefetched_batch": {
            "inputs": torch.tensor([[1, 2]], dtype=torch.long),
            "targets": torch.tensor([[2, 3]], dtype=torch.long),
        },
        "python_rng": (3, (1, 2, 3), None),
    }
    metadata = {"step": 7, "exact_resume_version": 1}

    save_checkpoint(
        str(tmp_path), 7, model, optimizer, metadata,
        rank=0, rank_training_state=rank_state,
    )
    loaded_model, loaded_optimizer, loaded_metadata = load_checkpoint(
        str(tmp_path), 7, "cpu", load_optimizer=True, rank=0,
    )
    loaded_rank_state = load_rank_training_state(str(tmp_path), 7, rank=0)

    torch.testing.assert_close(loaded_model["weight"], model["weight"])
    torch.testing.assert_close(loaded_optimizer["momentum"], optimizer["momentum"])
    assert loaded_metadata == metadata
    torch.testing.assert_close(
        loaded_rank_state["dataloader"]["doc_buffer"]["tokens"],
        rank_state["dataloader"]["doc_buffer"]["tokens"],
    )
    torch.testing.assert_close(
        loaded_rank_state["prefetched_batch"]["inputs"],
        rank_state["prefetched_batch"]["inputs"],
    )


def test_missing_rank_training_state_is_legacy_checkpoint(tmp_path):
    assert load_rank_training_state(str(tmp_path), 1, rank=0) is None


def test_checkpoint_loader_builds_model_through_registry(tmp_path, monkeypatch):
    spec = ModelSpec("nanochat_v1", 1, {
        "sequence_len": 16,
        "vocab_size": 64,
        "n_layer": 1,
        "n_head": 1,
        "n_kv_head": 1,
        "n_embd": 32,
        "window_pattern": "L",
    })
    model = build_model(spec)
    model.init_weights()
    metadata = {
        "model_config": spec.config,
        "model_spec": spec.to_dict(),
        "model_fingerprint": spec.fingerprint(),
    }
    save_checkpoint(str(tmp_path), 3, model.state_dict(), None, metadata)

    import nanochat.checkpoint_manager as checkpoint_manager
    monkeypatch.setattr(checkpoint_manager, "get_tokenizer_from_spec", lambda spec: _Tokenizer())
    restored, tokenizer, restored_meta = build_checkpoint_model(
        str(tmp_path), 3, torch.device("cpu"), "train"
    )

    assert restored.model_spec == spec
    assert restored.state_dict().keys() == model.state_dict().keys()
    assert tokenizer.get_vocab_size() == 64
    assert restored_meta["model_fingerprint"] == spec.fingerprint()


@pytest.mark.parametrize('aligned_segment_ends', [False, True])
@pytest.mark.parametrize('num_workers', [0, 2])
def test_interrupted_training_matches_uninterrupted_training(tmp_path, aligned_segment_ends, num_workers):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    texts = [chr(97 + i % 20) * (2 + i % 9) for i in range(47)]
    pq.write_table(pa.table({"text": texts}), data_dir / "train.parquet", row_group_size=6)
    pq.write_table(pa.table({"text": ["validation"]}), data_dir / "val.parquet")
    loader_kwargs = dict(
        tokenizer=_Tokenizer(), B=2, T=16, split="train", device="cpu",
        data_dir=str(data_dir), seq_align=4, tokenizer_batch_size=3,
        tokenizer_threads=2, buffer_size=5,
        aligned_segment_ends=aligned_segment_ends,
        num_workers=num_workers,
    )

    torch.manual_seed(1234)
    initial_model = _TinyLM()
    training_rng = torch.get_rng_state()

    def new_run(loader_state=None):
        model = copy.deepcopy(initial_model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
            **loader_kwargs, resume_state_dict=loader_state,
        )
        return model, optimizer, loader

    def train_step(model, optimizer, loader, batch):
        x, y, _, _, _ = batch
        optimizer.zero_grad(set_to_none=True)
        model(x, y).backward()
        optimizer.step()
        return next(loader)

    # Reference run.
    torch.set_rng_state(training_rng)
    reference_model, reference_optimizer, reference_loader = new_run()
    reference_batch = next(reference_loader)
    for _ in range(6):
        reference_batch = train_step(
            reference_model, reference_optimizer, reference_loader, reference_batch
        )

    # Interrupted run: checkpoint after two optimizer steps. The current batch is
    # already prefetched, while loader.state_dict() points immediately after it.
    torch.set_rng_state(training_rng)
    interrupted_model, interrupted_optimizer, interrupted_loader = new_run()
    interrupted_batch = next(interrupted_loader)
    for _ in range(2):
        interrupted_batch = train_step(
            interrupted_model, interrupted_optimizer, interrupted_loader, interrupted_batch
        )
    rank_state = {
        "version": 1,
        "dataloader": interrupted_loader.state_dict(),
        "prefetched_batch": {
            name: tensor.clone()
            for name, tensor in zip(
                ("inputs", "targets", "cu_seqlens", "segment_ids"),
                interrupted_batch[:4],
            )
        },
        "torch_rng": torch.get_rng_state(),
    }
    save_checkpoint(
        str(tmp_path), 2, interrupted_model.state_dict(),
        interrupted_optimizer.state_dict(), {"step": 2},
        rank=0, rank_training_state=rank_state,
    )

    model_state, optimizer_state, _ = load_checkpoint(
        str(tmp_path), 2, "cpu", load_optimizer=True, rank=0,
    )
    restored_rank_state = load_rank_training_state(str(tmp_path), 2, rank=0)
    resumed_model, resumed_optimizer, resumed_loader = new_run(
        restored_rank_state["dataloader"]
    )
    resumed_model.load_state_dict(model_state)
    resumed_optimizer.load_state_dict(optimizer_state)
    torch.manual_seed(9999)  # prove that restoring the checkpoint RNG is required
    torch.set_rng_state(restored_rank_state["torch_rng"])
    saved = restored_rank_state["prefetched_batch"]
    resumed_batch = (
        saved["inputs"], saved["targets"], saved["cu_seqlens"],
        saved["segment_ids"], resumed_loader.progress_state(),
    )
    for _ in range(4):
        resumed_batch = train_step(
            resumed_model, resumed_optimizer, resumed_loader, resumed_batch
        )

    for reference, resumed in zip(reference_model.parameters(), resumed_model.parameters()):
        torch.testing.assert_close(reference, resumed, rtol=0, atol=0)
    for reference, resumed in zip(reference_batch[:4], resumed_batch[:4]):
        torch.testing.assert_close(reference, resumed, rtol=0, atol=0)
    if num_workers:
        for loader in (reference_loader, interrupted_loader, resumed_loader):
            loader.close()
