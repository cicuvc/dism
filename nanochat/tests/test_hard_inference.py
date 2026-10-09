"""Integration checks for the experimental hard-inference harness.

The full prefill/decode parity needs a trained checkpoint and a GPU, so it is
not run here. These cover the path discovery and sampling helpers.
"""
import json
from pathlib import Path

import pytest
import torch

from nanochat.hard_inference import (default_tokenizer_dir, find_checkpoint_step,
                                     sample_next_token)


def _checkpoint(tmp_path, steps=(3, 7)):
    run = tmp_path / "run"
    ckpt = run / "base_checkpoints" / "tag"
    ckpt.mkdir(parents=True)
    for step in steps:
        (ckpt / f"meta_{step:06d}.json").write_text(json.dumps({"step": step}))
        (ckpt / f"model_{step:06d}.pt").write_bytes(b"")
    return ckpt


def test_find_checkpoint_step(tmp_path):
    ckpt = _checkpoint(tmp_path)
    assert find_checkpoint_step(ckpt) == 7
    assert find_checkpoint_step(ckpt, 3) == 3
    with pytest.raises(FileNotFoundError):
        find_checkpoint_step(ckpt, 5)
    with pytest.raises(FileNotFoundError):
        find_checkpoint_step(tmp_path / "missing")


def test_default_tokenizer_dir(tmp_path):
    ckpt = _checkpoint(tmp_path)
    (ckpt.parent.parent / "tokenizer").mkdir()
    assert default_tokenizer_dir(ckpt) == ckpt.parent.parent / "tokenizer"
    (ckpt / "tokenizer").mkdir()
    assert default_tokenizer_dir(ckpt) == ckpt / "tokenizer"


def test_sample_next_token_greedy_and_decoding():
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    assert int(sample_next_token(logits, temperature=0.0)) == 2
    torch.manual_seed(0)
    assert int(sample_next_token(logits, temperature=1.0, top_k=1)) == 2
    torch.manual_seed(0)
    assert int(sample_next_token(logits, temperature=1.0, top_p=0.01)) == 2


def test_sample_next_token_repetition_penalty():
    logits = torch.tensor([[4.0, 5.0, 0.0]])
    assert int(sample_next_token(logits, temperature=0.0)) == 1
    penalized = sample_next_token(logits, temperature=0.0, repetition_penalty=10.0, seen={1})
    assert int(penalized) == 0
