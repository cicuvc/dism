"""Standalone PyTorch oracles for the DISM v4 design (no CUDA extension)."""

from .reference import dism_v4_from_embeddings, dism_v4_ref

__all__ = ["dism_v4_ref", "dism_v4_from_embeddings", "dism_v4_blocked",
           "dism_v4_blocked_from_embeddings"]


def __getattr__(name):
    # Keep the CPU oracle importable without Triton or a v3 extension.
    if name in ("dism_v4_blocked", "dism_v4_blocked_from_embeddings"):
        from . import blocked
        return getattr(blocked, name)
    raise AttributeError(name)
