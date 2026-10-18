"""Vendored sliding-window DeltaNet operator.

Source: ``deltastar/sliding_delta_net_{triton,varlen}.py``.
The training operator is unchanged except for package-relative imports.
"""

from .layer import SlidingWindowDeltaNet
from .triton import sliding_window_delta_net

__all__ = ["SlidingWindowDeltaNet", "sliding_window_delta_net"]
