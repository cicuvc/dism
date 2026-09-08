"""Process-fixed experimental scan math; set before importing dism_v2.

Use separate processes for A/B tests so saved forward states never cross modes.
The default remains full LSE. Forward and backward share this configuration.
"""
import os

TILE_LSE = os.environ.get("DISM_TILE_LSE", "full")
if TILE_LSE not in ("full", "tanh"):
    raise ValueError("DISM_TILE_LSE must be full or tanh")
LSE_SUFFIX = "" if TILE_LSE == "full" else "_tanh"
LSE_FLAGS = [] if TILE_LSE == "full" else ["-DDISM_TILE_LSE_TANH=1"]
