"""Process-fixed experimental scan math; set before importing dism_v2.

Use separate processes for A/B tests so saved forward states never cross modes.
The default remains full LSE. Forward and backward share this configuration.
"""
import os

TILE_LSE = os.environ.get("DISM_TILE_LSE", "full")
if TILE_LSE not in ("full", "tanh", "tanh_finite"):
    raise ValueError("DISM_TILE_LSE must be full, tanh or tanh_finite")
LSE_SUFFIX = "" if TILE_LSE == "full" else "_" + TILE_LSE
LSE_FLAGS = [] if TILE_LSE == "full" else ["-DDISM_TILE_LSE_TANH=1"]
if TILE_LSE == "tanh_finite":
    LSE_FLAGS += ["-DDISM_FINITE_SENTINEL=1"]
