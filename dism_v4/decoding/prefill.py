"""Offline full-hard prefill control plane and CPU vector execution mock.

Single sequence/head, labels [N], vectors [N,R]/[N,DV]. No CUDA build, no
training-extension import. No cache mutation, autograd or finite soft delta.
Exported event arrays are the handoff to a future CUDA/Triton vector backend.
"""
from functools import lru_cache
import importlib.util
from pathlib import Path
import subprocess
import sysconfig
import os
import fcntl

import numpy as np


@lru_cache(None)
def load_prefill():
    import torch  # Header location only; no libtorch link or CUDA compilation.
    root = Path(__file__).resolve().parent
    build = root / "build"
    build.mkdir(exist_ok=True)
    output = build / ("_dism_prefill" + sysconfig.get_config_var("EXT_SUFFIX"))
    sources = [root / name for name in ("prefill_bind.cpp", "prefill.hpp", "planner.hpp")]
    with (build / "prefill.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not output.exists() or output.stat().st_mtime < max(p.stat().st_mtime for p in sources):
            temporary = output.with_suffix(output.suffix + f".{os.getpid()}.tmp")
            subprocess.run([
                os.environ.get("CXX", "c++"), "-O3", "-std=c++17", "-shared", "-fPIC",
                "-I" + str(Path(torch.__file__).parent / "include"),
                "-I" + sysconfig.get_paths()["include"], str(sources[0]), "-o", str(temporary),
            ], check=True)
            temporary.replace(output)
    spec = importlib.util.spec_from_file_location("_dism_prefill", output)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class HardPrefillPlan:
    def __init__(self, queries, keys, tau, *, reset=None):
        queries, keys = np.asarray(queries), np.asarray(keys)
        for x in (queries, keys):
            if x.ndim != 1 or x.dtype.kind not in "iu" or not x.size:
                raise ValueError("labels must be nonempty 1D integer arrays")
            if x.min() < np.iinfo(np.int32).min or x.max() > np.iinfo(np.int32).max:
                raise ValueError("labels must fit int32")
        if queries.shape != keys.shape:
            raise ValueError("query/key shapes differ")
        if reset is None:
            reset = np.zeros(keys.size, dtype=bool)
        reset = np.asarray(reset)
        if reset.shape != keys.shape or reset.dtype != np.bool_:
            raise ValueError("reset must be bool [N], applied before the current query")
        self.native = load_prefill().plan(queries.tolist(), keys.tolist(), reset.astype(np.int32).tolist(), float(tau))

    def execute(self, sq, sk, value, *, dtype=np.float32):
        """CPU mock. Only per-stream R*DV scratch, plus N*DV output.

        dtype controls coefficients AND all vector arithmetic; scalar planning
        remains FP64. Inputs may be strided CPU arrays; copies are explicit here.
        """
        dtype = np.dtype(dtype)
        if dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise ValueError("mock dtype must be float32 or float64")
        arrays = [np.ascontiguousarray(x, dtype=dtype) for x in (sq, sk, value)]
        if any(not np.isfinite(x).all() for x in arrays):
            raise ValueError("vector inputs must be finite")
        fn = self.native.execute_fp64 if dtype == np.float64 else self.native.execute_fp32
        return fn(*arrays)

    def arrays(self):
        return self.native.arrays()

    def statistics(self):
        return self.native.statistics()
