"""Full-hard SAM prefill plan and CPU numerical oracle."""
import numpy as np
from .build import load_prefill


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
