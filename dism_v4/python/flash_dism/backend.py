"""Diagnostic registry helpers. Production shape dispatch lives in C++ frontend."""
from functools import lru_cache
import cu_flash_dism as extension


def supported_configs():
    return tuple(sorted(extension.configs))


@lru_cache(None)
def get_backend(r=32, d=64, dv=64):
    key = (r, d, dv)
    if key not in extension.configs:
        raise ValueError(f'unsupported (R,D,DV)={key}; available: {supported_configs()}')
    return extension.get_config(*key)


def backend_for(q, sq, v):
    return get_backend(sq.shape[-1], q.shape[-1], v.shape[-1])


def check_precision(fp32_output):
    if fp32_output and not extension.fp32_enabled():
        raise ValueError('FP32 validation instances are disabled; output defaults to BF16')
