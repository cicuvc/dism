"""Full-hard SAM inference, independent of v4 training operators.

Prefill: parallel CPU planning and batched Triton vector execution.
Decoding: CPU or GPU planning, CUDA readout, periodic CPU/GPU rebuilds.
These explicit core APIs do not replace model projections or FLA caches.
"""
from .api import HardDismDecoder
from .runtime import NativeDecodeCache
from .gpu_runtime import GpuPlannerCache
from .prefill import HardPrefillPlan
from .prefill_batch import HardDismPrefill, PreparedHardPrefill

__all__ = ['HardDismDecoder', 'NativeDecodeCache', 'GpuPlannerCache',
           'HardPrefillPlan', 'HardDismPrefill', 'PreparedHardPrefill']
