"""MXFP8 GEMM with BF16 logits and FP32 partial row logsumexp.

Build with ``python -m flash_mxfp8.build --only lse`` in the CUDA environment.
The limited configuration IDs match tk_mxfp8_gemm; partial LSE is [M,ceil(V/C)].
softcap=s transforms logits in FP32 for LSE, while retaining raw BF16 output
for the CE chain rule. None disables the transform. Plans fix s at creation.
"""

import ctypes as ct
from pathlib import Path
import re
import subprocess

import torch

from .core import FLAGS, ROOT, BUILD_ROOT
from .mxfp8_ce import _validate_softcap

OUTPUT = BUILD_ROOT / "tk_mxfp8_lse"
CONFIGS = {
    int(m[0]): tuple(map(int, m[1:]))
    for m in re.findall(
        r"^CFG\((\d+),(\d+),(\d+),(\d+),(\d+)\);",
        (ROOT / "cuda/mxfp8_lse_configs.inc").read_text(), re.M,
    )
}


def build(output=OUTPUT):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "build.log").open("w") as log:
        subprocess.run(
            ["/usr/local/cuda/bin/nvcc", *FLAGS, "-Xptxas=-v", "-shared",
             "-Xcompiler=-fPIC", "cuda/persistent_mxfp8_lse.cu", "-lcuda",
             "-o", str(output / "liblse.so")],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600,
        )


class Library:
    def __init__(self, output=OUTPUT):
        self.lib = ct.CDLL(str(Path(output) / "liblse.so"))
        self.lib.tk_mx_create_lse.argtypes = [ct.c_void_p] * 6 + [ct.c_int] * 6
        self.lib.tk_mx_create_lse.restype = ct.c_void_p
        self.lib.tk_mx_create_lse_softcap.argtypes = [ct.c_void_p] * 6 + [ct.c_int] * 6 + [ct.c_float]
        self.lib.tk_mx_create_lse_softcap.restype = ct.c_void_p
        self.lib.tk_mx_rebind_lse.argtypes = [ct.c_void_p] * 7
        self.lib.tk_mx_rebind_lse.restype = ct.c_int
        self.lib.tk_mx_launch.argtypes = [ct.c_void_p] * 2
        self.lib.tk_mx_launch.restype = ct.c_int
        self.lib.tk_mx_destroy.argtypes = [ct.c_void_p]
        self.lib.tk_mx_error.restype = ct.c_char_p

    def plan(self, x, w, xs, ws, *, config=8, out=None, partial_lse=None,
             raster_group=13, softcap=None):
        return Plan(self, x, w, xs, ws, config, out, partial_lse, raster_group, softcap)


class Plan:
    def __init__(self, owner, x, w, xs, ws, config, out, partial_lse, raster_group, softcap=None):
        self.handle = None
        self.softcap = _validate_softcap(softcap)
        if type(config) is not int or config not in CONFIGS:
            raise ValueError("Invalid LSE config")
        if type(raster_group) is not int or not 0 < raster_group < 2**31:
            raise ValueError("Invalid raster group")
        if x.ndim != 2 or w.ndim != 2 or x.shape[1] != w.shape[1]:
            raise ValueError("Expected X[M,H], W[V,H]")
        self.m, self.h = x.shape
        self.v = w.shape[0]
        if min(self.m, self.h, self.v) <= 0 or self.h % 32:
            raise ValueError("Positive dimensions and H divisible by 32 required")
        self.device = x.device
        self.config = config
        self.block_n = CONFIGS[config][1]
        self.parts = (self.v + self.block_n - 1) // self.block_n
        self.owner = owner
        if out is None:
            out = torch.empty((self.m, self.v), device=x.device, dtype=torch.bfloat16)
        if partial_lse is None:
            partial_lse = torch.empty((self.m, self.parts), device=x.device, dtype=torch.float32)
        self._validate(x, w, xs, ws, out, partial_lse)
        with torch.cuda.device(x.device):
            self.handle = owner.lib.tk_mx_create_lse_softcap(
                *[t.data_ptr() for t in (x, w, xs, ws, out, partial_lse)],
                self.m, self.v, self.h, config, 0, raster_group,
                0. if self.softcap is None else self.softcap,
            )
        if not self.handle:
            raise RuntimeError(owner.lib.tk_mx_error().decode())
        self.refs = (x, w, xs, ws, out, partial_lse)
        self.out, self.partial_lse = out, partial_lse

    def _validate(self, x, w, xs, ws, out, lse):
        shapes = (
            (self.m, self.h), (self.v, self.h),
            ((self.h + 127) // 128, (self.m + 15) // 16 * 16),
            ((self.v + 31) // 32, self.h // 32),
            (self.m, self.v), (self.m, self.parts),
        )
        dtypes = (torch.float8_e4m3fn, torch.float8_e4m3fn, torch.int32,
                  torch.uint8, torch.bfloat16, torch.float32)
        tensors = (x, w, xs, ws, out, lse)
        for t, shape, dtype in zip(tensors, shapes, dtypes):
            if tuple(t.shape) != shape or t.dtype != dtype:
                raise ValueError(f"Expected {dtype} {shape}")
            if not t.is_cuda or t.device != self.device or not t.is_contiguous():
                raise ValueError("Expected contiguous tensors on the plan CUDA device")
        if torch.cuda.get_device_capability(self.device) != (12, 0):
            raise RuntimeError("SM120 required")
        if any(t.data_ptr() % 16 for t in (x, w, xs, out)):
            raise ValueError("Unaligned TMA pointer")
        for i in (4, 5):
            lo, hi = tensors[i].data_ptr(), tensors[i].data_ptr() + tensors[i].nbytes
            if any(lo < t.data_ptr()+t.nbytes and t.data_ptr() < hi for t in tensors[:i]):
                raise ValueError("Outputs must not overlap inputs or each other")

    def rebind(self, x, w, xs, ws, *, out=None, partial_lse=None):
        out = self.out if out is None else out
        partial_lse = self.partial_lse if partial_lse is None else partial_lse
        self._validate(x, w, xs, ws, out, partial_lse)
        if self.owner.lib.tk_mx_rebind_lse(
            self.handle, *[t.data_ptr() for t in (x, w, xs, ws, out, partial_lse)]
        ):
            raise RuntimeError(self.owner.lib.tk_mx_error().decode())
        self.refs = (x, w, xs, ws, out, partial_lse)
        self.out, self.partial_lse = out, partial_lse
        return self

    def __call__(self):
        with torch.cuda.device(self.device):
            if self.owner.lib.tk_mx_launch(self.handle, torch.cuda.current_stream(self.device).cuda_stream):
                raise RuntimeError(self.owner.lib.tk_mx_error().decode())
        return self.out, self.partial_lse

    def release(self):
        self.refs = ()
        self.out = self.partial_lse = None

    def __del__(self):
        if self.handle:
            self.owner.lib.tk_mx_destroy(self.handle)


if __name__ == "__main__":
    build()
