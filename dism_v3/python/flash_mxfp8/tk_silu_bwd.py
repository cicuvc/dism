"""Torch adapter for the all-BF16 elementwise SwiGLU backward kernel."""

import ctypes as ct
from pathlib import Path
import subprocess

import torch

from .core import BUILD_ROOT, ROOT, FLAGS

OUTPUT = BUILD_ROOT / "tk_silu_bwd"


def build():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for source,library in [
        ("cuda/silu_bwd_elementwise.cu", "libsilu_bwd.so"),
        ("cuda/silu_bwd_quantize.cu", "libsilu_bwd_quantize.so"),
    ]:
        command = [
            "/usr/local/cuda/bin/nvcc",
            *FLAGS,
            "-Xptxas=-v",
            "-shared",
            "-Xcompiler=-fPIC",
            source,
            "-lcuda",
            "-o",
            str(OUTPUT / library),
        ]
        with (OUTPUT / f"{library}.build.log").open("w") as log:
            subprocess.run(
                command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=300,
            )


class SiluBwdLibrary:
    def __init__(self):
        self.lib = ct.CDLL(str(OUTPUT / "libsilu_bwd.so"))
        self.lib.tk_silu_bwd_create.argtypes = [ct.c_void_p] * 5 + [ct.c_int64]
        self.lib.tk_silu_bwd_create.restype = ct.c_void_p
        self.lib.tk_silu_bwd_launch.argtypes = [ct.c_void_p, ct.c_void_p]
        self.lib.tk_silu_bwd_launch.restype = ct.c_int
        self.lib.tk_silu_bwd_info.argtypes = [ct.c_void_p, ct.POINTER(ct.c_int)]
        self.lib.tk_silu_bwd_destroy.argtypes = [ct.c_void_p]
        self.lib.tk_silu_bwd_error.restype = ct.c_char_p

    def plan(self, dy, half_gate, up, dgate=None, dup=None):
        return SiluBwdPlan(self, dy, half_gate, up, dgate, dup)


class SiluBwdPlan:
    def __init__(self, owner, dy, half_gate, up, dgate, dup):
        if dy.shape != half_gate.shape or dy.shape != up.shape:
            raise ValueError("dy, half_gate, and up must have equal shapes")
        if dy.dtype != torch.bfloat16 or half_gate.dtype != dy.dtype or up.dtype != dy.dtype:
            raise ValueError("Expected BF16 inputs")
        if dgate is None:
            dgate = torch.empty_like(dy)
        if dup is None:
            dup = torch.empty_like(dy)
        if dgate.shape != dy.shape or dup.shape != dy.shape:
            raise ValueError("Outputs must have the same shape as dy")
        tensors = [dy, half_gate, up, dgate, dup]
        if not dy.is_cuda or any(
            tensor.dtype != torch.bfloat16
            or tensor.device != dy.device
            or not tensor.is_contiguous()
            for tensor in tensors
        ):
            raise ValueError("Expected contiguous BF16 tensors on one CUDA device")
        if any(tensor.data_ptr() % 16 for tensor in tensors):
            raise ValueError("Tensor pointers must be 16-byte aligned")
        for index, output in enumerate([dgate, dup]):
            lo = output.data_ptr()
            hi = lo + output.numel() * output.element_size()
            if any(
                lo < tensor.data_ptr() + tensor.numel() * tensor.element_size()
                and tensor.data_ptr() < hi
                for tensor in [dy, half_gate, up, *([dgate] if index else [])]
            ):
                raise ValueError("Outputs must not overlap inputs or each other")
        self.owner = owner
        self.dy, self.half_gate, self.up = dy, half_gate, up
        self.dgate, self.dup = dgate, dup
        self.handle = owner.lib.tk_silu_bwd_create(
            dy.data_ptr(),
            half_gate.data_ptr(),
            up.data_ptr(),
            dgate.data_ptr(),
            dup.data_ptr(),
            dy.numel(),
        )
        if not self.handle:
            raise RuntimeError(self.owner.lib.tk_silu_bwd_error().decode())
        info = (ct.c_int * 2)()
        owner.lib.tk_silu_bwd_info(self.handle, info)
        self.info = {"grid": info[0], "threads": info[1]}

    def __call__(self):
        status = self.owner.lib.tk_silu_bwd_launch(
            self.handle,
            torch.cuda.current_stream(self.dy.device).cuda_stream,
        )
        if status:
            raise RuntimeError(self.owner.lib.tk_silu_bwd_error().decode())
        return self.dgate, self.dup

    def __del__(self):
        if getattr(self, "handle", None):
            self.owner.lib.tk_silu_bwd_destroy(self.handle)
            self.handle = None


class SiluBwdQuantizeLibrary:
    def __init__(self):
        self.lib = ct.CDLL(str(OUTPUT / "libsilu_bwd_quantize.so"))
        self.lib.tk_silu_bwd_q_create.argtypes = [ct.c_void_p] * 11 + [
            ct.c_int,
            ct.c_int,
        ]
        self.lib.tk_silu_bwd_q_create.restype = ct.c_void_p
        self.lib.tk_silu_bwd_q_launch.argtypes = [ct.c_void_p, ct.c_void_p]
        self.lib.tk_silu_bwd_q_launch.restype = ct.c_int
        self.lib.tk_silu_bwd_q_rebind.argtypes = [ct.c_void_p] * 12
        self.lib.tk_silu_bwd_q_rebind.restype = ct.c_int
        self.lib.tk_silu_bwd_q_info.argtypes = [ct.c_void_p, ct.POINTER(ct.c_int)]
        self.lib.tk_silu_bwd_q_destroy.argtypes = [ct.c_void_p]
        self.lib.tk_silu_bwd_q_error.restype = ct.c_char_p
        if hasattr(self.lib, "tk_silu_bwd_q_create_direct"):
            self.lib.tk_silu_bwd_q_create_direct.argtypes = [ct.c_void_p] * 5 + [
                ct.c_int,
                ct.c_int,
            ]
            self.lib.tk_silu_bwd_q_create_direct.restype = ct.c_void_p
        if hasattr(self.lib, "tk_silu_bwd_q_create_direct_transposed"):
            self.lib.tk_silu_bwd_q_create_direct_transposed.argtypes = [
                ct.c_void_p
            ] * 5 + [ct.c_int, ct.c_int]
            self.lib.tk_silu_bwd_q_create_direct_transposed.restype = ct.c_void_p

    def plan(self, dy, half_gate, up):
        return SiluBwdQuantizePlan(self, dy, half_gate, up)

    def plan_direct(self, x, *, transpose=False):
        return SiluBwdDirectQuantizePlan(self, x, transpose=transpose)


class SiluBwdQuantizePlan:
    """Fuse dG/dU computation with row- and column-wise MXFP8 quantization."""

    def __init__(self, owner, dy, half_gate, up):
        if dy.shape != half_gate.shape or dy.shape != up.shape:
            raise ValueError("dy, half_gate, and up must have equal shapes")
        if (
            dy.dtype != torch.bfloat16
            or half_gate.dtype != dy.dtype
            or up.dtype != dy.dtype
        ):
            raise ValueError("Expected BF16 inputs")
        m, n = dy.shape
        if min(m, n) <= 0 or n % 32:
            raise ValueError("N must be a positive multiple of 32")
        inputs = [dy, half_gate, up]
        if not dy.is_cuda or any(
            tensor.device != dy.device or not tensor.is_contiguous()
            for tensor in inputs
        ):
            raise ValueError("Expected contiguous tensors on one CUDA device")

        mp = (m + 127) // 128 * 128
        mp16 = (m + 15) // 16 * 16
        row_scale_shape = ((n + 127) // 128, mp16)
        col_scale_shape = (mp // 128, n)
        self.gate_row = torch.empty((m, n), device=dy.device, dtype=torch.float8_e4m3fn)
        self.gate_row_scale = torch.zeros(
            row_scale_shape, device=dy.device, dtype=torch.int32
        )
        self.up_row = torch.empty_like(self.gate_row)
        self.up_row_scale = torch.zeros_like(self.gate_row_scale)
        self.gate_col = torch.empty((n, mp), device=dy.device, dtype=torch.float8_e4m3fn)
        self.gate_col_scale = torch.zeros(
            col_scale_shape, device=dy.device, dtype=torch.int32
        )
        self.up_col = torch.empty_like(self.gate_col)
        self.up_col_scale = torch.zeros_like(self.gate_col_scale)
        outputs = [
            self.gate_row,
            self.gate_row_scale,
            self.up_row,
            self.up_row_scale,
            self.gate_col,
            self.gate_col_scale,
            self.up_col,
            self.up_col_scale,
        ]
        tensors = inputs + outputs
        if any(tensor.data_ptr() % 16 for tensor in tensors):
            raise ValueError("Tensor pointers must be 16-byte aligned")
        for index, output in enumerate(outputs):
            lo = output.data_ptr()
            hi = lo + output.numel() * output.element_size()
            if any(
                lo < tensor.data_ptr() + tensor.numel() * tensor.element_size()
                and tensor.data_ptr() < hi
                for tensor in inputs + outputs[:index]
            ):
                raise ValueError("Outputs must not overlap inputs or each other")

        self.owner = owner
        self.dy, self.half_gate, self.up = dy, half_gate, up
        self.m, self.n, self.device = m, n, dy.device
        self.handle = owner.lib.tk_silu_bwd_q_create(
            dy.data_ptr(),
            half_gate.data_ptr(),
            up.data_ptr(),
            self.gate_row.data_ptr(),
            self.gate_row_scale.data_ptr(),
            self.up_row.data_ptr(),
            self.up_row_scale.data_ptr(),
            self.gate_col.data_ptr(),
            self.gate_col_scale.data_ptr(),
            self.up_col.data_ptr(),
            self.up_col_scale.data_ptr(),
            m,
            n,
        )
        if not self.handle:
            raise RuntimeError(owner.lib.tk_silu_bwd_q_error().decode())
        info = (ct.c_int * 3)()
        owner.lib.tk_silu_bwd_q_info(self.handle, info)
        self.info = {
            "blocks": info[0],
            "threads": info[1],
            "shared_bytes": info[2],
            "padded_m": mp,
        }

    def __call__(self):
        status = self.owner.lib.tk_silu_bwd_q_launch(
            self.handle,
            torch.cuda.current_stream(self.dy.device).cuda_stream,
        )
        if status:
            raise RuntimeError(self.owner.lib.tk_silu_bwd_q_error().decode())
        return (
            self.gate_row,
            self.gate_row_scale,
            self.up_row,
            self.up_row_scale,
            self.gate_col,
            self.gate_col_scale,
            self.up_col,
            self.up_col_scale,
        )

    def reallocate_outputs(self):
        mp = (self.m + 127) // 128 * 128
        mp16 = (self.m + 15) // 16 * 16
        n = self.n
        self.gate_row = torch.empty(
            (self.m, self.n), device=self.device, dtype=torch.float8_e4m3fn
        )
        self.up_row = torch.empty_like(self.gate_row)
        self.gate_col = torch.empty(
            (n, mp), device=self.device, dtype=torch.float8_e4m3fn
        )
        self.up_col = torch.empty_like(self.gate_col)
        return self

    def rebind(self, dy, half_gate, up):
        if (
            dy.shape != (self.m, self.n)
            or half_gate.shape != dy.shape
            or up.shape != dy.shape
        ):
            raise ValueError("dy, half_gate, and up must keep the plan shape")
        if (
            dy.dtype != torch.bfloat16
            or half_gate.dtype != dy.dtype
            or up.dtype != dy.dtype
        ):
            raise ValueError("Expected BF16 inputs")
        inputs = [dy, half_gate, up]
        if any(
            tensor.device != self.device or not tensor.is_contiguous()
            for tensor in inputs
        ):
            raise ValueError("Expected contiguous tensors on one CUDA device")
        self.dy, self.half_gate, self.up = dy, half_gate, up
        status = self.owner.lib.tk_silu_bwd_q_rebind(
            self.handle,
            dy.data_ptr(),
            half_gate.data_ptr(),
            up.data_ptr(),
            self.gate_row.data_ptr(),
            self.gate_row_scale.data_ptr(),
            self.up_row.data_ptr(),
            self.up_row_scale.data_ptr(),
            self.gate_col.data_ptr(),
            self.gate_col_scale.data_ptr(),
            self.up_col.data_ptr(),
            self.up_col_scale.data_ptr(),
        )
        if status:
            raise RuntimeError(self.owner.lib.tk_silu_bwd_q_error().decode())
        return self

    def release(self):
        self.dy = self.half_gate = self.up = None
        self.gate_row = self.up_row = None
        self.gate_col = self.up_col = None

    def __del__(self):
        if getattr(self, "handle", None):
            self.owner.lib.tk_silu_bwd_q_destroy(self.handle)
            self.handle = None


class SiluBwdDirectQuantizePlan:
    """Direct BF16 -> dual-direction MXFP8 using the CUDA pipeline.

    The row result is always ``[M,K]``. The column result is token-major
    ``[MP,K]`` by default, or feature-major ``[K,MP]`` with ``transpose=True``
    for wgrad's A operand.
    """

    def __init__(self, owner, x, *, transpose=False):
        if x.ndim != 2 or x.dtype != torch.bfloat16 or not x.is_cuda:
            raise ValueError("Expected CUDA BF16 X[M,K]")
        if not x.is_contiguous():
            raise ValueError("Expected contiguous CUDA BF16 X[M,K]")
        m, k = x.shape
        if k % 32:
            raise ValueError("K must be a multiple of 32")
        if transpose and not hasattr(
            owner.lib, "tk_silu_bwd_q_create_direct_transposed"
        ):
            raise RuntimeError("Rebuild CUDA library for transposed quantization")
        self.x = x
        self.owner = owner
        self.m, self.k = m, k
        self.device = x.device
        self.transpose = bool(transpose)
        self.mp = (m + 127) // 128 * 128
        self.mp16 = (m + 15) // 16 * 16
        self.sp = (k + 15) // 16 * 16
        self.row_fp8 = torch.empty((m, k), device=x.device, dtype=torch.float8_e4m3fn)
        self.row_scale = torch.zeros(
            ((k + 127) // 128, self.mp16),
            device=x.device,
            dtype=torch.int32,
        )
        column_shape = (k, self.mp) if transpose else (self.mp, k)
        self.column_fp8 = torch.empty(
            column_shape, device=x.device, dtype=torch.float8_e4m3fn
        )
        self.column_scale = torch.zeros(
            (self.mp // 128, self.sp),
            device=x.device,
            dtype=torch.int32,
        )
        create = (
            owner.lib.tk_silu_bwd_q_create_direct_transposed
            if transpose
            else owner.lib.tk_silu_bwd_q_create_direct
        )
        self.handle = create(
            x.data_ptr(),
            self.row_fp8.data_ptr(),
            self.row_scale.data_ptr(),
            self.column_fp8.data_ptr(),
            self.column_scale.data_ptr(),
            m,
            k,
        )
        if not self.handle:
            raise RuntimeError(owner.lib.tk_silu_bwd_q_error().decode())
        info = (ct.c_int * 3)()
        owner.lib.tk_silu_bwd_q_info(self.handle, info)
        self.info = {
            "blocks": info[0],
            "threads": info[1],
            "shared_bytes": info[2],
        }

    def __call__(self):
        status = self.owner.lib.tk_silu_bwd_q_launch(
            self.handle,
            torch.cuda.current_stream(self.x.device).cuda_stream,
        )
        if status:
            raise RuntimeError(self.owner.lib.tk_silu_bwd_q_error().decode())
        return self.row_fp8, self.row_scale, self.column_fp8, self.column_scale

    def rebind(self, x):
        if x.shape != (self.m, self.k) or x.dtype != torch.bfloat16:
            raise ValueError("Expected BF16 X with the plan shape")
        if not x.is_cuda or not x.is_contiguous() or x.device != self.device:
            raise ValueError("Expected contiguous CUDA BF16 X[M,K]")
        self.x = x
        status = self.owner.lib.tk_silu_bwd_q_rebind(
            self.handle,
            x.data_ptr(),
            x.data_ptr(),
            x.data_ptr(),
            self.row_fp8.data_ptr(),
            self.row_scale.data_ptr(),
            self.row_fp8.data_ptr(),
            self.row_scale.data_ptr(),
            self.column_fp8.data_ptr(),
            self.column_scale.data_ptr(),
            self.column_fp8.data_ptr(),
            self.column_scale.data_ptr(),
        )
        if status:
            raise RuntimeError(self.owner.lib.tk_silu_bwd_q_error().decode())
        return self

    def reallocate_outputs(self, x=None):
        self.row_fp8 = torch.empty(
            (self.m, self.k), device=self.device, dtype=torch.float8_e4m3fn
        )
        column_shape = (self.k, self.mp) if self.transpose else (self.mp, self.k)
        self.column_fp8 = torch.empty(
            column_shape, device=self.device, dtype=torch.float8_e4m3fn
        )
        return self

    def release(self):
        self.x = None
        self.row_fp8 = self.column_fp8 = None

    def __del__(self):
        if getattr(self, "handle", None):
            self.owner.lib.tk_silu_bwd_q_destroy(self.handle)
            self.handle = None
