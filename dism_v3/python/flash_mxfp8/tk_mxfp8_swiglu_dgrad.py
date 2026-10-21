"""Fused prequantized MXFP8 dX for up(x)*SiLU(gate(x)).

Computes dX[M,K] = dG[M,N] @ Wg[N,K] + dU[M,N] @ Wu[N,K].
The gradient matrices use packed per-row/N32 scales and the weights use the
forward 32x32 scale grid. FP32 accumulation is written once as BF16.
"""

import ctypes as ct
from pathlib import Path
import re
import subprocess

import torch

from .core import BUILD_ROOT, ROOT, FLAGS

OUTPUT = BUILD_ROOT / "tk_mxfp8_swiglu_dgrad"
CONFIGS = [
    tuple(map(int, match))
    for match in re.findall(
        r"^CFG\((\d+),(\d+),(\d+),(\d+),(\d+)\);",
        (ROOT / "cuda/mxfp8_swiglu_dgrad_configs.inc").read_text(),
        re.MULTILINE,
    )
]


def build():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    command = [
        "/usr/local/cuda/bin/nvcc",
        *FLAGS,
        "-Xptxas=-v",
        "-shared",
        "-Xcompiler=-fPIC",
        "cuda/persistent_mxfp8_swiglu_dgrad.cu",
        "-lcuda",
        "-o",
        str(OUTPUT / "libswd.so"),
    ]
    with (OUTPUT / "build.log").open("w") as log:
        subprocess.run(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=600,
        )


class SwiGLUDgradLibrary:
    def __init__(self):
        self.lib = ct.CDLL(str(OUTPUT / "libswd.so"))
        self.lib.tk_swd_create.argtypes = [ct.c_void_p] * 9 + [ct.c_int] * 6
        self.lib.tk_swd_create.restype = ct.c_void_p
        self.lib.tk_swd_launch.argtypes = [ct.c_void_p, ct.c_void_p]
        self.lib.tk_swd_launch.restype = ct.c_int
        self.lib.tk_swd_rebind.argtypes = [ct.c_void_p] * 10
        self.lib.tk_swd_rebind.restype = ct.c_int
        self.lib.tk_swd_info.argtypes = [ct.c_void_p, ct.POINTER(ct.c_int)]
        self.lib.tk_swd_destroy.argtypes = [ct.c_void_p]
        self.lib.tk_swd_error.restype = ct.c_char_p

    def plan(
        self,
        gate,
        up,
        weight_gate,
        weight_up,
        gate_scale,
        up_scale,
        weight_gate_scale,
        weight_up_scale,
        config=6,
        out=None,
        sms=0,
        raster_group=1,
    ):
        return SwiGLUDgradPlan(
            self,
            gate,
            up,
            weight_gate,
            weight_up,
            gate_scale,
            up_scale,
            weight_gate_scale,
            weight_up_scale,
            config,
            out,
            sms,
            raster_group,
        )


class SwiGLUDgradPlan:
    def __init__(
        self,
        owner,
        gate,
        up,
        weight_gate,
        weight_up,
        gate_scale,
        up_scale,
        weight_gate_scale,
        weight_up_scale,
        config,
        out,
        sms,
        raster_group,
    ):
        self.handle = None
        if type(config) is not int or not 0 <= config < len(CONFIGS):
            raise ValueError("Invalid configuration")
        if (
            type(raster_group) is not int
            or raster_group <= 0
            or type(sms) is not int
            or sms < 0
        ):
            raise ValueError("Invalid raster group or SM count")
        if (
            gate.ndim != 2
            or up.shape != gate.shape
            or weight_gate.ndim != 2
            or weight_up.shape != weight_gate.shape
            or gate.shape[1] != weight_gate.shape[0]
        ):
            raise ValueError("Expected dG/dU[M,N], Wg/Wu[N,K]")
        m, n = gate.shape
        k = weight_gate.shape[1]
        if min(m, n, k) <= 0:
            raise ValueError("Invalid matrix dimensions")
        if (
            gate.dtype != torch.float8_e4m3fn
            or up.dtype != gate.dtype
            or weight_gate.dtype != gate.dtype
            or weight_up.dtype != gate.dtype
        ):
            raise ValueError("Expected E4M3 gradient and weight matrices")
        row_scale_shape = ((n + 127) // 128, (m + 15) // 16 * 16)
        weight_scale_shape = ((n + 31) // 32, k // 32)
        for scale in [gate_scale, up_scale]:
            if scale.dtype != torch.int32 or scale.shape != row_scale_shape:
                raise ValueError("Invalid packed gradient scale")
        for scale in [weight_gate_scale, weight_up_scale]:
            if scale.dtype != torch.uint8 or scale.shape != weight_scale_shape:
                raise ValueError("Invalid weight scale")
        if out is None:
            out = torch.empty((m, k), device=gate.device, dtype=torch.bfloat16)
        if (
            out.shape != (m, k)
            or out.dtype != torch.bfloat16
            or out.device != gate.device
            or not out.is_contiguous()
        ):
            raise ValueError("Expected BF16 dX[M,K]")
        inputs = [
            gate,
            up,
            weight_gate,
            weight_up,
            gate_scale,
            up_scale,
            weight_gate_scale,
            weight_up_scale,
        ]
        tensors = inputs + [out]
        if not gate.is_cuda or any(
            tensor.device != gate.device or not tensor.is_contiguous()
            for tensor in tensors
        ):
            raise ValueError("Expected contiguous tensors on one CUDA device")
        if any(tensor.data_ptr() % 16 for tensor in tensors):
            raise ValueError("Tensor pointers must be 16-byte aligned")
        lo, hi = out.data_ptr(), out.data_ptr() + out.numel() * out.element_size()
        if any(
            lo < tensor.data_ptr() + tensor.numel() * tensor.element_size()
            and tensor.data_ptr() < hi
            for tensor in inputs
        ):
            raise ValueError("dX aliases an input")

        self.owner = owner
        self.gate, self.up = gate, up
        self.weight_gate, self.weight_up = weight_gate, weight_up
        self.gate_scale, self.up_scale = gate_scale, up_scale
        self.weight_gate_scale = weight_gate_scale
        self.weight_up_scale = weight_up_scale
        self.out = out
        self.gate_shape = gate.shape
        self.weight_shape = weight_gate.shape
        self.gate_scale_shape = gate_scale.shape
        self.weight_scale_shape = weight_gate_scale.shape
        self.out_shape = out.shape
        with torch.cuda.device(gate.device):
            self.handle = owner.lib.tk_swd_create(
                gate.data_ptr(),
                up.data_ptr(),
                weight_gate.data_ptr(),
                weight_up.data_ptr(),
                gate_scale.data_ptr(),
                up_scale.data_ptr(),
                weight_gate_scale.data_ptr(),
                weight_up_scale.data_ptr(),
                out.data_ptr(),
                m,
                k,
                n,
                config,
                sms,
                raster_group,
            )
        if not self.handle:
            raise RuntimeError(owner.lib.tk_swd_error().decode())
        info = (ct.c_int * 8)()
        owner.lib.tk_swd_info(self.handle, info)
        self.info = dict(
            zip(
                ["R", "C", "BN", "stages", "shared_bytes", "initial_regs", "grid"],
                info,
            )
        )
        self.info["raster_group"] = info[7]

    def __call__(self):
        with torch.cuda.device(self.out.device):
            status = self.owner.lib.tk_swd_launch(
                self.handle,
                torch.cuda.current_stream(self.out.device).cuda_stream,
            )
        if status:
            raise RuntimeError(self.owner.lib.tk_swd_error().decode())
        return self.out

    def release(self):
        self.gate = self.up = None
        self.weight_gate = self.weight_up = None
        self.gate_scale = self.up_scale = None
        self.weight_gate_scale = self.weight_up_scale = None
        self.out = None

    def rebind(
        self,
        gate,
        up,
        weight_gate,
        weight_up,
        gate_scale,
        up_scale,
        weight_gate_scale,
        weight_up_scale,
        out=None,
    ):
        if up.shape != gate.shape or weight_up.shape != weight_gate.shape:
            raise ValueError("Expected dG/dU[M,N], Wg/Wu[N,K]")
        if gate.shape[1] != weight_gate.shape[0]:
            raise ValueError("Expected dG/dU[M,N], Wg/Wu[N,K]")
        if (
            gate.dtype != torch.float8_e4m3fn
            or up.dtype != gate.dtype
            or weight_gate.dtype != gate.dtype
            or weight_up.dtype != gate.dtype
        ):
            raise ValueError("Expected E4M3 gradient and weight matrices")
        if gate_scale.shape != self.gate_scale_shape or gate_scale.dtype != torch.int32:
            raise ValueError("Invalid packed gradient scale")
        if up_scale.shape != self.gate_scale_shape or up_scale.dtype != torch.int32:
            raise ValueError("Invalid packed gradient scale")
        if (
            weight_gate_scale.shape != self.weight_scale_shape
            or weight_gate_scale.dtype != torch.uint8
        ):
            raise ValueError("Invalid weight scale")
        if (
            weight_up_scale.shape != self.weight_scale_shape
            or weight_up_scale.dtype != torch.uint8
        ):
            raise ValueError("Invalid weight scale")
        if out is None:
            out = self.out
        if out.shape != self.out_shape or out.dtype != torch.bfloat16:
            raise ValueError("Expected BF16 dX[M,K]")
        inputs = [
            gate,
            up,
            weight_gate,
            weight_up,
            gate_scale,
            up_scale,
            weight_gate_scale,
            weight_up_scale,
        ]
        tensors = inputs + [out]
        if any(
            tensor.device != gate.device or not tensor.is_contiguous()
            for tensor in tensors
        ):
            raise ValueError("Expected contiguous tensors on one CUDA device")
        if any(tensor.data_ptr() % 16 for tensor in tensors):
            raise ValueError("Tensor pointers must be 16-byte aligned")
        self.gate, self.up = gate, up
        self.weight_gate, self.weight_up = weight_gate, weight_up
        self.gate_scale, self.up_scale = gate_scale, up_scale
        self.weight_gate_scale = weight_gate_scale
        self.weight_up_scale = weight_up_scale
        self.out = out
        status = self.owner.lib.tk_swd_rebind(
            self.handle,
            gate.data_ptr(),
            up.data_ptr(),
            weight_gate.data_ptr(),
            weight_up.data_ptr(),
            gate_scale.data_ptr(),
            up_scale.data_ptr(),
            weight_gate_scale.data_ptr(),
            weight_up_scale.data_ptr(),
            out.data_ptr(),
        )
        if status:
            raise RuntimeError(self.owner.lib.tk_swd_error().decode())
        return self

    def __del__(self):
        if getattr(self, "handle", None):
            self.owner.lib.tk_swd_destroy(self.handle)
            self.handle = None
