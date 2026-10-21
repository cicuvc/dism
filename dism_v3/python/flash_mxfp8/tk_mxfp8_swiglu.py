"""Torch adapter for the fused prequantized-MXFP8 SwiGLU CUDA kernel.

``x`` is E4M3 [M,K] with one E8M0 scale per row/K32. The gate and up weights
use the usual 32x32 block-scale layout. The fused result is BF16 [M,N].
Optionally, the pre-activation half-gate ``0.5*gate(x)`` and raw up linear
output are also written as BF16 through asynchronous TMA stores for backward.
"""
import ctypes as ct
from pathlib import Path
import re
import subprocess

import torch

from .core import BUILD_ROOT, ROOT, FLAGS

OUTPUT = BUILD_ROOT / "tk_mxfp8_swiglu"
CONFIGS = [
    tuple(map(int, match))
    for match in re.findall(
        r"^CFG\((\d+),(\d+),(\d+),(\d+),(\d+)\);",
        (ROOT / "cuda/mxfp8_swiglu_configs.inc").read_text(),
        re.MULTILINE,
    )
]


def build(output=OUTPUT):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    command = [
        "/usr/local/cuda/bin/nvcc",
        *FLAGS,
        "-Xptxas=-v",
        "-shared",
        "-Xcompiler=-fPIC",
        "cuda/persistent_mxfp8_swiglu.cu",
        "-lcuda",
        "-o",
        str(output / "libswiglu.so"),
    ]
    with (output / "build.log").open("w") as log:
        subprocess.run(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=600,
        )


class MXFP8SwiGLULibrary:
    def __init__(self, output=OUTPUT):
        self.lib = ct.CDLL(str(Path(output).resolve() / "libswiglu.so"))
        self.lib.tk_mx_swiglu_create.argtypes = [ct.c_void_p] * 7 + [ct.c_int] * 6
        self.lib.tk_mx_swiglu_create.restype = ct.c_void_p
        if hasattr(self.lib, "tk_mx_swiglu_create_aux"):
            self.lib.tk_mx_swiglu_create_aux.argtypes = (
                [ct.c_void_p] * 9 + [ct.c_int] * 6
            )
            self.lib.tk_mx_swiglu_create_aux.restype = ct.c_void_p
        self.lib.tk_mx_swiglu_launch.argtypes = [ct.c_void_p, ct.c_void_p]
        self.lib.tk_mx_swiglu_launch.restype = ct.c_int
        self.lib.tk_mx_swiglu_rebind.argtypes = [ct.c_void_p] * 10
        self.lib.tk_mx_swiglu_rebind.restype = ct.c_int
        self.lib.tk_mx_swiglu_info.argtypes = [ct.c_void_p, ct.POINTER(ct.c_int)]
        self.lib.tk_mx_swiglu_destroy.argtypes = [ct.c_void_p]
        self.lib.tk_mx_swiglu_error.restype = ct.c_char_p

    def plan(
        self,
        x,
        x_scale,
        gate,
        gate_scale,
        up,
        up_scale,
        config=3,
        out=None,
        gate_out=None,
        up_out=None,
        sms=0,
        raster_group=1,
    ):
        return MXFP8SwiGLUPlan(
            self,
            x,
            x_scale,
            gate,
            gate_scale,
            up,
            up_scale,
            config,
            out,
            gate_out,
            up_out,
            sms,
            raster_group,
        )


class MXFP8SwiGLUPlan:
    """Reusable fixed-shape plan; all input allocations are read on replay."""

    def __init__(
        self,
        owner,
        x,
        x_scale,
        gate,
        gate_scale,
        up,
        up_scale,
        config,
        out,
        gate_out,
        up_out,
        sms,
        raster_group,
    ):
        self.handle = None
        if config not in range(len(CONFIGS)):
            raise ValueError("Invalid configuration")
        if x.ndim != 2 or gate.ndim != 2 or gate.shape != up.shape:
            raise ValueError("Expected X[M,K] and equal gate/up [N,K]")
        m, k = x.shape
        n = gate.shape[0]
        if min(m, n, k) <= 0 or gate.shape[1] != k or k % 32:
            raise ValueError("Invalid matrix dimensions")
        if x.dtype != torch.float8_e4m3fn or gate.dtype != x.dtype or up.dtype != x.dtype:
            raise ValueError("Expected E4M3 x, gate, and up")
        expected_a_scale = ((k + 127) // 128, (m + 15) // 16 * 16)
        expected_b_scale = ((n + 31) // 32, k // 32)
        if x_scale.dtype != torch.int32 or x_scale.shape != expected_a_scale:
            raise ValueError("Expected packed int32 X scales [ceil(K/128),roundup(M,16)]")
        if gate_scale.dtype != torch.uint8 or gate_scale.shape != expected_b_scale:
            raise ValueError("Invalid gate scale shape or dtype")
        if up_scale.dtype != torch.uint8 or up_scale.shape != expected_b_scale:
            raise ValueError("Invalid up scale shape or dtype")
        if out is None:
            out = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
        if (
            out.shape != (m, n)
            or out.dtype != torch.bfloat16
            or out.device != x.device
            or not out.is_contiguous()
        ):
            raise ValueError("Invalid output buffer")
        if (gate_out is None) != (up_out is None):
            raise ValueError("gate_out and up_out must be supplied together")
        aux_outputs = [] if gate_out is None else [gate_out, up_out]
        for tensor in aux_outputs:
            if (
                tensor.shape != (m, n)
                or tensor.dtype != torch.bfloat16
                or tensor.device != x.device
                or not tensor.is_contiguous()
            ):
                raise ValueError("Invalid gate-half/up-linear output buffer")
        inputs = [x, x_scale, gate, gate_scale, up, up_scale]
        outputs = [out, *aux_outputs]
        tensors = inputs + outputs
        if not x.is_cuda or any(
            tensor.device != x.device or not tensor.is_contiguous()
            for tensor in tensors
        ):
            raise ValueError("Expected contiguous tensors on one CUDA device")
        if any(tensor.data_ptr() % 16 for tensor in [x, gate, up, *outputs]):
            raise ValueError("TMA tensor pointers must be 16-byte aligned")
        for index, output in enumerate(outputs):
            lo = output.data_ptr()
            hi = lo + output.numel() * output.element_size()
            if any(
                lo < tensor.data_ptr() + tensor.numel() * tensor.element_size()
                and tensor.data_ptr() < hi
                for tensor in inputs + outputs[:index]
            ):
                raise ValueError("Output buffers must not overlap inputs or each other")
        self.owner = owner
        self.x, self.x_scale = x, x_scale
        self.gate, self.gate_scale = gate, gate_scale
        self.up, self.up_scale, self.out = up, up_scale, out
        self.gate_out, self.up_out = gate_out, up_out
        self.x_scale_shape = x_scale.shape
        self.gate_scale_shape = gate_scale.shape
        self.up_scale_shape = up_scale.shape
        self.out_shape = out.shape
        self.aux_shape = None if gate_out is None else gate_out.shape
        self.has_aux = gate_out is not None
        with torch.cuda.device(x.device):
            create = (
                owner.lib.tk_mx_swiglu_create
                if gate_out is None
                else owner.lib.tk_mx_swiglu_create_aux
            )
            pointers = [
                x.data_ptr(),
                x_scale.data_ptr(),
                gate.data_ptr(),
                gate_scale.data_ptr(),
                up.data_ptr(),
                up_scale.data_ptr(),
                out.data_ptr(),
            ]
            if gate_out is not None:
                pointers.extend([gate_out.data_ptr(), up_out.data_ptr()])
            self.handle = create(
                *pointers,
                m,
                n,
                k,
                config,
                sms,
                raster_group,
            )
        if not self.handle:
            raise RuntimeError(owner.lib.tk_mx_swiglu_error().decode())
        info = (ct.c_int * 10)()
        owner.lib.tk_mx_swiglu_info(self.handle, info)
        self.info = dict(zip(
            ["R", "C", "BK", "stages", "shared_bytes", "initial_regs", "grid"],
            info[:7],
        ))
        self.info["raster_group"] = info[7]
        self.info["store_aux"] = bool(info[8])
        self.info["aux_buffers"] = info[9]

    def __call__(self):
        with torch.cuda.device(self.x.device):
            status = self.owner.lib.tk_mx_swiglu_launch(
                self.handle,
                torch.cuda.current_stream(self.x.device).cuda_stream,
            )
        if status:
            raise RuntimeError(self.owner.lib.tk_mx_swiglu_error().decode())
        return self.out

    def release(self):
        self.refs = []
        self.x = self.x_scale = None
        self.gate = self.gate_scale = None
        self.up = self.up_scale = self.out = None
        self.gate_out = self.up_out = None

    def rebind(
        self,
        x,
        x_scale,
        gate,
        gate_scale,
        up,
        up_scale,
        out=None,
        gate_out=None,
        up_out=None,
    ):
        if gate.shape != up.shape or gate.shape[1] != x.shape[1]:
            raise ValueError("Expected X[M,K] and equal gate/up [N,K]")
        if x.dtype != torch.float8_e4m3fn or gate.dtype != x.dtype or up.dtype != x.dtype:
            raise ValueError("Expected E4M3 x, gate, and up")
        if x_scale.shape != self.x_scale_shape or x_scale.dtype != torch.int32:
            raise ValueError("Invalid packed X scales")
        if gate_scale.shape != self.gate_scale_shape or gate_scale.dtype != torch.uint8:
            raise ValueError("Invalid gate scales")
        if up_scale.shape != self.up_scale_shape or up_scale.dtype != torch.uint8:
            raise ValueError("Invalid up scales")
        if out is None:
            out = self.out
        if out.shape != self.out_shape or out.dtype != torch.bfloat16:
            raise ValueError("Invalid output buffer")
        if (gate_out is None) != (up_out is None):
            raise ValueError("gate_out and up_out must be supplied together")
        if self.has_aux:
            if gate_out is None:
                gate_out, up_out = self.gate_out, self.up_out
            if gate_out is None or up_out is None:
                raise ValueError("Auxiliary outputs are required by this plan")
            if (
                gate_out.shape != self.aux_shape
                or up_out.shape != self.aux_shape
                or gate_out.dtype != torch.bfloat16
                or up_out.dtype != torch.bfloat16
            ):
                raise ValueError("Invalid gate-half/up-linear output buffer")
        elif gate_out is not None:
            raise ValueError("This plan has no auxiliary outputs")
        inputs = [x, x_scale, gate, gate_scale, up, up_scale]
        outputs = [out, *([gate_out, up_out] if gate_out is not None else [])]
        tensors = inputs + outputs
        if any(
            tensor.device != x.device or not tensor.is_contiguous()
            for tensor in tensors
        ):
            raise ValueError("Expected contiguous tensors on one CUDA device")
        if any(tensor.data_ptr() % 16 for tensor in [x, gate, up, *outputs]):
            raise ValueError("TMA tensor pointers must be 16-byte aligned")
        self.refs = tensors
        self.x, self.x_scale = x, x_scale
        self.gate, self.gate_scale = gate, gate_scale
        self.up, self.up_scale, self.out = up, up_scale, out
        self.gate_out, self.up_out = gate_out, up_out
        status = self.owner.lib.tk_mx_swiglu_rebind(
            self.handle,
            x.data_ptr(),
            x_scale.data_ptr(),
            gate.data_ptr(),
            gate_scale.data_ptr(),
            up.data_ptr(),
            up_scale.data_ptr(),
            out.data_ptr(),
            gate_out.data_ptr() if gate_out is not None else 0,
            up_out.data_ptr() if up_out is not None else 0,
        )
        if status:
            raise RuntimeError(self.owner.lib.tk_mx_swiglu_error().decode())
        return self

    def __del__(self):
        if getattr(self, "handle", None):
            self.owner.lib.tk_mx_swiglu_destroy(self.handle)
            self.handle = None
