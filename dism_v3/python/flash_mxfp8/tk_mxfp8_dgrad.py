"""Prequantized MXFP8 dX: dY[M,N] @ original W[N,K] -> BF16 [M,K].

Uses swapAB so both LDSM operand layouts feed QMMA directly. dY scales use
packed per-row/N32 E8M0; W retains the forward 32x32 block scale grid.
"""
import ctypes as ct
from pathlib import Path
import re
import subprocess
import torch
from .core import BUILD_ROOT,ROOT,FLAGS

OUTPUT=BUILD_ROOT/'tk_mxfp8_dgrad'
CONFIGS=[tuple(map(int,m[1:])) for m in re.findall(
    r'^CFG\((\d+),(\d+),(\d+),(\d+),(\d+)\);',
    (ROOT/'cuda/mxfp8_dgrad_configs.inc').read_text(),re.M)]


def build(output=OUTPUT):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    with (output/'build.log').open('w') as log:
        subprocess.run(['/usr/local/cuda/bin/nvcc',*FLAGS,'-Xptxas=-v','-shared','-Xcompiler=-fPIC',
                        'cuda/persistent_mxfp8_dgrad.cu','-lcuda','-o',str(output/'libdgrad.so')],
                       cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)


class Library:
    def __init__(self,output=OUTPUT):
        self.lib=ct.CDLL(str(Path(output)/'libdgrad.so'))
        self.lib.tk_mxd_create.argtypes=[ct.c_void_p]*5+[ct.c_int]*6
        self.lib.tk_mxd_create.restype=ct.c_void_p
        self.lib.tk_mxd_launch.argtypes=[ct.c_void_p,ct.c_void_p]
        self.lib.tk_mxd_rebind.argtypes=[ct.c_void_p]*6
        self.lib.tk_mxd_rebind.restype=ct.c_int
        self.lib.tk_mxd_error.restype=ct.c_char_p
        self.lib.tk_mxd_info.argtypes=[ct.c_void_p,ct.POINTER(ct.c_int)]
        self.lib.tk_mxd_destroy.argtypes=[ct.c_void_p]

    def plan(self,a,b,as_,bs,config=7,out=None,*,sms=0,raster_group=1):
        return Plan(self,a,b,as_,bs,config,out,sms,raster_group)


class Plan:
    """Fixed-buffer dX GEMM, without quantization or physical transposes.

    dY[M,N] and W[N,K] are contiguous E4M3. M tails are TMA-masked.
    N/K must align to the selected TMA swizzle panels. BF16 output only.
    """
    def __init__(self,owner,a,b,as_,bs,config,out,sms,group):
        self.handle=None
        if type(config)!=int or not 0<=config<len(CONFIGS):raise ValueError('Invalid config')
        if type(group)!=int or not 0<group<2**31 or type(sms)!=int or not 0<=sms<2**31:
            raise ValueError('Invalid raster group / SM count')
        if a.ndim!=2 or b.ndim!=2 or a.shape[1]!=b.shape[0] or min(*a.shape,*b.shape)<=0:
            raise ValueError('Expected dY[M,N] and W[N,K]')
        m,n=a.shape;k=b.shape[1];r,c,bn,stages=CONFIGS[config]
        if n%min(128,bn) or k%min(128,c):
            raise ValueError('N/K must align to TMA swizzle panels')
        if a.dtype!=torch.float8_e4m3fn or b.dtype!=a.dtype:raise ValueError('Expected E4M3 operands')
        if as_.shape!=((n+127)//128,(m+15)//16*16) or bs.shape!=((n+31)//32,k//32):
            raise ValueError('Invalid packed scale shapes')
        if as_.dtype!=torch.int32 or bs.dtype!=torch.uint8:raise ValueError('Expected int32 dY scales and uint8 weight scales')
        if out is None:out=torch.empty((m,k),device=a.device,dtype=torch.bfloat16)
        if out.shape!=(m,k) or out.dtype!=torch.bfloat16:raise ValueError('Expected BF16 output [M,K]')
        tensors=[a,b,as_,bs,out]
        if not a.is_cuda or any(t.device!=a.device or not t.is_contiguous() for t in tensors):
            raise ValueError('Expected contiguous tensors on one CUDA device')
        if torch.cuda.get_device_capability(a.device)!=(12,0):raise RuntimeError('SM120 required')
        if any(t.data_ptr()%16 for t in tensors):raise ValueError('Unaligned buffer')
        lo=out.data_ptr();hi=lo+out.numel()*out.element_size()
        if any(lo<t.data_ptr()+t.numel()*t.element_size() and t.data_ptr()<hi for t in tensors[:4]):
            raise ValueError('Output aliases an input')
        self.owner=owner;self.refs=tensors;self.out=out
        self.as_shape=as_.shape;self.bs_shape=bs.shape;self.out_shape=out.shape
        with torch.cuda.device(a.device):
            self.handle=owner.lib.tk_mxd_create(*[t.data_ptr() for t in tensors],m,k,n,config,sms,group)
        if not self.handle:raise RuntimeError(owner.lib.tk_mxd_error().decode())
        info=(ct.c_int*7)();owner.lib.tk_mxd_info(self.handle,info)
        self.info=dict(zip(['R','C','BN','stages','shared_bytes','initial_regs','grid'],info))
        self.info.update(raster_group=min(group,(m+8*r-1)//(8*r)),
                         shape_mnk=[m,n,k],bf16_output=True,swap_ab=True)

    def __call__(self):
        with torch.cuda.device(self.out.device):
            if self.owner.lib.tk_mxd_launch(self.handle,torch.cuda.current_stream(self.out.device).cuda_stream):
                raise RuntimeError(self.owner.lib.tk_mxd_error().decode())
        return self.out

    def release(self):
        self.refs=[]
        self.out=None

    def rebind(self,a,b,as_,bs,out=None):
        m,n=a.shape;k=b.shape[1]
        if b.shape[0]!=n:
            raise ValueError('Expected dY[M,N] and W[N,K]')
        if a.dtype!=torch.float8_e4m3fn or b.dtype!=a.dtype:
            raise ValueError('Expected E4M3 operands')
        if as_.shape!=self.as_shape or as_.dtype!=torch.int32:
            raise ValueError('Invalid packed dY scales')
        if bs.shape!=self.bs_shape or bs.dtype!=torch.uint8:
            raise ValueError('Invalid weight scales')
        if out is None:
            out=self.out
        if out.shape!=self.out_shape or out.dtype!=torch.bfloat16:
            raise ValueError('Expected BF16 output [M,K]')
        tensors=[a,b,as_,bs,out]
        if any(t.device!=a.device or not t.is_contiguous() for t in tensors):
            raise ValueError('Expected contiguous tensors on one CUDA device')
        if any(t.data_ptr()%16 for t in tensors):
            raise ValueError('Unaligned buffer')
        self.refs=tensors;self.out=out
        status=self.owner.lib.tk_mxd_rebind(
            self.handle,a.data_ptr(),b.data_ptr(),as_.data_ptr(),
            bs.data_ptr(),out.data_ptr(),
        )
        if status:
            raise RuntimeError(self.owner.lib.tk_mxd_error().decode())
        return self

    def __del__(self):
        if self.handle:self.owner.lib.tk_mxd_destroy(self.handle)
