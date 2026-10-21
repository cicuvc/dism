"""Prequantized WASP GEMM. A scales pack four K32 E8M0 bytes into int32 [ceil(K/128),roundup(M,16)]; B scales [ceil(N/32),K/32]."""
import ctypes as ct
from pathlib import Path
import subprocess
import re
import torch
from .core import BUILD_ROOT, ROOT, FLAGS
CONFIGS=[tuple(map(int,m[1:])) for m in re.findall(
    r'^CFG\((\d+),(\d+),(\d+),(\d+),(\d+)\);',
    (ROOT/'cuda/mxfp8_configs.inc').read_text(),re.M)]
OUTPUT=BUILD_ROOT/'tk_mxfp8_gemm'
def build(output=OUTPUT,movmatrix=False,factored_stores=False):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    extra=['-DFP8GEMM_EPILOGUE_MOVM=1'] if movmatrix else []
    if factored_stores:extra+=['-DFP8GEMM_FACTORED_STORES=1']
    with (output/'build.log').open('w') as log:
        subprocess.run(['/usr/local/cuda/bin/nvcc',*FLAGS,*extra,'-Xptxas=-v','-shared','-Xcompiler=-fPIC',
                        'cuda/persistent_mxfp8_gemm.cu','-lcuda','-o',str(output/'libgemm.so')],
                       cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
def pack_a_scales(scales):
    """Convert [M,ceil(K/32)] uint8 exponents to TMA-friendly four-byte words."""
    m,k=scales.shape
    padded=torch.zeros(((m+15)//16*16,(k+3)//4*4),device=scales.device,dtype=torch.int32)
    padded[:m,:k]=scales
    words=(padded.reshape(padded.shape[0],-1,4) << torch.tensor([0,8,16,24],device=scales.device,dtype=torch.int32)).sum(-1,dtype=torch.int32)
    return words.T.contiguous()

class Library:
    def __init__(self,output=OUTPUT):
        self.lib=ct.CDLL(str(Path(output)/'libgemm.so'))
        self.lib.tk_mx_create.argtypes=[ct.c_void_p]*5+[ct.c_int]*6
        self.lib.tk_mx_create.restype=ct.c_void_p
        if hasattr(self.lib,'tk_mx_create_quantized'):
            self.lib.tk_mx_create_quantized.argtypes=[ct.c_void_p]*7+[ct.c_int]*6
            self.lib.tk_mx_create_quantized.restype=ct.c_void_p
        self.lib.tk_mx_error.restype=ct.c_char_p
        self.lib.tk_mx_launch.argtypes=[ct.c_void_p,ct.c_void_p]
        self.lib.tk_mx_rebind.argtypes=[ct.c_void_p]*8
        self.lib.tk_mx_rebind.restype=ct.c_int
        self.lib.tk_mx_info.argtypes=[ct.c_void_p,ct.POINTER(ct.c_int)]
        self.lib.tk_mx_destroy.argtypes=[ct.c_void_p]
    def plan(self,a,b,as_,bs,config=8,out=None,sms=0,raster_group=1,
             save_quantized=False,quantized_out=None,quantized_scales=None,save_bf16=True):
        """Optionally expose row-major E4M3 qout and next-A packed qscales.

        Quantization uses the rounded BF16 output. Calling the plan still returns
        BF16 by default. With save_bf16=False, no BF16 output is allocated or
        written, and calling the plan returns qout. Rounding before quantization
        remains BF16. qout/qscales survive launches and graph replays.
        """
        return Plan(self,a,b,as_,bs,config,out,sms,raster_group,
                    save_quantized,quantized_out,quantized_scales,save_bf16)
class Plan:
    def __init__(self,owner,a,b,as_,bs,config,out,sms,group,save_quantized,qout,qscales,save_bf16=True):
        self.handle=None
        if type(group)!=int or not 0<group<2**31:raise ValueError('Invalid raster group')
        if config not in range(len(CONFIGS)):raise ValueError('Invalid config')
        if a.ndim!=2 or b.ndim!=2 or a.shape[1]!=b.shape[1]:raise ValueError('Invalid matrix shapes')
        m,k=a.shape;n=b.shape[0]
        if min(m,n,k)<=0 or k%32:raise ValueError('Invalid dimensions')
        if a.dtype!=torch.float8_e4m3fn or b.dtype!=a.dtype:raise ValueError('Expected E4M3 operands')
        if as_.shape!=((k+127)//128,(m+15)//16*16) or bs.shape!=((n+31)//32,k//32):raise ValueError('Invalid scale shapes')
        if as_.dtype!=torch.int32 or bs.dtype!=torch.uint8:raise ValueError('Expected packed int32 A scales and uint8 B scales')
        if not save_bf16:
            if not save_quantized:raise ValueError('save_bf16=False requires save_quantized=True')
            if out is not None:raise ValueError('out requires save_bf16=True')
        else:
            if out is None:out=torch.empty((m,n),device=a.device,dtype=torch.bfloat16)
            if out.shape!=(m,n) or out.dtype!=torch.bfloat16:raise ValueError('Invalid output')
        tensors=[a,b,as_,bs]+([out] if save_bf16 else [])
        if not save_quantized and (qout is not None or qscales is not None):
            raise ValueError('Quantized outputs require save_quantized=True')
        if save_quantized:
            if CONFIGS[config][1]%32:raise ValueError('Quantized epilogue requires C divisible by 32')
            c=CONFIGS[config][1]
            qpanel=128 if c%128==0 else 64 if c%64==0 else 32
            if n%qpanel:raise ValueError(f'N must align to FP8 output TMA swizzle panels ({qpanel} columns)')
            if qout is None:qout=torch.empty((m,n),device=a.device,dtype=a.dtype)
            if qscales is None:qscales=torch.empty(((n+127)//128,(m+15)//16*16),device=a.device,dtype=torch.int32)
            if qout.shape!=(m,n) or qout.dtype!=a.dtype:raise ValueError('Invalid quantized output')
            if qscales.shape!=((n+127)//128,(m+15)//16*16) or qscales.dtype!=torch.int32:
                raise ValueError('Invalid quantized scales')
            tensors.extend([qout,qscales])
        if not a.is_cuda or any(t.device!=a.device or not t.is_contiguous() for t in tensors):
            raise ValueError('Expected contiguous tensors on one CUDA device')
        if any(t.data_ptr()%16 for t in [a,b,as_,*tensors[4:]]):raise ValueError('Unaligned output/TMA pointer')
        for i,tensor in enumerate(tensors[4:],4):
            lo=tensor.data_ptr();hi=lo+tensor.numel()*tensor.element_size()
            if any(lo<t.data_ptr()+t.numel()*t.element_size() and t.data_ptr()<hi for t in tensors[:i]):
                raise ValueError('Output aliases input or another output')
        self.owner=owner;self.refs=tensors;self.out=out
        self.qout=qout;self.qscales=qscales
        self.result=out if save_bf16 else qout
        self.has_bf16=out is not None
        self.has_quantized=qout is not None
        self.as_shape=as_.shape;self.bs_shape=bs.shape
        self.out_shape=None if out is None else out.shape
        self.qout_shape=None if qout is None else qout.shape
        self.qscales_shape=None if qscales is None else qscales.shape
        with torch.cuda.device(a.device):
            create=owner.lib.tk_mx_create_quantized if save_quantized else owner.lib.tk_mx_create
            pointers=[t.data_ptr() for t in [a,b,as_,bs]]+[out.data_ptr() if out is not None else 0]
            if save_quantized:pointers += [qout.data_ptr(),qscales.data_ptr()]
            self.handle=create(*pointers,m,n,k,config,sms,group)
        if not self.handle:raise RuntimeError(owner.lib.tk_mx_error().decode())
        info=(ct.c_int*7)();owner.lib.tk_mx_info(self.handle,info)
        self.info=dict(zip(['R','C','BK','stages','shared_bytes','initial_regs','grid'],info))
        self.info.update(raster_group=min(group,(m+8*CONFIGS[config][0]-1)//(8*CONFIGS[config][0])))
    def __call__(self):
        with torch.cuda.device(self.result.device):
            if self.owner.lib.tk_mx_launch(self.handle,torch.cuda.current_stream(self.result.device).cuda_stream):
                raise RuntimeError(self.owner.lib.tk_mx_error().decode())
        return self.result
    def release(self):
        self.refs=[]
        self.out=None
        self.qout=None
        self.qscales=None
        self.result=None
    def rebind(self,a,b,as_,bs,out=None,quantized_out=None,quantized_scales=None):
        m,k=a.shape;n=b.shape[0]
        if a.shape!=(m,k) or b.shape[1]!=k:
            raise ValueError('Invalid rebind matrix shapes')
        if a.dtype!=torch.float8_e4m3fn or b.dtype!=a.dtype:
            raise ValueError('Expected E4M3 operands')
        if as_.shape!=self.as_shape or as_.dtype!=torch.int32:
            raise ValueError('Invalid packed A scales')
        if bs.shape!=self.bs_shape or bs.dtype!=torch.uint8:
            raise ValueError('Invalid B scales')
        if out is None:
            out=self.out
        if self.has_bf16:
            if out is None or out.shape!=self.out_shape or out.dtype!=torch.bfloat16:
                raise ValueError('Invalid BF16 output')
        elif out is not None:
            raise ValueError('This plan has no BF16 output')
        if quantized_out is None:
            quantized_out=self.qout
        if quantized_scales is None:
            quantized_scales=self.qscales
        if self.has_quantized:
            if (quantized_out is None or quantized_out.shape!=self.qout_shape
                    or quantized_out.dtype!=a.dtype):
                raise ValueError('Invalid quantized output')
            if (quantized_scales is None
                    or quantized_scales.shape!=self.qscales_shape
                    or quantized_scales.dtype!=torch.int32):
                raise ValueError('Invalid quantized scales')
        tensors=[a,b,as_,bs]+([out] if out is not None else [])+(
            [quantized_out,quantized_scales] if quantized_out is not None else [])
        if any(t.device!=a.device or not t.is_contiguous() for t in tensors):
            raise ValueError('Expected contiguous tensors on one CUDA device')
        if any(t.data_ptr()%16 for t in [a,b,as_,out,quantized_out,quantized_scales] if t is not None):
            raise ValueError('Unaligned output/TMA pointer')
        self.refs=tensors
        status=self.owner.lib.tk_mx_rebind(
            self.handle,a.data_ptr(),b.data_ptr(),as_.data_ptr(),bs.data_ptr(),
            out.data_ptr() if out is not None else 0,
            quantized_out.data_ptr() if quantized_out is not None else 0,
            quantized_scales.data_ptr() if quantized_scales is not None else 0,
        )
        if status:
            raise RuntimeError(self.owner.lib.tk_mx_error().decode())
        self.out=out;self.qout=quantized_out;self.qscales=quantized_scales
        self.result=out if out is not None else quantized_out
        return self
    def __del__(self):
        if self.handle:self.owner.lib.tk_mx_destroy(self.handle)
