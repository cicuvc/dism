"""Pointer/stream adapter for the WASP FP32 -> E4M3 + 32x32 E8M0 quantizer."""
import ctypes as ct
from pathlib import Path
import subprocess
import torch
from .core import BUILD_ROOT, ROOT, FLAGS

DEFAULT_OUTPUT=BUILD_ROOT/'tk_weight_quantize'
CONFIGS=[(4,2,True,32),(4,3,True,32),(8,2,True,32),(4,2,False,32),
         (4,3,False,32),(8,2,False,32),(4,1,True,128),(4,1,False,128)]


def build(output=DEFAULT_OUTPUT,cuda=Path('/usr/local/cuda')):
    output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    with (output/'build.log').open('w') as log:
        subprocess.run([str(cuda/'bin/nvcc'),*FLAGS,'-Xptxas=-v','-shared','-Xcompiler=-fPIC',
                        'cuda/persistent_weight_quantize.cu','-lcuda','-o',str(output/'libquant.so')],
                       cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)


class QuantLibrary:
    def __init__(self,output=DEFAULT_OUTPUT):
        self.lib=ct.CDLL(str(Path(output).resolve()/'libquant.so'))
        self.lib.tk_quant_create.argtypes=[ct.c_void_p]*3+[ct.c_int]*7
        self.lib.tk_quant_create.restype=ct.c_void_p
        self.lib.tk_quant_launch.argtypes=[ct.c_void_p,ct.c_void_p]
        self.lib.tk_quant_launch.restype=ct.c_int
        self.lib.tk_quant_error.restype=ct.c_char_p
        self.lib.tk_quant_info.argtypes=[ct.c_void_p,ct.POINTER(ct.c_int)]
        self.lib.tk_quant_destroy.argtypes=[ct.c_void_p]

    def plan(self,weight,config=2,round_bf16=False,out=None,scales=None,sms=0):
        return QuantPlan(self,weight,config,round_bf16,out,scales,sms)


class QuantPlan:
    """Finite FP32 weights [N,K] -> E4M3 [N,K], uint8 E8M0 [ceil(N/32),ceil(K/32)].

    Scale is the maximum biased FP32 exponent in each block minus 7, leaving
    one E4M3 exponent of overflow headroom; e=0 represents 2^-127. Optional
    round_bf16 matches the numerical BF16 weight quantization contract (values
    must remain finite after rounding).
    Input columns are contiguous, row stride is a multiple of four floats.
    Output is padded to a 16-byte row stride when necessary. Descriptor bounds
    remain logical N,K, so partial rows/columns are zero-filled on TMA loads.
    TMA stores may write the final 16-byte atom, including zero column padding.
    Plans retain allocations and can be reused after changing input contents.
    """
    def __init__(self,owner,w,config,round_bf16,out,scales,sms):
        self.handle=None
        if w.ndim!=2 or w.dtype!=torch.float32 or not w.is_cuda or w.stride(1)!=1 or w.stride(0)%4:
            raise ValueError('Expected CUDA FP32 [N,K], contiguous columns, 16-byte aligned row stride')
        if w.data_ptr()%16 or not 0<=config<len(CONFIGS):
            raise ValueError('Invalid base alignment or configuration')
        n,k=w.shape
        if out is None:
            out=torch.empty((n,(k+15)//16*16),device=w.device,dtype=torch.float8_e4m3fn)[:,:k]
        if (out.shape!=(n,k) or out.dtype!=torch.float8_e4m3fn or out.device!=w.device
                or out.stride(1)!=1 or out.stride(0)%16 or out.data_ptr()%16):
            raise ValueError('Invalid FP8 output shape, dtype, device, or alignment')
        required=out.storage_offset()+(n-1)*out.stride(0)+(k+15)//16*16
        if out.untyped_storage().nbytes()<required:
            raise ValueError('Output storage must include the final padded 16-byte TMA atom')
        scale_shape=((n+31)//32,(k+31)//32)
        if scales is None:
            scales=torch.empty(scale_shape,device=w.device,dtype=torch.uint8)
        if scales.shape!=scale_shape or scales.dtype!=torch.uint8 or scales.device!=w.device or not scales.is_contiguous():
            raise ValueError('Invalid scale buffer')
        self.owner,self.weight,self.out,self.scales=owner,w,out,scales
        with torch.cuda.device(w.device):
            self.handle=owner.lib.tk_quant_create(w.data_ptr(),out.data_ptr(),scales.data_ptr(),n,k,
                                                 w.stride(0),out.stride(0),config,int(round_bf16),sms)
        if not self.handle:
            raise RuntimeError(owner.lib.tk_quant_error().decode())
        info=(ct.c_int*8)()
        owner.lib.tk_quant_info(self.handle,info)
        self.info=dict(zip(['consumer_warps','stages','redux','round_bf16','shared_bytes','initial_regs','grid','warp_cols'],info))

    def __call__(self):
        with torch.cuda.device(self.weight.device):
            if self.owner.lib.tk_quant_launch(self.handle,torch.cuda.current_stream(self.weight.device).cuda_stream):
                raise RuntimeError(self.owner.lib.tk_quant_error().decode())
        return self.out,self.scales

    def __del__(self):
        if getattr(self,'handle',None):
            self.owner.lib.tk_quant_destroy(self.handle)
            self.handle=None
