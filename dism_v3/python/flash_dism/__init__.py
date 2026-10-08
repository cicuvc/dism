"""DISM v3: one extension, shape-selected fixed or packed CUDA kernels."""
import torch  # Load Torch's shared libraries before importing the native extension.
from .backend import supported_configs
from .forward import forward_core
from .backward import backward_core, dism_core as _fixed_core
from .voc import voc_dism as _fixed_voc
from .module import DismAttention
from .hybrid import DismSwaAttention
from .cache import DismCache
from .configuration_dism import DismConfig
from .modeling_dism import DismModel, DismForCausalLM
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

AutoConfig.register('dism_v3', DismConfig)
AutoModel.register(DismConfig, DismModel)
AutoModelForCausalLM.register(DismConfig, DismForCausalLM)
from .varlen import (VarlenLayout, forward_varlen, backward_varlen,
                     dism_core_varlen, voc_dism_varlen)


def dism_core(*args, cu_seqlens=None, layout=None, ctas=0):
    """Differentiable core; infer R/D/DV from sq/q/v, BF16 output.

    Supplying cu_seqlens or layout selects packed batch1 execution. All packed
    boundaries (including T) must be multiples of256; no alignment copies.
    ctas is a fixed-path scheduling control, not supported for packed inputs.
    """
    if cu_seqlens is not None or layout is not None:
        if ctas:
            raise ValueError('ctas is only supported for fixed-length execution')
        return dism_core_varlen(*args,cu_seqlens=cu_seqlens,layout=layout)
    return _fixed_core(*args,ctas=ctas)


def voc_dism(*args, direction, hard, cu_seqlens=None, layout=None, ctas=0):
    """Triton vocabulary interpolation followed by the shape-selected CUDA core."""
    if cu_seqlens is not None or layout is not None:
        if ctas:
            raise ValueError('ctas is only supported for fixed-length execution')
        return voc_dism_varlen(*args,cu_seqlens=cu_seqlens,layout=layout,
                               direction=direction,hard=hard)
    return _fixed_voc(*args,direction=direction,hard=hard,ctas=ctas)


__all__ = ['DismConfig','DismModel','DismForCausalLM','DismAttention','DismSwaAttention','DismCache','dism_core','voc_dism','forward_core','backward_core',
           'forward_varlen','backward_varlen','VarlenLayout','supported_configs',
           'dism_core_varlen','voc_dism_varlen']
