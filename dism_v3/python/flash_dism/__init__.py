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


def dism_core(*args, cu_seqlens=None, layout=None, ctas=0, hard_planner=None):
    """Differentiable core; infer R/D/DV from sq/q/v, BF16 output.

    Supplying cu_seqlens or layout selects packed batch1 execution. All packed
    boundaries (including T) must be multiples of256; no alignment copies.
    ctas is a fixed-path scheduling control, not supported for packed inputs.
    Pass an explicit AsyncHardPlanner as hard_planner to enable complete eager
    hard/soft training. Without it, the CUDA core remains soft-only.
    """
    if hard_planner is not None:
        from .hard_training import dism_core_hybrid
        if cu_seqlens is not None:
            if layout is not None:
                raise ValueError('provide either layout or cu_seqlens, not both')
            layout = VarlenLayout.from_cu_seqlens(cu_seqlens, args[0].shape[1])
        return dism_core_hybrid(*args, planner=hard_planner, layout=layout, ctas=ctas)
    if cu_seqlens is not None or layout is not None:
        if ctas:
            raise ValueError('ctas is only supported for fixed-length execution')
        return dism_core_varlen(*args,cu_seqlens=cu_seqlens,layout=layout)
    return _fixed_core(*args,ctas=ctas)


def voc_dism(*args, direction, hard, cu_seqlens=None, layout=None, ctas=0, beta=1.0, hard_planner=None):
    """Triton vocabulary interpolation followed by the shape-selected CUDA core.

    ``beta`` is the vocabulary-softmax temperature (scalar float or 0-dim tensor);
    ``beta=1.0`` is the original unscaled behavior.
    hard_planner opts into asynchronous hard backward and dedicated hard forward;
    the default and compiled module dispatcher remain soft-only.
    """
    if cu_seqlens is not None or layout is not None:
        if ctas:
            raise ValueError('ctas is only supported for fixed-length execution')
        return voc_dism_varlen(*args,cu_seqlens=cu_seqlens,layout=layout,
                               direction=direction,hard=hard,beta=beta,hard_planner=hard_planner)
    return _fixed_voc(*args,direction=direction,hard=hard,ctas=ctas,beta=beta,hard_planner=hard_planner)


__all__ = ['DismConfig','DismModel','DismForCausalLM','DismAttention','DismSwaAttention','DismCache','dism_core','voc_dism','forward_core','backward_core',
           'forward_varlen','backward_varlen','VarlenLayout','supported_configs',
           'dism_core_varlen','voc_dism_varlen']
