"""Torch-only module inference math and tensor-tuple FLA cache adapter."""
import torch
from torch.nn import functional as F
from .reference.dism_decode_ref import DismDecodeCache, dism_wrapper_decode


def _linear(x, layer):
    return F.linear(x, layer.weight.to(x.dtype),
                    None if layer.bias is None else layer.bias.to(x.dtype))


def _convolve(x, layer, state):
    """History contains pre-convolution projections, oldest to newest."""
    projected = F.linear(x, layer.weight.to(x.dtype))
    history_size = layer.kernel_size - 1
    if state is None:
        state = projected.new_zeros(x.shape[0], history_size, layer.out_channels)
    if (state.shape != (x.shape[0], history_size, layer.out_channels)
            or state.device != x.device or state.dtype != x.dtype):
        raise ValueError("convolution cache shape/device/dtype changed")
    history = torch.cat((state, projected), dim=1)
    weight = layer.conv_weight.to(x.dtype)
    values = sum(history[:, history_size-lag:history_size-lag+x.shape[1]] * weight[lag]
                 for lag in range(layer.kernel_size))
    updated = history[:, -history_size:].clone() if history_size else history[:, :0].clone()
    return F.silu(values), updated


def _readout(x, layer, cos, sin):
    projected = F.linear(x, layer.weight.to(x.dtype)).unflatten(-1, layer.rms_weight.shape)
    normalized = projected * torch.rsqrt(projected.square().mean(-1, keepdim=True) + layer.eps)
    normalized = normalized * layer.rms_weight.to(x.dtype)
    first, second = normalized.chunk(2, dim=-1)
    cos, sin = cos[None, :, None, :], sin[None, :, None, :]
    return torch.cat((first*cos - second*sin, second*cos + first*sin), dim=-1)


def _unpack_cache(state, extra_states=0):
    if state is None:
        return None, (None, None, None)
    tensors, conv = state['recurrent_state'], state['conv_state']
    if not isinstance(tensors, (tuple, list)) or len(tensors) != 8 + extra_states or conv is None or len(conv) != 3:
        raise ValueError("expected a DISM recurrent/conv cache")
    # All FLA tuple members have batch on axis 0, including per-head tau.
    if not torch.equal(tensors[7], tensors[7][:1].expand_as(tensors[7])):
        raise ValueError("cached tau differs across batch entries")
    return DismDecodeCache(*tensors[:7], tensors[7][0]), conv


@torch.no_grad()
def forward_torch(module, x, last_state, *, direction=None, hard=None,
                  hard_prob=None, hard_seed=None, generator=None):
    """Exact FP32 (FP64 for double models) inference; no Triton/CUDA kernel calls."""
    batch, length, _ = x.shape
    if batch < 1 or length < 1:
        raise ValueError("decoding requires nonempty batch and new tokens")
    core_cache, conv_states = _unpack_cache(last_state, module._decode_extra_states)
    probability = 1.0 if hard_prob is None else float(hard_prob)
    if hard is None and hard_seed is not None and probability not in (0.0, 1.0):
        if isinstance(hard_seed, torch.Tensor):
            if (hard_seed.ndim != 0 or hard_seed.dtype not in (torch.int32, torch.int64)
                    or hard_seed.device != x.device):
                raise ValueError("hard_seed must be a scalar integer tensor on the input device")
            # Torch Generator.manual_seed needs a host integer (reference path only).
            hard_seed = hard_seed.item()
        if isinstance(hard_seed, bool) or not isinstance(hard_seed, int):
            raise TypeError("hard_seed must be an integer or a scalar integer tensor")
        if not -(1 << 63) <= hard_seed < (1 << 63):
            raise ValueError("hard_seed must fit in int64")
        hard_generator = torch.Generator(device=x.device).manual_seed(hard_seed)
        hard = torch.rand((batch, module.heads, length), device=x.device,
                          generator=hard_generator) < probability
    dtype = torch.float64 if module.q_vocab.dtype == torch.float64 else torch.float32
    output_dtype = torch.get_autocast_dtype(x.device.type) if torch.is_autocast_enabled(x.device.type) else x.dtype
    with torch.autocast(device_type=x.device.type, enabled=False):
        x = x.to(dtype)
        q, cq = _convolve(x, module.q_conv, conv_states[0])
        k, ck = _convolve(x, module.k_conv, conv_states[1])
        v, cv = _convolve(x, module.v_conv, conv_states[2])
        q = q.reshape(batch, length, module.heads, module.head_dim)
        k = k.reshape_as(q)
        v = v.reshape(batch, length, module.heads, module.value_dim)
        sq = module._activate_readout(F.linear(x, module.sq_proj.weight.to(dtype)))
        sk = module._activate_readout(F.linear(x, module.sk_proj.weight.to(dtype)), is_key=True)
        output, cache = dism_wrapper_decode(
            q, k, sq, sk, module.q_vocab.to(dtype), module.k_vocab.to(dtype), v,
            F.softplus(module.log_sel_tau.to(dtype)), cache=core_cache,
            direction=direction, hard=hard, hard_prob=probability,
            generator=generator)
        previous_extra = () if last_state is None else last_state['recurrent_state'][8:]
        output, extra_state = module._combine_torch(output, x, cache, previous_extra)
        gate = _linear(_linear(x, module.g_proj_down), module.g_proj_up).reshape_as(output)
        output = output * torch.rsqrt(output.square().mean(-1, keepdim=True) + module.norm.eps)
        output = output * module.norm.weight.to(dtype) * F.silu(gate)
        output = _linear(output.flatten(2), module.o_proj).to(output_dtype)
    recurrent = (cache.k_vec, cache.sk_vec, cache.v, cache.k_lse, cache.idx_k,
                 cache.last_w, cache.direction, cache.rtau.expand(batch, -1).clone())
    return output, (*recurrent, *extra_state), (cq, ck, cv)
