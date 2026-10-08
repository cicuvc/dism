import math

import torch
import triton
import triton.language as tl


_BLOCK_SIZE = 256


@triton.jit
def _random_bool_kernel(
    output, seed_ptr, numel, probability,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = (
        tl.program_id(0).to(tl.uint64) * BLOCK_SIZE
        + tl.arange(0, BLOCK_SIZE).to(tl.uint64))
    mask = offsets < numel
    seed = tl.load(seed_ptr).to(tl.uint64)
    probability = probability.to(tl.float32)
    values = tl.rand(seed, offsets) < probability
    tl.store(output + offsets, values.to(tl.uint8), mask=mask)


def _normalize_shape(shape):
    if isinstance(shape, int):
        shape = (shape,)
    elif isinstance(shape, (tuple, list, torch.Size)):
        shape = tuple(shape)
    else:
        raise TypeError("shape must be an integer or a sequence of integers")
    if any(isinstance(size, bool) or not isinstance(size, int) for size in shape):
        raise TypeError("every shape dimension must be an integer")
    if any(size < 0 for size in shape):
        raise ValueError("shape dimensions must be nonnegative")
    return shape


def _seed_tensor(seed, device):
    if seed is None:
        return torch.randint(
            0, torch.iinfo(torch.int64).max, (),
            dtype=torch.int64, device=device)
    if isinstance(seed, bool):
        raise TypeError("seed must be an integer or a scalar integer tensor")
    if isinstance(seed, int):
        if not (torch.iinfo(torch.int64).min
                <= seed <= torch.iinfo(torch.int64).max):
            raise ValueError("integer seed must fit in int64")
        return torch.tensor(seed, dtype=torch.int64, device=device)
    if not isinstance(seed, torch.Tensor):
        raise TypeError("seed must be an integer or a scalar integer tensor")
    if (seed.ndim != 0 or seed.dtype not in (torch.int32, torch.int64)
            or seed.device != device):
        raise ValueError(
            "tensor seed must be a scalar CUDA int32/int64 tensor on output device")
    return seed


def triton_rand_bool(
    shape, prob, *, device=None, seed=None, dtype=torch.bool,
):
    """Generate an independent Bernoulli mask with Triton's ``tl.rand``.

    ``torch.bool`` and ``torch.uint8`` outputs both occupy one byte per value.
    Supplying an integer or scalar integer tensor seed makes the result
    reproducible. With ``seed=None``, one seed is drawn from PyTorch's CUDA RNG.
    """
    shape = _normalize_shape(shape)
    if isinstance(prob, bool) or not isinstance(prob, (int, float)):
        raise TypeError("probability must be a finite number")
    probability = float(prob)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must lie in [0, 1]")
    if dtype not in (torch.bool, torch.uint8):
        raise TypeError("dtype must be torch.bool or torch.uint8")

    device = torch.device("cuda" if device is None else device)
    if device.type != "cuda":
        raise ValueError("triton_rand_bool requires a CUDA device")
    output = torch.empty(shape, dtype=dtype, device=device)
    numel = output.numel()
    if numel == 0:
        return output

    seed = _seed_tensor(seed, output.device)
    _random_bool_kernel[(triton.cdiv(numel, _BLOCK_SIZE),)](
        output, seed, numel, probability,
        BLOCK_SIZE=_BLOCK_SIZE, num_warps=4)
    return output
