"""Read-only CUDA diagnostics, kept opaque to Dynamo/autograd."""
import torch
import triton
import triton.language as tl


@triton.jit
def _max_nan(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def _residual_partial(X, P, ROWS: tl.constexpr, D: tl.constexpr,
                      BD: tl.constexpr, BR: tl.constexpr):
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    cols = tl.arange(0, BD)
    x = tl.load(X + rows[:, None] * D + cols[None, :],
                (rows[:, None] < ROWS) & (cols[None, :] < D), 0).to(tl.float32)
    means = tl.sum(x * x, 1) / D
    tl.store(P + tl.program_id(0), tl.reduce(means, 0, _max_nan))


@triton.jit
def _residual_finish(P, O, N: tl.constexpr, B: tl.constexpr):
    i = tl.arange(0, B)
    x = tl.load(P + i, i < N, 0)
    tl.store(O, tl.sqrt(tl.reduce(x, 0, _max_nan)))


@torch.library.custom_op('nanochat_metrics::residual_rms_max', mutates_args=())
def residual_rms_max(x: torch.Tensor) -> torch.Tensor:
    rows, d = x.numel() // x.shape[-1], x.shape[-1]
    count = triton.cdiv(rows, 16)
    partial = torch.empty(count, device=x.device, dtype=torch.float32)
    out = torch.empty((), device=x.device, dtype=torch.float32)
    with torch.cuda.device(x.device):
        # Narrow rows fit within a warp; avoid a redundant cross-warp shared
        # reduction layout for padded widths below 32.
        _residual_partial[(count,)](x, partial, rows, d, triton.next_power_of_2(d), 16,
                                    num_warps=1 if d < 32 else 4)
        _residual_finish[(1,)](partial, out, count, triton.next_power_of_2(count))
    return out


@residual_rms_max.register_fake
def _residual_fake(x):
    return torch.empty((), device=x.device, dtype=torch.float32)
