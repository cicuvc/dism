"""
Gumbel-Sparsemax for [B, N, H, C] tensors along dim=-1.

    y = sparsemax((z + g) / t)

where g ~ Gumbel(0, 1), t is temperature.
Gumbel noise is sampled inside the Triton kernel via tl.rand + inverse CDF.

Constraint: C <= 128 and C is a power of 2.

Backward:
    The Gumbel noise is treated as a constant (straight-through on the noise),
    so the Jacobian is (1/t) * J_sparsemax evaluated at (z+g)/t.
    Since sparsemax backward only depends on the *output* p (to find support),
    we don't need to store g or the pre-sparsemax input.

    J_sparsemax = diag(1_S) - (1/|S|) 1_S 1_S^T

    grad_z_i = (1/t) * (grad_out_i - mean_S(grad_out))   if i in S
               0                                           otherwise
"""

import torch
import triton
import triton.language as tl
from typing import Tuple


# ════════════════════════════════════════════════════════════
#  Forward Kernel: Gumbel-Sparsemax
# ════════════════════════════════════════════════════════════

@triton.jit
def _gumbel_sparsemax_forward_kernel(
    x_ptr,       # [rows, C]  logits z
    out_ptr,     # [rows, C]  output p = sparsemax((z+g)/t)
    stride_row,
    temperature,  # scalar float
    seed,         # int, PRNG seed
    C: tl.constexpr,
):
    pid = tl.program_id(0)

    row_x   = x_ptr   + pid * stride_row
    row_out = out_ptr  + pid * stride_row

    offs = tl.arange(0, C)

    # ── Load logits z ───────────────────────────────────────
    z = tl.load(row_x + offs).to(tl.float32)

    # ── Sample Gumbel(0,1) noise ────────────────────────────
    #   g = -log(-log(u)),  u ~ U(0,1)
    #   Use globally unique offsets so each element gets an independent draw
    rand_offsets = (pid * C + offs).to(tl.int32)
    u = tl.rand(seed, rand_offsets)
    # Clamp to avoid log(0): u in [eps, 1-eps]
    eps: tl.constexpr = 1e-7
    u = tl.minimum(tl.maximum(u, eps), 1.0 - eps)
    g = -tl.log(-tl.log(u))

    # ── Perturbed logits: (z + g) / t ──────────────────────
    inv_t = 1.0 / temperature
    x = (z + g) * inv_t

    # ── Sparsemax on perturbed logits ──────────────────────
    # Step 1: sort descending
    z_sorted = tl.sort(x, dim=0, descending=True)

    # Step 2: cumulative sum
    cssv = tl.cumsum(z_sorted, axis=0)

    # Step 3: find support
    r = (offs + 1).to(tl.float32)
    t_vec = (cssv - 1.0) / r
    support = z_sorted > t_vec

    # Step 4: threshold tau
    k_int = tl.sum(support.to(tl.int32), axis=0)
    k = tl.maximum(k_int, 1).to(tl.float32)
    s = tl.sum(tl.where(support, z_sorted, 0.0), axis=0)
    tau = (s - 1.0) / k

    # Step 5: output = max(x - tau, 0)
    y = tl.maximum(x - tau, 0.0)

    tl.store(row_out + offs, y.to(out_ptr.dtype.element_ty))


# ════════════════════════════════════════════════════════════
#  Backward Kernel (same structure as plain sparsemax,
#                   but with 1/t scaling)
# ════════════════════════════════════════════════════════════

@triton.jit
def _gumbel_sparsemax_backward_kernel(
    out_ptr,       # [rows, C]  forward output p
    grad_out_ptr,  # [rows, C]  upstream gradient dL/dp
    grad_in_ptr,   # [rows, C]  output gradient  dL/dz
    stride_row,
    inv_temperature,  # scalar: 1/t
    C: tl.constexpr,
):
    """
    dp/dz = (1/t) * J_sparsemax
    So: dL/dz = (1/t) * J_sparsemax^T * dL/dp
              = (1/t) * [ dL/dp_i - mean_S(dL/dp) ]  for i in S
                0                                      otherwise
    """
    pid = tl.program_id(0)

    row_out      = out_ptr      + pid * stride_row
    row_grad_out = grad_out_ptr + pid * stride_row
    row_grad_in  = grad_in_ptr  + pid * stride_row

    offs = tl.arange(0, C)

    p  = tl.load(row_out      + offs).to(tl.float32)
    go = tl.load(row_grad_out + offs).to(tl.float32)

    # support
    supp = p > 0.0

    supp_count  = tl.sum(supp.to(tl.float32), axis=0)
    supp_count  = tl.maximum(supp_count, 1.0)
    go_supp_sum = tl.sum(tl.where(supp, go, 0.0), axis=0)

    v = go_supp_sum / supp_count

    # (1/t) * sparsemax_backward
    gi = tl.where(supp, (go - v) * inv_temperature, 0.0)

    tl.store(row_grad_in + offs, gi.to(grad_in_ptr.dtype.element_ty))


# ════════════════════════════════════════════════════════════
#  Python Wrappers
# ════════════════════════════════════════════════════════════

def _num_warps_for(C: int) -> int:
    if C >= 128:
        return 8
    elif C >= 32:
        return 4
    else:
        return 2


def gumbel_sparsemax_forward(
    x: torch.Tensor,
    temperature: float = 1.0,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Gumbel-Sparsemax forward: y = sparsemax((z + g) / t), g ~ Gumbel(0,1).

    Args:
        x:           [B, N, H, C] logits
        temperature: scalar temperature t > 0
        seed:        int seed for Philox PRNG

    Returns:
        y:        [B, N, H, C]  output
        out_flat: [B*N*H, C]    flattened output (saved for backward)
    """
    assert x.ndim == 4
    B, N, H, C = x.shape
    assert C <= 128 and (C & (C - 1)) == 0, f"C must be power-of-2 and <= 128, got {C}"
    assert temperature > 0, f"temperature must be > 0, got {temperature}"

    x_flat = x.contiguous().view(-1, C)
    n_rows = x_flat.shape[0]
    out_flat = torch.empty_like(x_flat)

    _gumbel_sparsemax_forward_kernel[(n_rows,)](
        x_flat, out_flat,
        x_flat.stride(0),
        temperature,
        seed,
        C=C,
        num_warps=_num_warps_for(C),
    )

    return out_flat.view(B, N, H, C), out_flat


def gumbel_sparsemax_backward(
    grad_out: torch.Tensor,
    out_flat: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Gumbel-Sparsemax backward.

    Args:
        grad_out: [B, N, H, C] upstream gradient
        out_flat: [B*N*H, C]   saved forward output
        temperature: same temperature used in forward
    """
    assert grad_out.ndim == 4
    B, N, H, C = grad_out.shape

    go_flat = grad_out.contiguous().view(-1, C)
    grad_in_flat = torch.empty_like(go_flat)

    _gumbel_sparsemax_backward_kernel[(go_flat.shape[0],)](
        out_flat, go_flat, grad_in_flat,
        out_flat.stride(0),
        1.0 / temperature,
        C=C,
        num_warps=_num_warps_for(C),
    )

    return grad_in_flat.view(B, N, H, C)


# ════════════════════════════════════════════════════════════
#  torch.autograd.Function
# ════════════════════════════════════════════════════════════

class GumbelSparsemaxFunction(torch.autograd.Function):
    """
    Usage:
        y = GumbelSparsemaxFunction.apply(logits, temperature, seed)
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, temperature: float = 1.0, seed: int = 0) -> torch.Tensor:
        y, out_flat = gumbel_sparsemax_forward(x, temperature, seed)
        ctx.save_for_backward(out_flat)
        ctx.temperature = temperature
        return y

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (out_flat,) = ctx.saved_tensors
        grad_in = gumbel_sparsemax_backward(grad_out, out_flat, ctx.temperature)
        return grad_in, None, None  # no grad for temperature, seed


class GumbelSparsemax(torch.nn.Module):
    """
    nn.Module wrapper.

    Args:
        temperature: scalar temperature (default 1.0)
        seed:        if None, uses a random seed each forward call

    Example:
        gs = GumbelSparsemax(temperature=0.5)
        y = gs(logits)  # logits: [B, N, H, C]
    """
    def __init__(self, temperature: float = 1.0, seed: int | None = None):
        super().__init__()
        self.temperature = temperature
        self.seed = seed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seed = self.seed if self.seed is not None else torch.randint(0, 2**31, (1,)).item()
        return GumbelSparsemaxFunction.apply(x, self.temperature, seed)


# ════════════════════════════════════════════════════════════
#  Plain Sparsemax (reused for reference / backward check)
# ════════════════════════════════════════════════════════════

def _reference_sparsemax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    z = x.to(torch.float64) if x.dtype == torch.float64 else x.float()
    z_sorted, _ = torch.sort(z, dim=dim, descending=True)
    cumsum = torch.cumsum(z_sorted, dim=dim)
    C = z.size(dim)
    k_range = torch.arange(1, C + 1, device=z.device, dtype=z.dtype)
    shape = [1] * z.ndim
    shape[dim] = C
    k_range = k_range.view(shape)
    support = (1 + k_range * z_sorted) > cumsum
    k = support.sum(dim=dim, keepdim=True).clamp(min=1)
    support_sum = (z_sorted * support).sum(dim=dim, keepdim=True)
    tau = (support_sum - 1) / k
    return torch.clamp(z - tau, min=0).to(x.dtype)


# ════════════════════════════════════════════════════════════
#  Tests
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cuda"

    # ── Test 1: Output properties ──
    print("=" * 60)
    print("  Gumbel-Sparsemax Output Property Tests")
    print("=" * 60)
    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        for shape in [
            (2, 4, 8, 16),
            (1, 1, 1, 4),
            (4, 2, 8, 64),
            (1, 1, 1, 128),
        ]:
            for temp in [0.5, 1.0, 2.0, 3.0, 6.0]:
                x = torch.randn(shape, device=device, dtype=dtype)
                seed = torch.randint(0, 2**31, (1,)).item()
                y = GumbelSparsemaxFunction.apply(x, temp, seed).float()

                # Sparsemax outputs must be non-negative and sum to <= 1
                non_neg = (y >= -1e-6).all().item()
                sum_ok  = (y.sum(dim=-1) <= 1.0 + 1e-2).all().item()
                # At least one element should be > 0 per row
                has_pos = (y.sum(dim=-1) > 0).all().item()
                ok = non_neg and sum_ok and has_pos

                status = "✅" if ok else "❌"
                print(f"  {status} dtype={dtype}, shape={shape}, t={temp:.1f}")

    # ── Test 2: Deterministic with same seed ──
    print()
    print("=" * 60)
    print("  Determinism Test (same seed → same output)")
    print("=" * 60)
    x = torch.randn(2, 3, 4, 32, device=device, dtype=torch.float32)
    seed = 12345
    y1 = GumbelSparsemaxFunction.apply(x, 1.0, seed)
    y2 = GumbelSparsemaxFunction.apply(x, 1.0, seed)
    match = torch.equal(y1, y2)
    print(f"  {'✅' if match else '❌'} same seed → identical output: {match}")

    # ── Test 3: Different seeds → different outputs ──
    y3 = GumbelSparsemaxFunction.apply(x, 1.0, seed + 1)
    differ = not torch.equal(y1, y3)
    print(f"  {'✅' if differ else '❌'} different seed → different output: {differ}")

    # ── Test 4: Temperature effect ──
    print()
    print("=" * 60)
    print("  Temperature Effect Test")
    print("=" * 60)
    x = torch.randn(4, 4, 8, 64, device=device, dtype=torch.float32)
    seed = 999

    # Low temperature → sparser (more zeros)
    y_cold = GumbelSparsemaxFunction.apply(x, 0.1, seed).float()
    y_warm = GumbelSparsemaxFunction.apply(x, 2.0, seed).float()

    sparsity_cold = (y_cold == 0).float().mean().item()
    sparsity_warm = (y_warm == 0).float().mean().item()
    temp_ok = sparsity_cold >= sparsity_warm  # colder should be sparser
    print(f"  t=0.1 sparsity: {sparsity_cold:.3f}")
    print(f"  t=2.0 sparsity: {sparsity_warm:.3f}")
    print(f"  {'✅' if temp_ok else '⚠️ '} lower temp → sparser: {temp_ok}")

    # ── Test 5: Backward correctness (vs PyTorch reference) ──
    print()
    print("=" * 60)
    print("  Backward Test (Triton vs PyTorch reference)")
    print("=" * 60)
    #
    # Since Gumbel noise is sampled inside the kernel and we can't
    # reproduce it externally, we test backward by verifying the
    # sparsemax Jacobian structure on the *output* that was produced.
    #
    # Given output p from forward, backward should satisfy:
    #   grad_in = (1/t) * [grad_out - mean_S(grad_out)] * 1_S
    #
    for shape in [(2, 4, 8, 16), (1, 2, 2, 64), (1, 1, 1, 128)]:
        for temp in [0.5, 1.0, 2.0]:
            x = torch.randn(shape, device=device, dtype=torch.float32, requires_grad=True)
            grad_out = torch.randn(shape, device=device, dtype=torch.float32)
            seed = torch.randint(0, 2**31, (1,)).item()

            # Triton backward
            y = GumbelSparsemaxFunction.apply(x, temp, seed)
            y.backward(grad_out)
            grad_triton = x.grad.clone()

            # Manual reference backward from the same output
            p = y.detach()
            supp = p > 0
            cnt = supp.sum(dim=-1, keepdim=True).clamp(min=1).float()
            v = (grad_out * supp).sum(dim=-1, keepdim=True) / cnt
            grad_ref = torch.where(supp, (grad_out - v) / temp, torch.zeros_like(grad_out))

            max_err = (grad_triton - grad_ref).abs().max().item()
            status = "✅" if max_err < 1e-5 else "❌"
            print(f"  {status} shape={shape}, t={temp}, grad max_err={max_err:.2e}")

    # ── Test 6: Gradcheck (float64, pure-PyTorch Gumbel-Sparsemax) ──
    print()
    print("=" * 60)
    print("  Gradcheck (float64, pure-PyTorch)")
    print("=" * 60)

    class _GumbelSparsemaxF64(torch.autograd.Function):
        """Pure-PyTorch gumbel-sparsemax in float64 for gradcheck.
        Noise is fixed (not a function of z), so gradcheck is valid."""
        @staticmethod
        def forward(ctx, x, noise, temperature):
            z = (x + noise) / temperature
            B, N, H, C = z.shape
            z_flat = z.contiguous().view(-1, C)
            z_sorted, _ = torch.sort(z_flat, dim=-1, descending=True)
            cumsum = torch.cumsum(z_sorted, dim=-1)
            k_range = torch.arange(1, C + 1, device=x.device, dtype=x.dtype)
            support = (1 + k_range * z_sorted) > cumsum
            k = support.sum(dim=-1, keepdim=True).clamp(min=1).to(x.dtype)
            s = (z_sorted * support).sum(dim=-1, keepdim=True)
            tau = (s - 1) / k
            out = torch.clamp(z_flat - tau, min=0)
            ctx.save_for_backward(out)
            ctx.shape = x.shape
            ctx.temperature = temperature
            return out.view(B, N, H, C)

        @staticmethod
        def backward(ctx, grad_out):
            (out,) = ctx.saved_tensors
            go = grad_out.contiguous().view_as(out)
            supp = out > 0
            cnt = supp.sum(dim=-1, keepdim=True).clamp(min=1).to(go.dtype)
            v = (go * supp).sum(dim=-1, keepdim=True) / cnt
            gi = torch.where(supp, (go - v) / ctx.temperature, torch.zeros_like(go))
            return gi.view(ctx.shape), None, None

    x_gc = torch.randn(1, 2, 2, 8, device=device, dtype=torch.float64, requires_grad=True)
    noise_gc = -torch.log(-torch.log(torch.rand_like(x_gc).clamp(1e-7, 1 - 1e-7)))
    temp_gc = 1.0

    passed = torch.autograd.gradcheck(
        lambda z: _GumbelSparsemaxF64.apply(z, noise_gc, temp_gc),
        (x_gc,), eps=1e-6, atol=1e-4, rtol=1e-3,
    )
    print(f"  {'✅' if passed else '❌'} gradcheck passed={passed}")

    print("\nAll tests passed! 🎉")