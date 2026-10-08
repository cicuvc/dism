import torch
import triton
import triton.language as tl


def _check_cuda(x: torch.Tensor, name: str):
    if not x.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")


def _check_dtype(x: torch.Tensor, name: str, allowed):
    if x.dtype not in allowed:
        allowed_str = ", ".join(str(a) for a in allowed)
        raise ValueError(f"{name} dtype must be one of {{{allowed_str}}}, got {x.dtype}")


def _check_row_major_2d(x: torch.Tensor, name: str):
    if x.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape={tuple(x.shape)}")
    if x.stride(1) != 1:
        raise ValueError(
            f"{name} must have stride(1) == 1 (last dim contiguous), got stride={x.stride()}"
        )
    if x.stride(0) < x.shape[1]:
        raise ValueError(
            f"{name} must be non-overlapping row-major-like, got shape={tuple(x.shape)}, "
            f"stride={x.stride()}"
        )


def _check_contiguous_lastdim_3d(x: torch.Tensor, name: str):
    if x.ndim != 3:
        raise ValueError(f"{name} must be 3D, got shape={tuple(x.shape)}")
    if x.stride(-1) != 1:
        raise ValueError(
            f"{name} last dim must be contiguous, stride(-1) must be 1, got stride={x.stride()}"
        )


@triton.autotune(
    configs=[
        triton.Config({"BM": 16, "BV": 64,  "BK": 32}, num_warps=4, num_stages=2),
        triton.Config({"BM": 32, "BV": 64,  "BK": 32}, num_warps=4, num_stages=2),
        triton.Config({"BM": 16, "BV": 128, "BK": 32}, num_warps=4, num_stages=2),
        triton.Config({"BM": 32, "BV": 128, "BK": 32}, num_warps=8, num_stages=2),
        triton.Config({"BM": 16, "BV": 64,  "BK": 64}, num_warps=4, num_stages=2),
        triton.Config({"BM": 32, "BV": 64,  "BK": 64}, num_warps=8, num_stages=2),
        triton.Config({"BM": 16, "BV": 128, "BK": 64}, num_warps=8, num_stages=2),
    ],
    key=["M", "C", "V"],
)
@triton.jit
def stage1_logsumexp_kernel(
    x_ptr,       # [M, C]
    w_ptr,       # [V, C]
    lse_ptr,     # [M]
    M,
    C,
    V,
    stride_xm,
    stride_xc,
    stride_wv,
    stride_wc,
    stride_lm,
    BM: tl.constexpr,
    BV: tl.constexpr,
    BK: tl.constexpr,
):
    pid_m = tl.program_id(0)

    offs_m = pid_m * BM + tl.arange(0, BM)
    mask_m = offs_m < M

    neg_inf = -float("inf")
    m_i = tl.full([BM], neg_inf, tl.float32)
    s_i = tl.zeros([BM], tl.float32)

    for v_start in tl.range(0, V, BV):
        offs_v = v_start + tl.arange(0, BV)
        mask_v = offs_v < V

        acc = tl.zeros([BM, BV], dtype=tl.float32)

        for k_start in tl.range(0, C, BK):
            offs_k = k_start + tl.arange(0, BK)
            mask_k = offs_k < C

            x = tl.load(
                x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xc,
                mask=mask_m[:, None] & mask_k[None, :],
                other=0.0,
            )  # [BM, BK]

            w = tl.load(
                w_ptr + offs_v[:, None] * stride_wv + offs_k[None, :] * stride_wc,
                mask=mask_v[:, None] & mask_k[None, :],
                other=0.0,
            )  # [BV, BK]

            acc += tl.dot(x, tl.trans(w))

        acc = tl.where(mask_m[:, None] & mask_v[None, :], acc, neg_inf)

        block_max = tl.max(acc, axis=1)
        new_m = tl.maximum(m_i, block_max)

        old_scale = tl.exp(m_i - new_m)
        block_exp = tl.exp(acc - new_m[:, None])
        block_sum = tl.sum(block_exp, axis=1)

        s_i = s_i * old_scale + block_sum
        m_i = new_m

    lse = m_i + tl.log(s_i)
    tl.store(lse_ptr + offs_m * stride_lm, lse, mask=mask_m)


@triton.autotune(
    configs=[
        triton.Config({"BM": 32,  "BK": 32}, num_warps=4, num_stages=2),
        triton.Config({"BM": 64,  "BK": 32}, num_warps=4, num_stages=2),
        triton.Config({"BM": 32,  "BK": 64}, num_warps=4, num_stages=2),
        triton.Config({"BM": 64,  "BK": 64}, num_warps=8, num_stages=2),
        triton.Config({"BM": 128, "BK": 32}, num_warps=8, num_stages=2),
        triton.Config({"BM": 128, "BK": 64}, num_warps=8, num_stages=2),
    ],
    key=["M", "C"],
)
@triton.jit
def stage2_target_nll_kernel(
    x_ptr,       # [M, C]
    w_ptr,       # [V, C]
    labels_ptr,  # [M]
    lse_ptr,     # [M]
    nll_ptr,     # [M]
    M,
    C,
    V,
    stride_xm,
    stride_xc,
    stride_wv,
    stride_wc,
    stride_lab,
    stride_lse,
    stride_nll,
    BM: tl.constexpr,
    BK: tl.constexpr,
):
    pid_m = tl.program_id(0)

    offs_m = pid_m * BM + tl.arange(0, BM)
    mask_m = offs_m < M

    labels = tl.load(labels_ptr + offs_m * stride_lab, mask=mask_m, other=0).to(tl.int32)
    lse = tl.load(lse_ptr + offs_m * stride_lse, mask=mask_m, other=0.0)

    valid_label = labels < V
    acc = tl.zeros([BM], dtype=tl.float32)

    for k_start in tl.range(0, C, BK):
        offs_k = k_start + tl.arange(0, BK)
        mask_k = offs_k < C

        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xc,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )  # [BM, BK]

        w = tl.load(
            w_ptr + labels[:, None] * stride_wv + offs_k[None, :] * stride_wc,
            mask=mask_m[:, None] & valid_label[:, None] & mask_k[None, :],
            other=0.0,
        )  # [BM, BK]

        acc += tl.sum(x * w, axis=1)

    nll = lse - acc
    tl.store(nll_ptr + offs_m * stride_nll, nll, mask=mask_m)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_B": 32}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_B": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_B": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_B": 256}, num_warps=8, num_stages=2),
    ],
    key=["B", "N"],
)
@triton.jit
def mean_over_batch_kernel(
    nll_ptr,   # [B, N]
    out_ptr,   # [N]
    B,
    N,
    stride_nb,
    stride_nn,
    stride_on,
    BLOCK_B: tl.constexpr,
):
    n = tl.program_id(0)
    if n >= N:
        return

    acc = 0.0
    for b_start in tl.range(0, B, BLOCK_B):
        offs_b = b_start + tl.arange(0, BLOCK_B)
        mask_b = offs_b < B

        vals = tl.load(
            nll_ptr + offs_b * stride_nb + n * stride_nn,
            mask=mask_b,
            other=0.0,
        )
        acc += tl.sum(vals, axis=0)

    acc = acc / B
    tl.store(out_ptr + n * stride_on, acc)


def per_position_nll_triton_final(
    hidden_states: torch.Tensor,   # [B, N, C]
    lm_head_weight: torch.Tensor,  # [VOCAB, C]
    labels: torch.Tensor,          # [B, N]
    *,
    enforce_strict_layout: bool = True,
):
    """
    默认输入布局:
      hidden_states: [B, N, C]
      lm_head_weight: [VOCAB, C]
      labels: [B, N]

    计算:
      logits[b, n, v] = dot(hidden_states[b, n, :], lm_head_weight[v, :])
      nll[b, n] = -log_softmax(logits[b, n, :])[labels[b, n]]
      out[n] = mean_b nll[b, n]

    返回:
      out: [N] float32

    性能推荐:
      lm_head_weight 应为 [VOCAB, C] 且 contiguous，尤其 stride(1) == 1
    """
    _check_cuda(hidden_states, "hidden_states")
    _check_cuda(lm_head_weight, "lm_head_weight")
    _check_cuda(labels, "labels")

    _check_dtype(hidden_states, "hidden_states", (torch.float16, torch.bfloat16, torch.float32))
    _check_dtype(lm_head_weight, "lm_head_weight", (torch.float16, torch.bfloat16, torch.float32))
    _check_dtype(labels, "labels", (torch.int32, torch.int64))

    if hidden_states.ndim != 3:
        raise ValueError(f"hidden_states must be [B, N, C], got shape={tuple(hidden_states.shape)}")
    if lm_head_weight.ndim != 2:
        raise ValueError(
            f"lm_head_weight must be [VOCAB, C], got shape={tuple(lm_head_weight.shape)}"
        )
    if labels.ndim != 2:
        raise ValueError(f"labels must be [B, N], got shape={tuple(labels.shape)}")

    B, N, C = hidden_states.shape
    V, Cw = lm_head_weight.shape
    if C != Cw:
        raise ValueError(
            f"C mismatch: hidden_states.shape[-1]={C}, lm_head_weight.shape[-1]={Cw}"
        )
    if labels.shape != (B, N):
        raise ValueError(f"labels shape must be {(B, N)}, got {tuple(labels.shape)}")

    _check_contiguous_lastdim_3d(hidden_states, "hidden_states")

    if enforce_strict_layout:
        _check_row_major_2d(lm_head_weight, "lm_head_weight")
        w = lm_head_weight
    else:
        # 自动整理成 kernel 友好的 [V, C] contiguous
        if lm_head_weight.stride(1) != 1 or not lm_head_weight.is_contiguous():
            w = lm_head_weight.contiguous()
        else:
            w = lm_head_weight

    if w.stride(1) != 1:
        raise ValueError(
            f"lm_head_weight must have contiguous last dim for performance; got stride={w.stride()}"
        )

    M = B * N
    x = hidden_states.reshape(M, C).contiguous()
    lab = labels.reshape(M).contiguous()

    lse = torch.empty((M,), device=hidden_states.device, dtype=torch.float32)
    nll = torch.empty((M,), device=hidden_states.device, dtype=torch.float32)
    out = torch.empty((N,), device=hidden_states.device, dtype=torch.float32)

    grid1 = (triton.cdiv(M, 16),)
    stage1_logsumexp_kernel[grid1](
        x_ptr=x,
        w_ptr=w,
        lse_ptr=lse,
        M=M,
        C=C,
        V=V,
        stride_xm=x.stride(0),
        stride_xc=x.stride(1),
        stride_wv=w.stride(0),
        stride_wc=w.stride(1),
        stride_lm=lse.stride(0),
    )

    grid2 = (triton.cdiv(M, 32),)
    stage2_target_nll_kernel[grid2](
        x_ptr=x,
        w_ptr=w,
        labels_ptr=lab,
        lse_ptr=lse,
        nll_ptr=nll,
        M=M,
        C=C,
        V=V,
        stride_xm=x.stride(0),
        stride_xc=x.stride(1),
        stride_wv=w.stride(0),
        stride_wc=w.stride(1),
        stride_lab=lab.stride(0),
        stride_lse=lse.stride(0),
        stride_nll=nll.stride(0),
    )

    nll_2d = nll.view(B, N)
    grid3 = (N,)
    mean_over_batch_kernel[grid3](
        nll_ptr=nll_2d,
        out_ptr=out,
        B=B,
        N=N,
        stride_nb=nll_2d.stride(0),
        stride_nn=nll_2d.stride(1),
        stride_on=out.stride(0),
    )

    return out


def per_position_nll_triton_from_cv(
    hidden_states: torch.Tensor,   # [B, N, C]
    lm_head_weight_cv: torch.Tensor,  # [C, VOCAB]
    labels: torch.Tensor,          # [B, N]
    *,
    enforce_strict_layout: bool = True,
):
    """
    兼容旧格式 lm_head_weight=[C, VOCAB]。
    内部转成默认格式 [VOCAB, C] 再调用主实现。
    """
    _check_cuda(lm_head_weight_cv, "lm_head_weight_cv")
    _check_dtype(
        lm_head_weight_cv,
        "lm_head_weight_cv",
        (torch.float16, torch.bfloat16, torch.float32),
    )

    if lm_head_weight_cv.ndim != 2:
        raise ValueError(
            f"lm_head_weight_cv must be [C, VOCAB], got shape={tuple(lm_head_weight_cv.shape)}"
        )

    lm_head_weight = lm_head_weight_cv.transpose(0, 1).contiguous()
    return per_position_nll_triton_final(
        hidden_states,
        lm_head_weight,
        labels,
        enforce_strict_layout=enforce_strict_layout,
    )


def per_position_nll_reference(
    hidden_states: torch.Tensor,   # [B, N, C]
    lm_head_weight: torch.Tensor,  # [VOCAB, C]
    labels: torch.Tensor,          # [B, N]
):
    logits = torch.einsum("bnc,vc->bnv", hidden_states.float(), lm_head_weight.float())
    log_probs = torch.log_softmax(logits, dim=-1)
    nll = -torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return nll.mean(dim=0)