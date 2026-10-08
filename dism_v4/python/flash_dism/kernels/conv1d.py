import math

import torch
import triton
import triton.language as tl

if __package__:
    from .dynamo_utils import mark_cu_seqlens_dynamic
else:
    from dynamo_utils import mark_cu_seqlens_dynamic


def _short_conv_configs():
    return [
        triton.Config(
            {"N_CDIM_SIZE": 64, "N_SEQDIM_SIZE": 64, "N_RED_KSIZE": 64},
            num_warps=4, num_stages=3),
        triton.Config(
            {"N_CDIM_SIZE": 64, "N_SEQDIM_SIZE": 32, "N_RED_KSIZE": 64},
            num_warps=4, num_stages=3),
        triton.Config(
            {"N_CDIM_SIZE": 32, "N_SEQDIM_SIZE": 64, "N_RED_KSIZE": 32},
            num_warps=4, num_stages=4),
        triton.Config(
            {"N_CDIM_SIZE": 128, "N_SEQDIM_SIZE": 32, "N_RED_KSIZE": 64},
            num_warps=8, num_stages=3),
    ]


def _short_conv_bwd_configs():
    # N_SEQDIM_SIZE is part of the dmma_halo layout consumed by the following
    # GEMM kernel, so keep it fixed while tuning the channel and reduction tiles.
    return [
        triton.Config(
            {"N_CDIM_SIZE": 64, "N_SEQDIM_SIZE": 64, "N_RED_KSIZE": 64},
            num_warps=4, num_stages=3),
        triton.Config(
            {"N_CDIM_SIZE": 32, "N_SEQDIM_SIZE": 64, "N_RED_KSIZE": 32},
            num_warps=4, num_stages=4),
        triton.Config(
            {"N_CDIM_SIZE": 128, "N_SEQDIM_SIZE": 64, "N_RED_KSIZE": 64},
            num_warps=8, num_stages=2),
    ]


def _short_conv_gemm_configs():
    return [
        triton.Config(
            {"BLOCK_SEQ": 32, "BLOCK_IN": 32, "BLOCK_OUT": 32},
            num_warps=8, num_stages=3),
        triton.Config(
            {"BLOCK_SEQ": 16, "BLOCK_IN": 32, "BLOCK_OUT": 32},
            num_warps=4, num_stages=4),
        triton.Config(
            {"BLOCK_SEQ": 32, "BLOCK_IN": 64, "BLOCK_OUT": 32},
            num_warps=8, num_stages=3),
        triton.Config(
            {"BLOCK_SEQ": 32, "BLOCK_IN": 32, "BLOCK_OUT": 64},
            num_warps=8, num_stages=3),
    ]

def ref_short_conv(x: torch.Tensor, weight: torch.Tensor, conv_weight: torch.Tensor):
    # x: [B, N, C]
    # weight: [D, C]
    # conv_weight: [CONV_SIZE, D]
    N = x.shape[1]
    CONV_SIZE = conv_weight.shape[0]
    x = torch.nn.functional.linear(x.float(), weight.float()) # [B, N, D]
    components = []
    idx = torch.arange(N, device=x.device)
    for i in range(CONV_SIZE):
        components.append(torch.where((idx >= i)[None,:,None], torch.roll(x, (i,),(1,)), torch.zeros_like(x)) * conv_weight[None,None,i,:])
    return torch.nn.functional.silu(sum(components))

@triton.autotune(
    configs=_short_conv_configs(),
    key=["n_seq", "in_channel", "out_channel", "N_CONV_SIZE",
         "IS_VARLEN", "HAS_STATE"],
    cache_results=True,
)
@triton.jit
def fwd_short_conv_silu(
    x, weight, conv_weight, cu_seqlens, input_state, o, output_state,
    b, n_seq, in_channel, out_channel,
    x_stride0, x_stride1, w_stride0, c_stride0, o_stride0, o_stride1,
    state_stride0, state_stride1,
    N_CDIM_SIZE: tl.constexpr, N_SEQDIM_SIZE: tl.constexpr, N_CONV_HALO_SIZE: tl.constexpr, N_RED_KSIZE: tl.constexpr, N_CONV_SIZE: tl.constexpr, N_STATE_SIZE: tl.constexpr,
    IS_VARLEN: tl.constexpr, HAS_STATE: tl.constexpr):

    pid = tl.program_id(0)
    pid_b = tl.program_id(1)

    channel_blocks = tl.cdiv(out_channel, N_CDIM_SIZE)
    k_red_steps = tl.cdiv(in_channel, N_RED_KSIZE)
    idx_seqblk = pid // channel_blocks
    idx_cblk = pid % channel_blocks

    seq_start = 0
    seq_len = n_seq
    x_batch_offset = pid_b * x_stride0
    o_batch_offset = pid_b * o_stride0
    if IS_VARLEN:
        seq_start = tl.load(cu_seqlens + pid_b)
        seq_len = tl.load(cu_seqlens + pid_b + 1) - seq_start
        x_batch_offset = seq_start * x_stride1
        o_batch_offset = seq_start * o_stride1

    x_block = tl.make_block_ptr(
        x + x_batch_offset,
        shape=(seq_len, in_channel),
        strides=(x_stride1, 1),
        offsets=(idx_seqblk * N_SEQDIM_SIZE, 0),
        block_shape=(N_SEQDIM_SIZE, N_RED_KSIZE),
        order=(1,0)
    )
    x_halo_block = tl.make_block_ptr(
        x + x_batch_offset,
        shape=(seq_len, in_channel),
        strides=(x_stride1, 1),
        offsets=(idx_seqblk * N_SEQDIM_SIZE - N_CONV_HALO_SIZE, 0),
        block_shape=(N_CONV_HALO_SIZE, N_RED_KSIZE),
        order=(1,0)
    )
    w_block = tl.make_block_ptr(
        weight,
        shape=(out_channel, in_channel),
        strides=(w_stride0, 1),
        offsets=(idx_cblk*N_CDIM_SIZE, 0),
        block_shape=(N_CDIM_SIZE,N_RED_KSIZE),
        order=(1,0)
    )

    acc_block = tl.zeros((N_CDIM_SIZE, N_SEQDIM_SIZE), dtype=tl.float32)
    acc_halo_block = tl.zeros((N_CDIM_SIZE, N_CONV_HALO_SIZE), dtype=tl.float32)

    for i in tl.range(0, k_red_steps):
        x_vals = tl.load(x_block, boundary_check=(0,1),padding_option='zero')
        x_halo_vals = tl.load(x_halo_block, boundary_check=(0,1),padding_option='zero')
        w_vals = tl.load(w_block, boundary_check=(0,1),padding_option='zero')

        acc_block = tl.dot(w_vals, tl.trans(x_vals), acc_block, input_precision='ieee')
        acc_halo_block = tl.dot(w_vals, tl.trans(x_halo_vals), acc_halo_block, input_precision='ieee')

        x_block = tl.advance(x_block, (0, N_RED_KSIZE))
        w_block = tl.advance(w_block, (0, N_RED_KSIZE))
        x_halo_block = tl.advance(x_halo_block, (0, N_RED_KSIZE))

    if HAS_STATE:
        hidx = tl.arange(0, N_CONV_HALO_SIZE)
        state_cidx = tl.arange(0, N_CDIM_SIZE) + idx_cblk * N_CDIM_SIZE
        state_row = tl.maximum(hidx - (N_CONV_HALO_SIZE - N_STATE_SIZE), 0)
        state_vals = tl.load(
            input_state + pid_b * state_stride0
            + state_cidx[:, None] + state_row[None, :] * state_stride1,
            mask=(state_cidx < out_channel)[:, None]
            & (hidx >= N_CONV_HALO_SIZE - N_STATE_SIZE)[None, :],
            other=0.0)
        acc_halo_block = tl.where(idx_seqblk == 0, state_vals, acc_halo_block)

    conv_acc = tl.zeros_like(acc_block) # [N_CDIM_SIZE, N_SEQDIM_SIZE]

    idx = tl.arange(0, N_SEQDIM_SIZE)
    cidx = tl.arange(0, N_CDIM_SIZE)
    for i in tl.static_range(0, N_CONV_SIZE):
        conv_weight_vals = tl.load(conv_weight + c_stride0 * i + N_CDIM_SIZE * idx_cblk + cidx, mask=(N_CDIM_SIZE * idx_cblk + cidx < out_channel), other=0.0)
        # gather lowers through shared memory for these layouts, so both sides
        # of the select must have an in-bounds index.
        current_idx = tl.maximum(idx - i, 0)
        halo_idx = tl.minimum(idx - i + N_CONV_HALO_SIZE, N_CONV_HALO_SIZE - 1)
        roll_acc = tl.where((idx - i >= 0)[None,:], tl.gather(acc_block, tl.broadcast_to(current_idx[None,:], (N_CDIM_SIZE, N_SEQDIM_SIZE)), 1), tl.gather(acc_halo_block, tl.broadcast_to(halo_idx[None,:], (N_CDIM_SIZE, N_SEQDIM_SIZE)), 1)) # [N_CDIM_SIZE, N_SEQDIM_SIZE]
        conv_acc += roll_acc * conv_weight_vals[:,None]

    conv_acc = tl.sigmoid(conv_acc) * conv_acc
    o_block = tl.make_block_ptr(
        o + o_batch_offset,
        shape=(seq_len, out_channel),
        strides=(o_stride1, 1),
        offsets=(idx_seqblk*N_SEQDIM_SIZE, idx_cblk*N_CDIM_SIZE),
        block_shape=(N_SEQDIM_SIZE,N_CDIM_SIZE),
        order=(1,0)
    )
    tl.store(o_block, tl.trans(conv_acc.to(o.dtype.element_ty)), boundary_check=(0,1))

    if HAS_STATE:
        state_idx = tl.arange(0, N_CONV_HALO_SIZE)
        state_idx_mask = state_idx < N_STATE_SIZE
        state_channel = idx_cblk * N_CDIM_SIZE + cidx
        state_time = seq_len - N_STATE_SIZE + state_idx
        local_time_unclamped = state_time - idx_seqblk * N_SEQDIM_SIZE
        local_time = tl.maximum(
            tl.minimum(local_time_unclamped, N_SEQDIM_SIZE - 1), 0)
        current_state = tl.gather(
            acc_block,
            tl.broadcast_to(local_time[None, :], (N_CDIM_SIZE, N_CONV_HALO_SIZE)),
            1)
        halo_time = tl.maximum(
            tl.minimum(local_time_unclamped + N_CONV_HALO_SIZE,
                       N_CONV_HALO_SIZE - 1), 0)
        halo_state = tl.gather(
            acc_halo_block,
            tl.broadcast_to(halo_time[None, :], (N_CDIM_SIZE, N_CONV_HALO_SIZE)),
            1)
        projected_state = tl.where(
            (local_time_unclamped >= 0)[None, :], current_state, halo_state)
        old_state_row = tl.maximum(seq_len + state_idx, 0)
        old_state = tl.load(
            input_state + pid_b * state_stride0
            + state_channel[:, None] + old_state_row[None, :] * state_stride1,
            mask=(state_channel < out_channel)[:, None]
            & state_idx_mask[None, :] & (state_time < 0)[None, :],
            other=0.0)
        new_state = tl.where((state_time >= 0)[None, :], projected_state, old_state)
        last_seqblk = tl.cdiv(seq_len, N_SEQDIM_SIZE) - 1
        tl.store(
            output_state + pid_b * state_stride0
            + state_idx[:, None] * state_stride1 + state_channel[None, :],
            tl.trans(new_state).to(output_state.dtype.element_ty),
            mask=(idx_seqblk == last_seqblk) & (seq_len > 0)
            & state_idx_mask[:, None]
            & (state_channel < out_channel)[None, :])


@triton.autotune(
    configs=_short_conv_bwd_configs(),
    key=["n_seq", "in_channel", "out_channel", "N_CONV_SIZE",
         "IS_VARLEN", "HAS_STATE"],
    reset_to_zero=["dconv_weight"],
    cache_results=True,
)
@triton.jit
def bwd_short_conv_silu(
    x, weight, conv_weight, cu_seqlens, input_state, do, doutput_state,
    dmma, dmma_halo, dconv_weight, dinput_state,
    b, n_seq, in_channel, out_channel,
    x_stride0, x_stride1, w_stride0, c_stride0, do_stride0, do_stride1, dmma_stride0, dmma_stride1, dmma_halo_stride0, dmma_halo_stride1,
    state_stride0, state_stride1,
    N_CDIM_SIZE: tl.constexpr, N_SEQDIM_SIZE: tl.constexpr, N_CONV_HALO_SIZE: tl.constexpr, N_RED_KSIZE: tl.constexpr, N_CONV_SIZE: tl.constexpr, N_STATE_SIZE: tl.constexpr,
    IS_VARLEN: tl.constexpr, HAS_STATE: tl.constexpr):

    pid = tl.program_id(0)
    pid_b = tl.program_id(1)

    channel_blocks = tl.cdiv(out_channel, N_CDIM_SIZE)
    k_red_steps = tl.cdiv(in_channel, N_RED_KSIZE)
    idx_seqblk = pid // channel_blocks
    idx_cblk = pid % channel_blocks

    num_seqblk = tl.num_programs(0) // channel_blocks

    seq_start = 0
    seq_len = n_seq
    x_batch_offset = pid_b * x_stride0
    do_batch_offset = pid_b * do_stride0
    dmma_batch_offset = pid_b * dmma_stride0
    if IS_VARLEN:
        seq_start = tl.load(cu_seqlens + pid_b)
        seq_len = tl.load(cu_seqlens + pid_b + 1) - seq_start
        x_batch_offset = seq_start * x_stride1
        do_batch_offset = seq_start * do_stride1
        dmma_batch_offset = seq_start * dmma_stride1

    x_block = tl.make_block_ptr(
        x + x_batch_offset,
        shape=(seq_len, in_channel),
        strides=(x_stride1, 1),
        offsets=(idx_seqblk * N_SEQDIM_SIZE, 0),
        block_shape=(N_SEQDIM_SIZE, N_RED_KSIZE),
        order=(1,0)
    )
    x_halo_block = tl.make_block_ptr(
        x + x_batch_offset,
        shape=(seq_len, in_channel),
        strides=(x_stride1, 1),
        offsets=(idx_seqblk * N_SEQDIM_SIZE - N_CONV_HALO_SIZE, 0),
        block_shape=(N_CONV_HALO_SIZE, N_RED_KSIZE),
        order=(1,0)
    )
    w_block = tl.make_block_ptr(
        weight,
        shape=(out_channel, in_channel),
        strides=(w_stride0, 1),
        offsets=(idx_cblk*N_CDIM_SIZE, 0),
        block_shape=(N_CDIM_SIZE,N_RED_KSIZE),
        order=(1,0)
    )

    acc_block = tl.zeros((N_CDIM_SIZE, N_SEQDIM_SIZE), dtype=tl.float32)
    acc_halo_block = tl.zeros((N_CDIM_SIZE, N_CONV_HALO_SIZE), dtype=tl.float32)

    for i in tl.range(0, k_red_steps):
        x_vals = tl.load(x_block, boundary_check=(0,1),padding_option='zero')
        x_halo_vals = tl.load(x_halo_block, boundary_check=(0,1),padding_option='zero')
        w_vals = tl.load(w_block, boundary_check=(0,1),padding_option='zero')

        acc_block = tl.dot(w_vals, tl.trans(x_vals), acc_block, input_precision='ieee')
        acc_halo_block = tl.dot(w_vals, tl.trans(x_halo_vals), acc_halo_block, input_precision='ieee')

        x_block = tl.advance(x_block, (0, N_RED_KSIZE))
        w_block = tl.advance(w_block, (0, N_RED_KSIZE))
        x_halo_block = tl.advance(x_halo_block, (0, N_RED_KSIZE))

    if HAS_STATE:
        state_hidx = tl.arange(0, N_CONV_HALO_SIZE)
        state_channel = idx_cblk * N_CDIM_SIZE + tl.arange(0, N_CDIM_SIZE)
        state_row = tl.maximum(
            state_hidx - (N_CONV_HALO_SIZE - N_STATE_SIZE), 0)
        state_vals = tl.load(
            input_state + pid_b * state_stride0
            + state_channel[:, None] + state_row[None, :] * state_stride1,
            mask=(state_channel < out_channel)[:, None]
            & (state_hidx >= N_CONV_HALO_SIZE - N_STATE_SIZE)[None, :],
            other=0.0)
        acc_halo_block = tl.where(idx_seqblk == 0, state_vals, acc_halo_block)

    conv_acc = tl.zeros_like(acc_block) # [N_CDIM_SIZE, N_SEQDIM_SIZE]

    idx = tl.arange(0, N_SEQDIM_SIZE)
    cidx = tl.arange(0, N_CDIM_SIZE)
    for i in tl.static_range(0, N_CONV_SIZE):
        conv_weight_vals = tl.load(conv_weight + c_stride0 * i + N_CDIM_SIZE * idx_cblk + cidx, mask=(N_CDIM_SIZE * idx_cblk + cidx < out_channel), other=0.0)
        # See the forward kernel: the shared-memory gather needs valid indices
        # even for values discarded by the following select.
        current_idx = tl.maximum(idx - i, 0)
        halo_idx = tl.minimum(idx - i + N_CONV_HALO_SIZE, N_CONV_HALO_SIZE - 1)
        roll_acc = tl.where((idx - i >= 0)[None,:], tl.gather(acc_block, tl.broadcast_to(current_idx[None,:], (N_CDIM_SIZE, N_SEQDIM_SIZE)), 1), tl.gather(acc_halo_block, tl.broadcast_to(halo_idx[None,:], (N_CDIM_SIZE, N_SEQDIM_SIZE)), 1)) # [N_CDIM_SIZE, N_SEQDIM_SIZE]
        conv_acc += roll_acc * conv_weight_vals[:,None]

    do_vals = tl.trans(tl.load(tl.make_block_ptr(
        do + do_batch_offset,
        shape=(seq_len, out_channel),
        strides=(do_stride1, 1),
        offsets=(idx_seqblk*N_SEQDIM_SIZE, idx_cblk*N_CDIM_SIZE),
        block_shape=(N_SEQDIM_SIZE,N_CDIM_SIZE),
        order=(1,0)
    ), boundary_check=(0,1),padding_option="zero")) # [N_CDIM_SIZE, N_SEQDIM_SIZE]

    sx = tl.sigmoid(conv_acc)
    dconv = do_vals * ((-conv_acc*sx+conv_acc)*sx+sx) # 2 FFMA

    mma_grad = tl.zeros_like(dconv)
    mma_halo_grad = tl.zeros_like(acc_halo_block)

    hidx = tl.arange(0, N_CONV_HALO_SIZE)

    for i in tl.static_range(0, N_CONV_SIZE):
        conv_weight_vals = tl.load(conv_weight + c_stride0 * i + N_CDIM_SIZE * idx_cblk + cidx, mask=(N_CDIM_SIZE * idx_cblk + cidx < out_channel), other=0.0)
        mma_idx = tl.minimum(idx + i, N_SEQDIM_SIZE - 1)
        mma_grad += conv_weight_vals[:,None] * tl.where((idx + i < N_SEQDIM_SIZE)[None,:], tl.gather(dconv, tl.broadcast_to(mma_idx[None, :], (N_CDIM_SIZE, N_SEQDIM_SIZE)), 1), tl.zeros_like(dconv))
        dconv_halo_idx = tl.maximum(hidx + i - N_CONV_HALO_SIZE, 0)
        mma_halo_grad += conv_weight_vals[:, None] * tl.where((hidx + i >= N_CONV_HALO_SIZE)[None, :], tl.gather(dconv, tl.broadcast_to(dconv_halo_idx[None, :], (N_CDIM_SIZE, N_CONV_HALO_SIZE)), 1), tl.zeros_like(mma_halo_grad))
        current_idx = tl.maximum(idx - i, 0)
        input_halo_idx = tl.minimum(idx - i + N_CONV_HALO_SIZE, N_CONV_HALO_SIZE - 1)
        roll_acc = tl.where((idx - i >= 0)[None,:], tl.gather(acc_block, tl.broadcast_to(current_idx[None,:], (N_CDIM_SIZE, N_SEQDIM_SIZE)), 1), tl.gather(acc_halo_block, tl.broadcast_to(input_halo_idx[None,:], (N_CDIM_SIZE, N_SEQDIM_SIZE)), 1))
        dconv_weight_vals = tl.sum(roll_acc * dconv, axis=1, keep_dims=False)
        tl.atomic_add(dconv_weight + c_stride0 * i + N_CDIM_SIZE * idx_cblk + cidx, dconv_weight_vals, mask=(N_CDIM_SIZE * idx_cblk + cidx < out_channel))

    if HAS_STATE:
        state_channel = idx_cblk * N_CDIM_SIZE + cidx
        global_time = idx_seqblk * N_SEQDIM_SIZE + idx
        output_state_row = global_time - seq_len + N_STATE_SIZE
        safe_output_state_row = tl.maximum(
            tl.minimum(output_state_row, N_STATE_SIZE - 1), 0)
        state_to_mma = tl.load(
            doutput_state + pid_b * state_stride0
            + state_channel[:, None]
            + safe_output_state_row[None, :] * state_stride1,
            mask=(state_channel < out_channel)[:, None]
            & (output_state_row >= 0)[None, :]
            & (output_state_row < N_STATE_SIZE)[None, :],
            other=0.0)
        mma_grad += state_to_mma

        state_idx = tl.arange(0, N_CONV_HALO_SIZE)
        state_idx_mask = state_idx < N_STATE_SIZE
        halo_idx = tl.minimum(
            N_CONV_HALO_SIZE - N_STATE_SIZE + state_idx,
            N_CONV_HALO_SIZE - 1)
        conv_state_grad = tl.gather(
            mma_halo_grad,
            tl.broadcast_to(halo_idx[None, :], (N_CDIM_SIZE, N_CONV_HALO_SIZE)),
            1)
        carried_output_row = state_idx - seq_len
        safe_carried_row = tl.maximum(carried_output_row, 0)
        carried_state_grad = tl.load(
            doutput_state + pid_b * state_stride0
            + state_channel[:, None]
            + safe_carried_row[None, :] * state_stride1,
            mask=(state_channel < out_channel)[:, None]
            & state_idx_mask[None, :]
            & (carried_output_row >= 0)[None, :],
            other=0.0)
        input_state_grad = conv_state_grad + carried_state_grad
        tl.store(
            dinput_state + pid_b * state_stride0
            + state_idx[:, None] * state_stride1 + state_channel[None, :],
            tl.trans(input_state_grad).to(dinput_state.dtype.element_ty),
            mask=(idx_seqblk == 0) & state_idx_mask[:, None]
            & (state_channel < out_channel)[None, :])

    dmma_block = tl.make_block_ptr(
        dmma + dmma_batch_offset,
        shape=(seq_len, out_channel),
        strides=(dmma_stride1, 1),
        offsets=(idx_seqblk*N_SEQDIM_SIZE, idx_cblk*N_CDIM_SIZE),
        block_shape=(N_SEQDIM_SIZE,N_CDIM_SIZE),
        order=(1,0)
    )
    tl.store(dmma_block, tl.trans(mma_grad.to(dmma.dtype.element_ty)), boundary_check=(0,1))
    dmma_halo_block = tl.make_block_ptr(
        dmma_halo + pid_b * dmma_halo_stride0,
        shape=(num_seqblk * N_CONV_HALO_SIZE, out_channel),
        strides=(dmma_halo_stride1, 1),
        offsets=(idx_seqblk*N_CONV_HALO_SIZE, idx_cblk*N_CDIM_SIZE),
        block_shape=(N_CONV_HALO_SIZE,N_CDIM_SIZE),
        order=(1,0)
    )
    tl.store(dmma_halo_block, tl.trans(mma_halo_grad.to(dmma_halo.dtype.element_ty)), boundary_check=(0,1))


@triton.autotune(
    configs=_short_conv_gemm_configs(),
    key=["n_seq", "in_channel", "out_channel", "IS_VARLEN"],
    reset_to_zero=["dweight"],
    cache_results=True,
)
@triton.jit
def bwd_mma_gemm(
    x, weight, dmma, dmma_halo, cu_seqlens, dx, dweight,
    b, n_seq, in_channel, out_channel,
    x_stride0, x_stride1, w_stride0,
    dmma_stride0, dmma_stride1,
    dmma_halo_stride0, dmma_halo_stride1,
    dx_stride0, dx_stride1, dw_stride0,
    N_SEQDIM_SIZE: tl.constexpr, N_CONV_HALO_SIZE: tl.constexpr,
    BLOCK_SEQ: tl.constexpr, BLOCK_IN: tl.constexpr,
    BLOCK_OUT: tl.constexpr, IS_VARLEN: tl.constexpr):
    """Backpropagate the projection GEMM, folding the next block's halo into dz."""
    pid = tl.program_id(0)
    pid_b = tl.program_id(1)

    num_seq_tiles = tl.cdiv(n_seq, BLOCK_SEQ)
    idx_cblk = pid // num_seq_tiles
    idx_seq_tile = pid % num_seq_tiles

    seq_start = 0
    seq_len = n_seq
    x_batch_offset = pid_b * x_stride0
    dmma_batch_offset = pid_b * dmma_stride0
    dx_batch_offset = pid_b * dx_stride0
    if IS_VARLEN:
        seq_start = tl.load(cu_seqlens + pid_b)
        seq_len = tl.load(cu_seqlens + pid_b + 1) - seq_start
        x_batch_offset = seq_start * x_stride1
        dmma_batch_offset = seq_start * dmma_stride1
        dx_batch_offset = seq_start * dx_stride1

    seq = idx_seq_tile * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
    c = idx_cblk * BLOCK_IN + tl.arange(0, BLOCK_IN)
    seq_mask = seq < seq_len
    c_mask = c < in_channel

    x_vals = tl.load(
        x + x_batch_offset + seq[:, None] * x_stride1 + c[None, :],
        mask=seq_mask[:, None] & c_mask[None, :], other=0.0)
    dx_acc = tl.zeros((BLOCK_SEQ, BLOCK_IN), tl.float32)

    # A forward block owns dmma for its regular rows. Its halo rows belong to
    # the tail of the preceding block, so dz includes the next block's halo.
    fwd_block = seq // N_SEQDIM_SIZE
    local_seq = seq % N_SEQDIM_SIZE
    num_fwd_blocks = tl.cdiv(seq_len, N_SEQDIM_SIZE)
    halo_row = tl.maximum(local_seq - (N_SEQDIM_SIZE - N_CONV_HALO_SIZE), 0)

    for out_start in tl.range(0, out_channel, BLOCK_OUT):
        d = out_start + tl.arange(0, BLOCK_OUT)
        d_mask = d < out_channel
        dz_vals = tl.load(
            dmma + dmma_batch_offset + seq[:, None] * dmma_stride1 + d[None, :],
            mask=seq_mask[:, None] & d_mask[None, :], other=0.0)
        halo_vals = tl.load(
            dmma_halo + pid_b * dmma_halo_stride0
            + ((fwd_block + 1) * N_CONV_HALO_SIZE + halo_row)[:, None] * dmma_halo_stride1
            + d[None, :],
            mask=seq_mask[:, None]
            & (local_seq >= N_SEQDIM_SIZE - N_CONV_HALO_SIZE)[:, None]
            & (fwd_block + 1 < num_fwd_blocks)[:, None]
            & d_mask[None, :],
            other=0.0)
        dz_vals += halo_vals

        w_vals = tl.load(
            weight + d[:, None] * w_stride0 + c[None, :],
            mask=d_mask[:, None] & c_mask[None, :], other=0.0)
        dx_acc = tl.dot(dz_vals, w_vals, dx_acc, input_precision='ieee')

        dw_vals = tl.dot(tl.trans(dz_vals), x_vals, input_precision='ieee')
        tl.atomic_add(
            dweight + d[:, None] * dw_stride0 + c[None, :], dw_vals,
            mask=d_mask[:, None] & c_mask[None, :])

    tl.store(
        dx + dx_batch_offset + seq[:, None] * dx_stride1 + c[None, :],
        dx_acc.to(dx.dtype.element_ty),
        mask=seq_mask[:, None] & c_mask[None, :])


def torch_fwd_short_conv_silu(
    x, weight, conv_weight, cu_seqlens=None, max_seqlen=None,
    input_state=None,
):
    if cu_seqlens is not None:
        return torch_fwd_varlen_short_conv_silu(
            x, weight, conv_weight, cu_seqlens, max_seqlen, input_state)
    assert x.stride(-1) == 1 and weight.stride(-1) == 1 and conv_weight.stride(-1) == 1

    B, N, C = x.shape
    D = weight.shape[0]
    CONV_SIZE = conv_weight.shape[0]
    N_CONV_HALO_SIZE = 16
    assert CONV_SIZE <= N_CONV_HALO_SIZE + 1
    has_state = input_state is not None and CONV_SIZE > 1
    if input_state is not None:
        assert input_state.shape == (B, CONV_SIZE - 1, D)
        assert input_state.stride(-1) == 1
        assert input_state.device == x.device and input_state.dtype == x.dtype
        output_state = input_state.clone()
        state_arg = input_state
    else:
        output_state = x
        state_arg = x

    o = torch.empty((B, N, D), dtype = x.dtype, device = x.device)

    fwd_short_conv_silu[lambda meta: (
        triton.cdiv(D, meta['N_CDIM_SIZE'])
        * triton.cdiv(N, meta['N_SEQDIM_SIZE']), B)](
        x, weight, conv_weight, x, state_arg, o, output_state,
        B, N, C, D,
        x.stride(0), x.stride(1), weight.stride(0), conv_weight.stride(0), o.stride(0), o.stride(1),
        state_arg.stride(0), state_arg.stride(1),
        N_CONV_HALO_SIZE = N_CONV_HALO_SIZE, N_CONV_SIZE = CONV_SIZE, N_STATE_SIZE=CONV_SIZE - 1,
        IS_VARLEN=False, HAS_STATE=has_state)

    return (o, output_state) if input_state is not None else o

def torch_bwd_short_conv_silu(
    x, weight, conv_weight, do, cu_seqlens=None, max_seqlen=None,
    input_state=None, doutput_state=None,
):
    if cu_seqlens is not None:
        return torch_bwd_varlen_short_conv_silu(
            x, weight, conv_weight, do, cu_seqlens, max_seqlen,
            input_state, doutput_state)
    assert x.stride(-1) == 1 and weight.stride(-1) == 1 and conv_weight.stride(-1) == 1 and do.stride(-1) == 1

    B, N, C = x.shape
    D = weight.shape[0]
    CONV_SIZE = conv_weight.shape[0]

    N_SEQDIM_SIZE = 64
    N_CONV_HALO_SIZE = 16
    assert CONV_SIZE <= N_CONV_HALO_SIZE + 1
    has_state = input_state is not None and CONV_SIZE > 1
    if input_state is not None:
        assert input_state.shape == (B, CONV_SIZE - 1, D)
        assert input_state.stride(-1) == 1
        if doutput_state is None:
            doutput_state = torch.zeros_like(input_state)
        else:
            assert doutput_state.shape == input_state.shape
            assert doutput_state.stride(-1) == 1
            assert doutput_state.device == x.device and doutput_state.dtype == x.dtype
        dinput_state = torch.empty_like(input_state)
        state_arg = input_state
        doutput_state_arg = doutput_state
    else:
        assert doutput_state is None
        dinput_state = x
        state_arg = x
        doutput_state_arg = x

    dmma = torch.empty((B, N, D), dtype=x.dtype, device=x.device)
    dmma_halo = torch.empty((B, triton.cdiv(N,N_SEQDIM_SIZE) * N_CONV_HALO_SIZE, D), dtype=x.dtype, device=x.device)
    dconv_weight = torch.zeros_like(conv_weight, dtype = torch.float32)

    bwd_short_conv_silu[lambda meta: (
        triton.cdiv(D, meta['N_CDIM_SIZE'])
        * triton.cdiv(N, meta['N_SEQDIM_SIZE']), B)](
        x, weight, conv_weight, x, state_arg, do, doutput_state_arg,
        dmma, dmma_halo, dconv_weight, dinput_state,
        B, N, C, D,
        x.stride(0), x.stride(1), weight.stride(0), conv_weight.stride(0), do.stride(0), do.stride(1), dmma.stride(0), dmma.stride(1), dmma_halo.stride(0), dmma_halo.stride(1),
        state_arg.stride(0), state_arg.stride(1),
        N_CONV_HALO_SIZE = N_CONV_HALO_SIZE, N_CONV_SIZE = CONV_SIZE, N_STATE_SIZE=CONV_SIZE - 1,
        IS_VARLEN=False, HAS_STATE=has_state)

    dx = torch.empty_like(x)
    dweight = torch.zeros_like(weight, dtype=torch.float32)
    bwd_mma_gemm[lambda meta: (
        triton.cdiv(C, meta['BLOCK_IN'])
        * triton.cdiv(N, meta['BLOCK_SEQ']), B)](
        x, weight, dmma, dmma_halo, x, dx, dweight,
        B, N, C, D,
        x.stride(0), x.stride(1), weight.stride(0),
        dmma.stride(0), dmma.stride(1),
        dmma_halo.stride(0), dmma_halo.stride(1),
        dx.stride(0), dx.stride(1), dweight.stride(0),
        N_SEQDIM_SIZE=N_SEQDIM_SIZE, N_CONV_HALO_SIZE=N_CONV_HALO_SIZE,
        IS_VARLEN=False)

    grads = (dx, dweight, dconv_weight)
    return grads + (dinput_state,) if input_state is not None else grads


def _varlen_info(x, cu_seqlens, max_seqlen):
    assert x.shape[0] == 1
    assert cu_seqlens.is_cuda and cu_seqlens.device == x.device
    assert cu_seqlens.ndim == 1
    assert cu_seqlens.dtype in (torch.int32, torch.int64)
    assert cu_seqlens.is_contiguous() and cu_seqlens.numel() >= 2
    if max_seqlen is None:
        if torch.compiler.is_compiling():
            raise ValueError(
                "max_seqlen must be provided when compiling varlen input")
        max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())
    return cu_seqlens.numel() - 1, int(max_seqlen)


def torch_fwd_varlen_short_conv_silu(
    x, weight, conv_weight, cu_seqlens, max_seqlen=None,
    input_state=None,
):
    """Run packed causal convolution; max_seqlen avoids reading cu_seqlens on the host."""
    assert x.stride(-1) == 1 and weight.stride(-1) == 1 and conv_weight.stride(-1) == 1
    num_groups, max_seqlen = _varlen_info(x, cu_seqlens, max_seqlen)
    _, N, C = x.shape
    D = weight.shape[0]
    CONV_SIZE = conv_weight.shape[0]
    assert weight.shape[1] == C and conv_weight.shape[1] == D
    N_CONV_HALO_SIZE = 16
    assert CONV_SIZE <= N_CONV_HALO_SIZE + 1
    has_state = input_state is not None and CONV_SIZE > 1
    if input_state is not None:
        assert input_state.shape == (num_groups, CONV_SIZE - 1, D)
        assert input_state.stride(-1) == 1
        assert input_state.device == x.device and input_state.dtype == x.dtype
        output_state = input_state.clone()
        state_arg = input_state
    else:
        output_state = x
        state_arg = x

    o = torch.empty((1, N, D), dtype=x.dtype, device=x.device)
    if max_seqlen == 0:
        return (o, output_state) if input_state is not None else o
    fwd_short_conv_silu[lambda meta: (
        triton.cdiv(D, meta['N_CDIM_SIZE'])
        * triton.cdiv(max_seqlen, meta['N_SEQDIM_SIZE']), num_groups)](
        x, weight, conv_weight, cu_seqlens, state_arg, o, output_state,
        num_groups, max_seqlen, C, D,
        x.stride(0), x.stride(1), weight.stride(0), conv_weight.stride(0), o.stride(0), o.stride(1),
        state_arg.stride(0), state_arg.stride(1),
        N_CONV_HALO_SIZE=N_CONV_HALO_SIZE, N_CONV_SIZE=CONV_SIZE,
        N_STATE_SIZE=CONV_SIZE - 1, IS_VARLEN=True,
        HAS_STATE=has_state)
    return (o, output_state) if input_state is not None else o


def torch_bwd_varlen_short_conv_silu(
    x, weight, conv_weight, do, cu_seqlens, max_seqlen=None,
    input_state=None, doutput_state=None,
):
    """Backward for packed groups delimited by cu_seqlens."""
    assert x.stride(-1) == 1 and weight.stride(-1) == 1
    assert conv_weight.stride(-1) == 1 and do.stride(-1) == 1
    num_groups, max_seqlen = _varlen_info(x, cu_seqlens, max_seqlen)
    _, N, C = x.shape
    D = weight.shape[0]
    CONV_SIZE = conv_weight.shape[0]
    assert weight.shape[1] == C and conv_weight.shape[1] == D
    assert do.shape == (1, N, D)
    N_SEQDIM_SIZE = 64
    N_CONV_HALO_SIZE = 16
    assert CONV_SIZE <= N_CONV_HALO_SIZE + 1
    has_state = input_state is not None and CONV_SIZE > 1
    if input_state is not None:
        assert input_state.shape == (num_groups, CONV_SIZE - 1, D)
        assert input_state.stride(-1) == 1
        if doutput_state is None:
            doutput_state = torch.zeros_like(input_state)
        else:
            assert doutput_state.shape == input_state.shape
            assert doutput_state.stride(-1) == 1
            assert doutput_state.device == x.device and doutput_state.dtype == x.dtype
        dinput_state = torch.empty_like(input_state)
        state_arg = input_state
        doutput_state_arg = doutput_state
    else:
        assert doutput_state is None
        dinput_state = x
        state_arg = x
        doutput_state_arg = x

    if max_seqlen == 0:
        grads = (
            torch.empty_like(x),
            torch.zeros_like(weight, dtype=torch.float32),
            torch.zeros_like(conv_weight, dtype=torch.float32),
        )
        return grads + (doutput_state.clone(),) if input_state is not None else grads

    max_seqblocks = triton.cdiv(max_seqlen, N_SEQDIM_SIZE)
    dmma = torch.empty((1, N, D), dtype=x.dtype, device=x.device)
    dmma_halo = torch.empty(
        (num_groups, max_seqblocks * N_CONV_HALO_SIZE, D),
        dtype=x.dtype, device=x.device)
    dconv_weight = torch.zeros_like(conv_weight, dtype=torch.float32)

    bwd_short_conv_silu[lambda meta: (
        triton.cdiv(D, meta['N_CDIM_SIZE'])
        * triton.cdiv(max_seqlen, meta['N_SEQDIM_SIZE']), num_groups)](
        x, weight, conv_weight, cu_seqlens, state_arg, do,
        doutput_state_arg, dmma, dmma_halo, dconv_weight, dinput_state,
        num_groups, max_seqlen, C, D,
        x.stride(0), x.stride(1), weight.stride(0), conv_weight.stride(0),
        do.stride(0), do.stride(1), dmma.stride(0), dmma.stride(1),
        dmma_halo.stride(0), dmma_halo.stride(1),
        state_arg.stride(0), state_arg.stride(1),
        N_CONV_HALO_SIZE=N_CONV_HALO_SIZE,
        N_CONV_SIZE=CONV_SIZE, N_STATE_SIZE=CONV_SIZE - 1,
        IS_VARLEN=True, HAS_STATE=has_state)

    dx = torch.empty_like(x)
    dweight = torch.zeros_like(weight, dtype=torch.float32)
    bwd_mma_gemm[lambda meta: (
        triton.cdiv(C, meta['BLOCK_IN'])
        * triton.cdiv(max_seqlen, meta['BLOCK_SEQ']), num_groups)](
        x, weight, dmma, dmma_halo, cu_seqlens, dx, dweight,
        num_groups, max_seqlen, C, D,
        x.stride(0), x.stride(1), weight.stride(0),
        dmma.stride(0), dmma.stride(1),
        dmma_halo.stride(0), dmma_halo.stride(1),
        dx.stride(0), dx.stride(1), dweight.stride(0),
        N_SEQDIM_SIZE=N_SEQDIM_SIZE, N_CONV_HALO_SIZE=N_CONV_HALO_SIZE,
        IS_VARLEN=True)
    grads = (dx, dweight, dconv_weight)
    return grads + (dinput_state,) if input_state is not None else grads


class ShortConvSiluFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, x, weight, conv_weight, cu_seqlens=None, max_seqlen=None,
        input_state=None,
    ):
        has_varlen = cu_seqlens is not None
        has_state = input_state is not None
        saved_cu = cu_seqlens if has_varlen else torch.empty(
            0, dtype=torch.int32, device=x.device)
        saved_state = input_state if has_state else torch.empty(
            0, dtype=x.dtype, device=x.device)
        ctx.save_for_backward(x, weight, conv_weight, saved_cu, saved_state)
        ctx.has_varlen = has_varlen
        ctx.has_state = has_state
        ctx.max_seqlen = max_seqlen
        return torch_fwd_short_conv_silu(
            x, weight, conv_weight, cu_seqlens, max_seqlen, input_state)

    @staticmethod
    def backward(ctx, *grad_outputs):
        x, weight, conv_weight, saved_cu, saved_state = ctx.saved_tensors
        do = grad_outputs[0]
        if do is None:
            do = torch.zeros(
                (*x.shape[:-1], weight.shape[0]),
                dtype=x.dtype, device=x.device)

        cu_seqlens = saved_cu if ctx.has_varlen else None
        input_state = saved_state if ctx.has_state else None
        doutput_state = grad_outputs[1] if ctx.has_state else None
        grads = torch_bwd_short_conv_silu(
            x, weight, conv_weight, do, cu_seqlens, ctx.max_seqlen,
            input_state, doutput_state)
        dx, dweight, dconv_weight = grads[:3]
        dinput_state = grads[3] if ctx.has_state else None
        return (
            dx,
            dweight.to(weight.dtype),
            dconv_weight.to(conv_weight.dtype),
            None,
            None,
            dinput_state,
        )


def short_conv_silu(
    x, weight, conv_weight, cu_seqlens=None, max_seqlen=None,
    input_state=None,
):
    # Explicit casts outside custom_fwd avoid leaked autocast state when Dynamo
    # traces several consecutive applications with FP32 master parameters.
    if torch.is_autocast_enabled('cuda'):
        x, weight, conv_weight = (t.to(torch.bfloat16) for t in (x, weight, conv_weight))
        if input_state is not None:
            input_state = input_state.to(torch.bfloat16)
    with torch.autocast('cuda', enabled=False):
        return ShortConvSiluFunction.apply(
            x, weight, conv_weight, cu_seqlens, max_seqlen, input_state)


class CausalShortConv1d(torch.nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size,
        device=None, dtype=None,
    ):
        super().__init__()
        if kernel_size < 1 or kernel_size > 17:
            raise ValueError("kernel_size must be in [1, 17]")
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weight = torch.nn.Parameter(torch.empty(
            out_channels, in_channels, **factory_kwargs))
        self.conv_weight = torch.nn.Parameter(torch.empty(
            kernel_size, out_channels, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(self.kernel_size)
        torch.nn.init.uniform_(self.conv_weight, -bound, bound)

    def forward(
        self, x, cu_seqlens=None, max_seqlen=None, input_state=None,
    ):
        return short_conv_silu(
            x, self.weight, self.conv_weight,
            cu_seqlens, max_seqlen, input_state)

    def extra_repr(self):
        return (
            f"in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}"
        )

if __name__ == "__main__":
    torch.set_default_device('cuda:0')
    x = torch.randn((4, 256, 128), dtype=torch.bfloat16)
    weight = torch.randn((256,128), dtype=torch.bfloat16)
    cweight = torch.randn((4,256), dtype=torch.float32)

    o_ref = ref_short_conv(x, weight, cweight)
    o_kernel = torch_fwd_short_conv_silu(x, weight, cweight)
    print((o_kernel-o_ref).abs())
