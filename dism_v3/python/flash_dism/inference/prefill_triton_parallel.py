"""Three-phase forward executor for long prefill streams.

Factored form of `prefill_triton._execute`: instead of one CTA carrying the
`BR*BD` state through every chunk of a stream sequentially, split the work into

  1. `_chunk_summary`  - one CTA per selected chunk, parallel over chunks:
                         S_c = sk_c^T @ (v_c * key_scale_c)
  2. `_state_passing`  - one CTA per selected stream, serial over its chunks:
                         M_{c+1} = a_c * M_c + S_c, storing every incoming M_c
  3. `_execute_chunks` - one CTA per selected chunk, parallel over chunks:
                         history = (sq_c @ M_c) * scale, plus intra-chunk local

This is the chunkwise SSD / Simple-GLA decomposition of the stream recurrence
(the associative element is one scalar plus an R*DV matrix). Forward only; the
contract is numerical parity with `_execute`, not bitwise identity, because the
atomic output reduction and the MMA tiling differ.

Every kernel takes an explicit id list (`CHUNK_IDS` / `STREAM_IDS`) so the same
code serves the full plan and a long-stream subset. Summary/execute results are
addressed by the program's position in `CHUNK_IDS` (`SUMS`/`STATES` are compact,
size = number of selected chunks); `_state_passing` recovers that position from
`LOCAL_OFFSETS`. Pass `arange`+`offsets[:-1]` for the whole plan.
"""
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _chunk_summary(SK, V, ROWS, LOG_PREFIX, WEIGHTS, LENGTHS, SUMS, CHUNK_IDS,
                   R: tl.constexpr, DV: tl.constexpr, C: tl.constexpr,
                   BR: tl.constexpr, BD: tl.constexpr, BF16: tl.constexpr):
    p = tl.program_id(0)
    chunk = tl.load(CHUNK_IDS + p)
    t = tl.arange(0, C)
    r = tl.arange(0, BR)
    d = tl.arange(0, BD)
    length = tl.load(LENGTHS + chunk)
    row = tl.load(ROWS + chunk*C + t)
    prefix = tl.load(LOG_PREFIX + chunk*C + t)
    weight = tl.load(WEIGHTS + chunk*C + t)
    is_key = (t < length) & (row >= 0)
    ki = tl.where(is_key, row, 0)
    sk = tl.load(SK + ki[:, None]*R + r[None, :],
                 is_key[:, None] & (r[None, :] < R), 0).to(tl.float32)
    v = tl.load(V + ki[:, None]*DV + d[None, :],
                is_key[:, None] & (d[None, :] < DV), 0).to(tl.float32)
    last_prefix = tl.sum(tl.where(t == length-1, prefix, 0.), 0)
    key_scale = tl.where(is_key, weight*tl.exp(tl.minimum(last_prefix - prefix, 0.)), 0.)
    if BF16:
        summary = tl.dot(tl.trans(sk).to(tl.bfloat16), (v*key_scale[:, None]).to(tl.bfloat16))
    else:
        summary = tl.dot(tl.trans(sk), v*key_scale[:, None], input_precision="tf32x3")
    tl.store(SUMS + p*BR*BD + r[:, None]*BD + d[None, :], summary)


@tr.jit
def _state_passing(SUMS, OFFSETS, LOCAL_OFFSETS, STREAM_IDS, LENGTHS, LOG_PREFIX, RESETS, STATES,
                   BR: tl.constexpr, BD: tl.constexpr, C: tl.constexpr):
    k = tl.program_id(0)
    stream = tl.load(STREAM_IDS + k)
    local = tl.load(LOCAL_OFFSETS + k)
    begin = tl.load(OFFSETS + stream)
    end = tl.load(OFFSETS + stream + 1)
    r = tl.arange(0, BR)
    d = tl.arange(0, BD)
    m = tl.zeros((BR, BD), tl.float32)
    for chunk in range(begin, end):
        p = local + (chunk - begin)
        tl.store(STATES + p*BR*BD + r[:, None]*BD + d[None, :], m)
        length = tl.load(LENGTHS + chunk)
        last_prefix = tl.load(LOG_PREFIX + chunk*C + (length-1))
        reset = tl.load(RESETS + chunk)
        a = tl.where(reset != 0, 0., tl.exp(last_prefix))
        s = tl.load(SUMS + p*BR*BD + r[:, None]*BD + d[None, :])
        m = a*m + s


@tr.jit
def _execute_chunks(SQ, SK, V, OUT, STATES, ROWS, LOG_PREFIX, WEIGHTS, LENGTHS, RESET, CHUNK_IDS,
                    R: tl.constexpr, DV: tl.constexpr, C: tl.constexpr,
                    BR: tl.constexpr, BD: tl.constexpr, BF16: tl.constexpr):
    p = tl.program_id(0)
    chunk = tl.load(CHUNK_IDS + p)
    t = tl.arange(0, C)
    r = tl.arange(0, BR)
    d = tl.arange(0, BD)
    length = tl.load(LENGTHS + chunk)
    reset = tl.load(RESET + chunk)
    row = tl.load(ROWS + chunk*C + t)
    prefix = tl.load(LOG_PREFIX + chunk*C + t)
    weight = tl.load(WEIGHTS + chunk*C + t)
    is_key = (t < length) & (row >= 0)
    is_query = (t < length) & (row < 0)
    ki = tl.where(is_key, row, 0)
    qi = tl.where(is_query, -row-1, 0)
    sq = tl.load(SQ + qi[:, None]*R + r[None, :],
                 is_query[:, None] & (r[None, :] < R), 0).to(tl.float32)
    sk = tl.load(SK + ki[:, None]*R + r[None, :],
                 is_key[:, None] & (r[None, :] < R), 0).to(tl.float32)
    v = tl.load(V + ki[:, None]*DV + d[None, :],
                is_key[:, None] & (d[None, :] < DV), 0).to(tl.float32)
    state = tl.load(STATES + p*BR*BD + r[:, None]*BD + d[None, :]).to(tl.float32)

    history_scale = tl.where(reset != 0, 0., tl.exp(prefix))
    if BF16:
        history = tl.dot(sq.to(tl.bfloat16), state.to(tl.bfloat16))
    else:
        history = tl.dot(sq, state, input_precision="tf32x3")
    history *= (history_scale*weight)[:, None]

    valid_pair = is_query[:, None] & is_key[None, :] & (t[:, None] > t[None, :])
    log_ratio = tl.minimum(prefix[:, None]-prefix[None, :], 0.)
    pair = tl.where(valid_pair, tl.exp(log_ratio)*weight[:, None]*weight[None, :], 0.)
    if BF16:
        score = tl.dot(sq.to(tl.bfloat16), tl.trans(sk).to(tl.bfloat16))
        local = tl.dot((score*pair).to(tl.bfloat16), v.to(tl.bfloat16))
    else:
        score = tl.dot(sq, tl.trans(sk), input_precision="tf32x3")
        local = tl.dot(score*pair, v, input_precision="tf32x3")
    tl.atomic_add(OUT + qi[:, None]*DV + d[None, :], history+local,
                  is_query[:, None] & (d[None, :] < DV), sem="relaxed")


def run_parallel(core, sq, sk, value, chunk_ids, stream_ids, local_offsets, out=None):
    """Run summary -> passing -> output on the selected chunk/stream subsets.

    chunk_ids and stream_ids are int32 CUDA tensors; local_offsets[k] is the
    position of stream_ids[k]'s first chunk inside chunk_ids. Non-selected
    chunks must not write OUT.
    """
    R, DV = sq.shape[1], value.shape[1]
    BR = max(32, tr.next_power_of_2(R))
    BD = max(32, tr.next_power_of_2(DV))
    C = core.c
    n_chunks = int(chunk_ids.numel())
    if out is None:
        out = torch.zeros((core.n, DV), device=core.device, dtype=torch.float32)
    if not n_chunks:
        return out
    sums = torch.empty((n_chunks, BR, BD), device=core.device, dtype=torch.float32)
    states = torch.empty((n_chunks, BR, BD), device=core.device, dtype=torch.float32)
    bf16 = core.mma_precision == "bf16"
    args = dict(num_warps=8, num_stages=1)
    _chunk_summary[(n_chunks,)](sk, value, core.rows, core.prefixes, core.weights,
                                core.lengths, sums, chunk_ids, R, DV, C, BR, BD, bf16, **args)
    if stream_ids.numel():
        _state_passing[(int(stream_ids.numel()),)](sums, core.offsets, local_offsets, stream_ids,
                                                   core.lengths, core.prefixes, core.resets,
                                                   states, BR, BD, C, **args)
    _execute_chunks[(n_chunks,)](sq, sk, value, out, states, core.rows, core.prefixes,
                                 core.weights, core.lengths, core.resets, chunk_ids,
                                 R, DV, C, BR, BD, bf16, **args)
    return out
