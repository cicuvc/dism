# Compact head-paired Q/K vocabularies

`vocab_head_pairing="swap_halves"` halves the number of q/k codebook tables
without collapsing q and k roles inside a head.

For an even head count `H`, a DISM layer stores:

- `H/2` q tables: `Q_0 ... Q_(H/2-1)`
- `H/2` k tables: `K_0 ... K_(H/2-1)`

The per-head role mapping is:

| head | query codebook | key codebook |
|---:|---|---|
| `0 .. H/2-1` | `Q_i` | `K_i` |
| `H/2 .. H-1` | `K_(i-H/2)` | `Q_(i-H/2)` |

Thus every head still has distinct query and key tables; the second half only
reuses the pair in the opposite roles. This is intentionally different from
`vocab_share_qk`, where one table serves both roles within a head.

The compact tables are expanded with concatenation at the start of the DISM
selector computation. The existing interpolation kernels, hard labels, DISM
core, planner and output path are unchanged.

## Requirements

- even `n_head`
- `gqa_query_channels=1`
- `kv_group_size=1`
- no PQ, TransVQ, `vocab_share_qk`, `vocab_share_heads`, or
  `vocab_share_layers`

## Cost

For the 72M model, each DISM layer changes from `6+6` dense tables to `3+3`,
saving `1,179,648` parameters in total:

- valres baseline: `72,035,808`
- head-paired variant: `70,856,160`

## Validation

- Expanding the compact tables and loading them into an otherwise identical
  full-table module produces bitwise-equal outputs.
- Compiled forward/backward, table gradients and the q/k utilization probe pass.
- A compiled 72M `16x2048` smoke step peaks at `15.4 GiB` and reaches about
  `206k tokens/s` after warmup.
