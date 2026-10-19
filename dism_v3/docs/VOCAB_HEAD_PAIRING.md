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

## Final 72M result

The completed run reached:

- hard BPB: `1.183505`
- soft BPB: `1.176530`
- gap BPB: `0.006975`

For comparison, restored valres A reached `1.184223 / 0.008547`. Head pairing
therefore saves `1,179,648` parameters, improves hard BPB by about `0.00072`,
and narrows the gap, but the retrieval tradeoff is not uniform.

Offline NIAH used the same 16-trial paired protocol:

| context | swap-halves margin | valres A margin | delta |
|---:|---:|---:|---:|
| 2048 | **0.935** | 0.856 | +0.079 |
| 8192 | 0.511 | 0.612 | -0.101 |
| 16384 | 0.298 | 0.406 | -0.109 |

The 8k mean hides opposite depth trends:

| 8k depth | swap-halves margin | A margin | delta |
|---:|---:|---:|---:|
| 0.05 (far) | 0.152 | 0.574 | **-0.422** |
| 0.5 | 0.250 | 0.379 | -0.129 |
| 0.95 (near) | **1.133** | 0.883 | +0.250 |

At 16k the corresponding margins are `0.147/0.095/0.651` versus
`0.297/0.206/0.716`. Near-needle token accuracy is slightly higher than A
(`0.521` vs `0.469` at 16k), while sparse far-needle retrieval is worse.

Interpretation: cross-head role-paired codebooks improve common/near-pattern
matching and produce slightly better in-domain BPB, but reduce the diversity
needed for rare distal suffix matches. This is a materially different failure
mode from shareqk: it is a capability tradeoff rather than uniform role
collapse.
