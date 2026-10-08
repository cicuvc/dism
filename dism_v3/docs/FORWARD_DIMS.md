# Independent forward D/DV bring-up

The table below is the committed K64 baseline. Forward now also accepts
DISM_FORWARD_WARP_K_SIZE=32; see FORWARD_WARP_K.md for the separate matrix.

Defaults are now tanh, BF16 output and warp-private shared staging. TMA output
has been removed; nonzero DISM_OUTPUT_TMA is rejected. EX2 is parked, not part
of this validation. Historical benchmark reports remain unchanged.

DISM_KC_KEY_DIM=D and DISM_KC_HEAD_DIM=DV independently accept32/64/128.
Both default64. One extension contains one selected pair (not runtime dispatch).
R remains32, scan tile16x64,128 query rows/CTA,8 consumers+4 producer-group
warps, interleaved0,4,1,5,2,6,3,7 rows, FP32 scan/accumulation/normalization.

Score, readout similarity and probability tiles retain64 key columns.
Only V/output/accumulator channels and their rescale/store loops depend on DV.
The ordinary warp shared->global output checks tail rows. BF16 conversion
happens after FP32 normalization; lse2 remains FP32.

## Shared capacity

Each CTA keeps one128xD Q plus128x32 SQ prefetch region, aliased with the
combined K/SK/V ring. Output staging is eight16xDV BF16 tiles, independently
live. Communication, metadata, allocator alignment and barriers are included
in forward_shared_bytes(), not just the tensor payloads.

All combinations retain two input slots except D128/DV128: two slots would
require112KiB, above the device's99KiB opt-in CTA limit. This pair uses one
slot, requiring80KiB. No scan tile, R or register budget reduction is applied.

## Reproduce

From this directory:

```bash
PATH=/usr/local/cuda/bin:/home/cicuvc/miniconda3/envs/blkw/bin:$PATH \
DISM_KC_KEY_DIM=32 DISM_KC_HEAD_DIM=128 DISM_LINEINFO=1 \
/home/cicuvc/miniconda3/envs/blkw/bin/python build.py

PYTHONPATH=python:.. /home/cicuvc/miniconda3/envs/blkw/bin/python -m pytest -q tests

/home/cicuvc/miniconda3/envs/blkw/bin/python tools/check_forward_dims.py --warp-k-size 64
```

The sweep performs nine isolated builds/tests, saves ptxas resources and SASS,
runs selected memcheck/synccheck plus strict standalone output-probe racecheck,
and restores D64/DV64. Reports: build/forward_dims/results.json and per-pair
logs in that directory. Existing oracle tolerances are unchanged.
This sweep covers the default BF16 path, not every FP32 control's shared budget.

## Measured results (RTX5090/sm120a)

All nine pairs:478 passed,3 skipped each, with original tolerances. Skips are
EX2-only diagnostics, not failed precision cases. All forward SASS has no CALL.
Each pair also passed15 selected memcheck cases,15 synccheck cases, and10
standalone output-probe/codegen racecheck cases, with zero reported errors/
hazards. The latter does not certify the full input/summary pipelines race-free.

Forward ptxas resource bytes (static report, not dynamic traffic):

|D|DV|Shared KiB|Input slots|Stack B|Spill stores B|Spill loads B|
|---:|---:|---:|---:|---:|---:|---:|
|32|32|40|2|0|0|0|
|32|64|56|2|0|0|0|
|32|128|88|2|104|112|120|
|64|32|48|2|0|0|0|
|64|64|64|2|8|16|24|
|64|128|96|2|128|140|152|
|128|32|64|2|0|0|0|
|128|64|80|2|40|48|56|
|128|128|80|1|184|200|216|

All report168 registers with the existing consumer inc232/producer dec40
protocol. DV32 is spill-free for every D. DV64 has none/small/moderate spill;
DV128 has substantial static spill, largest at D128/DV128. No spill optimization
or performance attribution was attempted in this scope.

The unchanged summary kernel is spill-free for D32/D64. D128 summary reports
56B stack,60B spill stores,84B spill loads (independent of DV), no CALL.

Numerical cases cover N1/17/33/65/128/129/256/257/500/513 in both directions
and soft/mixed/hard, plus N769 repeated tasks with CTA counts1/2/default,
signed/zero readout, causal-value checks, fallback and all-match long chains
at tau0/.1/ln(D). No backward, varlen or simultaneous multi-D dispatch is claimed.
