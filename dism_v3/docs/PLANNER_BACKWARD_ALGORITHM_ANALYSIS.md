# Planner event hard backward: algebra, traffic and algorithm experiments

Baseline **8d47ebf326abe736a1b17bf41eae1cf9e3e13141**, RTX 5090 / sm120,
2026-10-20, conda `blkw`, PyTorch 2.13.0+cu130 / Triton 3.7.1.
Retain the BF16-only one-pair specialization: the final R32/DV64 compact C64
implementation with W4 speeds up B2/H6/N512 repeat by 1.115–1.116x and N2048
repeat by 1.094–1.096x. With the unchanged consumer warp defaults, the respective
gains are 1.094–1.095x and 1.071–1.082x, with gains on random labels too.
Reject the tested tau-reuse,
parallel-scan and persistent alternatives. The tau identity is valid over real
event gradients, but does not justify reusing the currently rounded local
gradients. The expanded R16/DV64 near-zero tau failure shared by baseline and
candidate, and unavailable hardware traffic counters, are documented below.

This is a new comparison against the three-component compensated implementation,
not against the older precision repair used in PLANNER_BACKWARD_PERFORMANCE.md.
Only this worktree's Python modules and independent local `cu_flash_dism` binary
are used. No production planner default, training process, checkpoint, optimizer,
RNG counter, annealing buffer, dataloader position, hard-row sampling or direction
setting is changed. PID 2058134 stays stopped (`Tsl`).

## Event computation graph and exact identities

Use event indices, not token indices. In one chunk let Q=SQ, K=SK, U=dO,
V=value, with zero rows for the other event type and padding. The causal event
mask T has T_ij=1 exactly when i is a query, j is a key, and i>j.
The native planner has nonincreasing local log-prefix p because every pivot
rescaling decay is in [0,1]. Define

```
P_ij = T_ij exp(min(p_i-p_j,0)) w_i w_j
B = U V^T                         C x C
S = Q K^T                         C x C
A = B * P                         C x C       (* is elementwise)
dV_local = (S * P)^T U             C x DV
dQ_local = A K                    C x R
dK_local = A^T Q                  C x R
E = A * S                         C x C
tau_local = sum_ij E_ij (dw_i+dw_j)
```

The graph shares **A between dQ and dK**, and **S between dV and tau**.
Neither B nor S is itself redundant: they contract different vector spaces.
`pair`, `ds` and `score` are register/layout values; there is no explicit global
C-by-C allocation to remove. Register pressure can nevertheless spill them.

The derivative metadata includes the physical weight derivative, not a missing
prefix term. Write a raw query log-weight as a_i, a raw key log-weight as b_j,
and the running numerical pivot as pi. Native key/query weights are
`exp(b_j-pi_j)` and `exp(a_i+pi_i-logZ_i)`. Intervening decay factors telescope
to `exp(pi_j-pi_i)`, so their product is `exp(a_i+b_j-logZ_i)` and
`dlogP = a_i' + b_j' - dlogZ_i = dw_i+dw_j`. Pivot derivatives cancel. Resets
remove earlier pairs altogether; they do not add a derivative to surviving
pairs. Stored FP32 coefficients approximate this FP64 construction, which is
one remaining source of numerical error even with accurate vector contractions.

For exact real arithmetic,

```
<Q_i, dQ_i> = sum_j A_ij <Q_i,K_j> = sum_j E_ij
<K_j, dK_j> = sum_i A_ij <Q_i,K_j> = sum_i E_ij
=> tau_local = sum_i dw_i <Q_i,dQ_i>
             + sum_j dw_j <K_j,dK_j>.
```

This proof applies to **any fixed P**, even one whose prefixes do not satisfy
monotonicity: it does not factor the exponential or change the mask. Native
`chunk_program` cuts immediately before a reset key, so no reset occurs inside
a chunk. The first reset key is retained in local P; incoming history is killed
separately. Padding has P=0. Repeated token indices are separate events and
must retain multiplicity and their own dw. The proof covers all of these cases.
It does **not** justify assigning a single dw to an already-scattered token.

### Why using the current vector results is numerically wrong

In BF16 mode the installed code computes

```
dQ_hat = round_bf16(A) K; dK_hat = round_bf16(A)^T Q
```

with FP32 MMA accumulation. Tau instead consumes the **unrounded FP32 A**.
The identity using dQ_hat/dK_hat therefore differentiates a different matrix.
Even if both gradients used the identical rounded A and their inner products
were exact, the error would be

```
sum_ij (round_bf16(A_ij)-A_ij) S_ij (dw_i+dw_j).
```

In practice the two inner products and reductions also round independently;
the error bound contains `sum |dw_i| |Q_i dot delta_dQ_i|` and its key analogue.
Large opposite event derivatives amplify these independent errors. There is no
uniform relative bound when the final derivative is near zero.

A concrete BF16-source counterexample needs only one pair: P=1, Q=K=(1,0,...),
U=(1,1/16,0,...), V=(1,1/32,0,...), dw_query=0, dw_key=1. All inputs are exactly
BF16. A=1+1/512 is exact FP32, S=1, so baseline tau is 1+1/512. BF16(A)=1,
and reusing the local vector gradient returns tau=1. This failure needs no
reset, padding, atomics, or long stream. The one-pair specialization retains
unrounded A for tau and therefore avoids this counterexample.

`tau_vectors` tests direct reuse. `tau_compensated` replaces both local gradient
MMAs with three-component contractions: `dQ=(K^T A^T)^T`,
`dK=(Q^T A)^T`. The left operands are exact BF16, so the existing helper applies.
This restores adequate tested precision, but changes five local dot calls into
nine, retains A longer, and is slower. It does not eliminate S, which dV needs.

A cheaper identity is

```
tau_local = dot(dw, row_sum(E)+column_sum(E)).
```

`tau_rowcol` tests this. It removes C-by-C derivative multiplication but splits
the cancellation across reductions. It passes the sampled cancellation gate,
without providing a throughput win. The pairwise baseline deliberately combines
`dw_i+dw_j` before the reduction and remains the safer order.

### Can pre-scaling remove S*P or merge local and carried readouts?

Within a reset-free chunk, formally P=diag(alpha) T diag(beta), where
alpha_i=w_i exp(p_i), beta_j=w_j exp(-p_j). Thus

```
dV_j = beta_j sum_{i>j} <K_j,Q_i> alpha_i U_i.
```

A single `K (Q^T (alpha U))` is incorrect: it loses the strict event mask.
For interleaved k0,q0,k1,q1 the key/query mask is `[[1,0],[1,1]]`, determinant
one. No two diagonal scalings of an all-ones mask reproduce it. Pre-scaling
U/V only changes coefficients, not this rank/causality obstruction. The same
obstruction applies to dQ/dK. Cross-chunk rectangles are complete and do admit
state factorization; that is exactly what H/G already implement.

A prefix/suffix outer-product recurrence avoids the local C-by-C matrix, but
requires a distinct R-by-DV state at each readout, or serial updates. A tiled
divide-and-conquer algorithm can contract complete off-diagonal rectangles and
leave smaller diagonal tiles; its extra states and approximately O(C R DV log C)
work must beat the current O(C^2(2DV+3R)) masked tensor work and its small tiles.
It is a legitimate future algorithm, not an equality allowing removal of T.

Moving weights into BF16 U, V, Q or K introduces new rounding **before** the
inner products. In particular `round_bf16(S*P)` is not reproduced by rounding
scaled Q/K and then multiplying. The carried compensation does not repair that
new local rounding. This is why no unvalidated pre-scaled production variant is
introduced.

Concatenation formulas exist, e.g.

```
dQ = [A, diag(qs) U] [K; H_in^T]
dV = [(S*P)^T, diag(ks) K] [U; G_out].
```

They do not reduce mathematical multiply-adds. The former also mixes the
intentional BF16-rounded local A with the compensated carried contraction;
putting qs inside the product changes the cancellation-sensitive order.
A shared output accumulator may reduce atomics, but extends live ranges.
The preceding performance report already measured combined atomics: a C64 gain
and a C32 regression. No inference that fewer atomic sites means faster execution
is made here.

## Summary/boundary graph and sufficient statistics

Let l_c be the final local prefix, z_c indicate reset, and M=R*DV. The code uses

```
ks_j = w_j exp(min(l_c-p_j,0))
qs_i = w_i exp(p_i)
a_c = (1-z_c) exp(l_c)
h_c = sum_key ks_j K_j V_j^T
g_c = (1-z_c) sum_query qs_i Q_i U_i^T
H_out = a_c H_in + h_c
G_in  = a_c G_out + g_c
```

The final query readout also zeroes qs at a reset. The forward summary h_c is
retained at a reset, while both a_c and g_c are zero. This distinction prevents
reverse history leaking into the preceding chunk.

H/G are sufficient statistics already. For arbitrary vector inputs, a sum of
R-by-DV outer products can have full rank. Arbitrary later U probes every column
of H and arbitrary K/V probes G; neither can be replaced by a scalar or one
vector without additional assumptions. A low-rank factorization is only smaller
when there are very few events and then requires variable-rank storage and
readout work. The single-pair specialization below exploits the extreme case
without trying to compress general boundaries.

Independent summary CTAs cannot directly emit final boundary readouts: they
lack preceding/succeeding summaries. Combining summary and scan requires either
one persistent CTA per stream, a supported cooperative global synchronization,
or a prefix propagation protocol with extra synchronization/storage. No unsafe
cross-CTA spin waiting or shared FP32 atomic is proposed.

The summary kernel **already computes both H and G in one launch**. A batched
MMA can express these two independent contractions but does not halve their
products. A dense block-diagonal embedding doubles or quadruples work on zero
blocks. H uses exact BF16 V^T against weighted K; G uses exact BF16 Q^T against
weighted U. They share neither exact left operand nor equal output orientation.

Likewise hq/tq **and hk/tk are already reused** in the compensated path.
`hk = V G^T`, `hv = K G` have different left operands and opposing G layout.
The BF16 high part of G is common, but the contractions are not. Packing them
into a bigger dense MMA adds unused cross blocks; a true batched operation keeps
the same useful MMA count. Local dK uses Q and A, dV uses U and S*P, so there is
no further exact BF16 left-operand CSE there. Layout conversion experiments
could still matter, but this analysis finds no removable mathematical product.

### Repeated coordinate work and its boundary correction

Summary and final kernels repeat row gathering, masks and ks/qs exponentials.
They are already in bounded chunk-local coordinates. With the formal alpha/beta
above, `ks=exp(l_c)*beta`; local P would need
`alpha_i*ks_j/exp(l_c)` plus T. Dividing by a_c is wrong at a reset, and even
`exp(l_c)` can underflow. beta can overflow while every physical P remains
bounded. Eliminating bounded coordinates therefore sacrifices stability.

Writing precomputed qs/ks costs 8C bytes written and 8C read per multi chunk,
plus another arena; it saves O(C) scalars against O(C R DV) state work, while
vectors still need gathering. Writing weighted vectors costs O(C(R+DV)) FP32
traffic and can lose the exact BF16 operand needed by compensation. This is a
possible metadata-fusion experiment only after bandwidth profiling; it is not a
free cross-kernel CSE. Native `finish()` already caches key decay/weight between
its scalar and coefficient passes, so the CPU exponential duplication described
in its comment is already eliminated.

## Parallel scans, persistent recurrence and closed forms

Affine composition `(a,b) o (c,d)=(ca,cb+d)` in forward evaluation order is
associative over real numbers. Identity is (1,0); a reset has a=0 and remains a
valid segmented-scan element. A tree scan changes FP32 association, requiring
numerical validation rather than bitwise claims.

`scan32` processes 32 affine steps at once with `tl.associative_scan`, 128 matrix
elements per CTA, exclusive forward/reverse boundaries, and a serial loop over
32-step slabs for longer streams. Every CTA owns disjoint matrix elements.
Both directional loads precede their in-place stores; no CTA reads another's
outputs. Invalid padded elements have (1,0), including at the last slab.
It uses the same global H/G arena and traffic as the serial scan, with replicated
alpha loads, more CTAs, padding, and shared layout exchange. Depth is
O(ceil(L/32) log 32), not an arbitrary-length log-depth claim.

At C64, N512 repeat has 13 multi streams / 42 chunks / maximum 8 chunks;
B2/H6/N2048 has 708 / 3360 / maximum 32. Only 12 of the 708 streams have 32
chunks. The length histogram is
`2:372, 3:12, 4:168, 5:12, 8:72, 9:12, 16:36, 17:12, 32:12`.
The average is 4.75; padding all of them to 32 is costly. The measured ~26 us
serial scan is a small part of the full call and is not the only long-stream
bottleneck.

A multi-kernel Blelloch scan can work in O(LM) operations with O(log L) depth,
but adds temporary/up-sweep/down-sweep traffic and launches. Hillis-Steele does
O(LM log L) work. A hierarchical two/three-pass algorithm scans local groups,
scans group totals, then adds prefix/suffix corrections. Without fusing that
correction into readout it adds at least a boundary read and write (16LM bytes
for two FP32 states), plus group totals. With fusion it still needs group states
and extra correction arithmetic. For the present maximum 32 and mean 4.75 it
has no demonstrated traffic advantage over the in-place scan. Length buckets
could reduce tree padding but add host/device arenas and launches; this should
be revisited only when long streams dominate, not to rescue this benchmark.

`persistent_tf32` measures the existing per-stream register recurrence with
accurate TF32x3 arithmetic for multi streams, preserving the same single buckets
and compact multi event arena. It avoids all global H/G snapshots, but repeats
local contractions in the query/key passes, serializes chunk MMA work, and
spills heavily. It also fails one existing cancellation case: even TF32x3 is not
a license to change the tau reduction order. Its timings are diagnostic rejected
results, not a replacement for the repaired path. No crossover in its favor
appears at 8 or 32 maximum chunks; for random labels it is never launched.
A new compensated persistent design remains unmeasured and cannot inherit this
implementation's numerical or performance claims.

The compact consumer already skips all summary/scan work when multi_chunks=0.
Both random fixtures in this report take that path. There is no extra summary
launch to remove for those plans.

### Global tau, normalization and inverse/geometric proposals

For a physical causal pair with matched length m, set
`F_m=sum_{l=1..m} exp(l*tau)`, `D_m=d log(F_m)/d tau`,
`Z_i=1+sum_j F_mij`, `DZi=sum_j (F_mij/Z_i) D_mij`.
With S_ij=<SQ_i,SK_j>, B_ij=<dO_i,V_j>, the exact derivative is

```
dtau = sum_ij (F_mij/Z_i) S_ij B_ij (D_mij-DZi).
```

The fallback contributes one to Z, zero output, and zero derivative. No extra
fallback numerator event exists. Direction and hard selection are fixed;
this formula does not differentiate discrete routing or change packed-row
sampling into per-document sampling.

At **event** resolution the local identity extends to all carried pairs:
`sum_e dw_e <vector_e, gradient_e>` with accurate event gradients. At **token**
resolution it generally does not. A counterexample is two event contributions
u and -u to the same token with derivatives 1 and 2: the scattered gradient is
zero but the derivative-weighted contraction is `-<vector,u>`. Event derivatives
are stream/match dependent, even for repeated token indices.

`sum_i <dO_i,O_i>` equals `sum_j <V_j,dV_j>` for the linear value map. This gives
total energy, not tau: it lacks D_mij. The denominator term can be written
`-sum_i DZi <dO_i,O_i>`, but the numerator still needs a D_m-weighted contraction.
One could carry derivative moment states along with H/G. That doubles relevant
matrix statistics and adds readouts; it does not follow from ordinary dV or
output gradients. Using a BF16 saved O also loses the precision that tau needs.

The CPU planner already uses stable log-domain geometric sums:

```
log F_m = (m*tau if tau>0 else tau)
        + log(-expm1(-m*abs(tau))) - log(-expm1(-abs(tau)))
log F_m = log(m) at tau=0
D_m = exp(logF_(m-1)-logF_m) D_(m-1) + exp(m*tau-logF_m) m.
```

The denominator logaddexp combines independent streams plus fallback. It cannot
be inferred from SQ/SK, which affect only the numerator. CPU planning is outside
GPU timings. Removing a denominator/prefix computation by differentiating pivot
coordinates would double-count the gauge derivative already cancelled in pair
weights. No redundant GPU logsumexp is present in these backward kernels.

A triangular recurrence can be written `(I-L_a)^-1 b`, and its nilpotent inverse
is a finite geometric series. Computing a dense inverse costs O(C^3), while
applying it costs O(C^2 M); the simple recurrence costs O(CM). Matrix inversion
is not a cheaper scan. For a constant ratio r, `(1-r^m)/(1-r)` loses digits near
r=1, has a removable singularity at tau=0, and overflows for positive tau if
formed directly. For negative tau it can underflow long tails; match lengths
and resets vary, so one constant-r inverse does not describe all event weights.
The current expm1/log-domain scalar formula handles these cases. Gauge decays
remain in [0,1] even for negative tau, avoiding exponential state growth, but
near-one decays can accumulate summation error. Neither a dense inverse nor an
unchecked geometric approximation is adopted.

There is also a scalar closed form for D_m: for positive tau,
`m + m/expm1(m*tau) - 1/expm1(tau)`; for negative tau, with t=-tau,
`1 - m/expm1(m*t) + 1/expm1(t)`. Subtracting the two O(1/|tau|) terms is
ill-conditioned near zero. The series starts `(m+1)/2 + (m²-1)*tau/12`, with
validity governed by m*|tau|, not tau alone. A carefully bounded series/expm1
implementation might accelerate CPU derivative-table construction, but this is
outside GPU backward timings and is not validated or substituted here.

## One-pair specialization: useful work instead of a padded matrix

The native compact width16 bucket contains many streams with exactly two events,
a key followed by a query. For these streams P, A and S have exactly one useful
entry. Compute the two scalar inner products, apply the **same** BF16 rounding
sites for vector coefficients, and write the same gradients/tau:

```
a = <U,V> P; s = <Q,K>
dQ = bf16(a) K; dK = bf16(a) Q; dV = bf16(s P) U
dtau = a s (dw_key+dw_query).
```

All products use promoted BF16 vectors and FP32 accumulation. SIMD reductions
associate sums differently from MMA, so this is algebraic equivalence, not a
bitwise claim. The cancellation-sensitive derivatives are still added before
multiplication; the vector coefficient rounding is not reused for tau.

The installed candidate uses this path only when every source vector is BF16,
BF16 mode is selected, there is no carried boundary, length is exactly two,
and the event signs are key then query. The branch is uniform within a CTA.
General FP32, mixed storage, malformed two-event order, key-only streams, and
multi chunks use the original implementation. Reset at the first key needs no
special case because incoming state is zero; padding never enters the scalar
loads. Separate events/streams retain separate tau partials and vector atomics.
No host plan bucketing, launch defaults, chunk sizes, tolerances or atomic count
changes. No R64 production support is added.

The specialization removes most of the width16 matrix work for those CTAs,
but keeps their launches, vector gathers and atomic contention. This explains
both the gains at large grids and the lack of measurable improvement for tiny
B1/H1/N512 calls. It is not an atomic-reduction optimization.

## Operation and byte accounting

Counts below distinguish (a) real-arithmetic useful formulas, (b) dense padded
MMA execution, (c) compiler resources/static instruction sites, and (d) actual
memory-system traffic. FMA counts as two FLOPs. Let C be the physical event tile,
r=max(32,R), d=DV for BF16; replace R/DV with r/d for padded tensor work.
All input bytes below assume BF16 and output/state bytes assume FP32.

| Node, per chunk | `tl.dot` products in compensated BF16 | Dense tensor FLOPs |
| --- | ---: | ---: |
| B=U V^T | 1 | 2 C² d |
| S=Q K^T | 1 | 2 C² r |
| dV=(S*P)^T U | 1 | 2 C² d |
| dQ=A K and dK=A^T Q | 2 | 4 C² r |
| Both accurate summaries H/G | 6 | 12 C r d |
| Accurate hq/tq and hk/tk | 6 | 12 C r d |
| hv=K G | 1 | 2 C r d |
| Single total | 5 | 2 C² (2d+3r) |
| Multi summary + readout total | 18 | 2 C² (2d+3r) + 26 C r d |
| `tau_compensated` extra local work | +4 | +8 C² r |

`tau_vectors` and `tau_rowcol` keep baseline tensor counts; their changes are
scalar reductions and live ranges. `scan32` also keeps those tensor counts.
The two-pass persistent reference has six local dots (U V^T is recomputed),
three boundary dots and two summary dots: **11 logical dots**, mathematical
work `6C²(r+d)+10Crd` per chunk, all through TF32x3 rather than BF16. Each
TF32x3 contraction represents three partial TF32 products, with 1688 hardware
tiles, versus 16816 BF16 tiles. Padding, warp replication and loop control can
add executed instructions; the table of actual static SASS sites below is
reported separately rather than equating one `tl.dot` with one MMA instruction.

For R32/DV64, C16/32/64 single tiles execute respectively
114,688 / 458,752 / 1,835,008 tensor FLOPs, corresponding to
28 / 112 / 448 **warp-level HMMA.16816 equivalents** (4096 FLOPs each).
C64 summaries use 384 equivalents; carried readouts use 448. Thus a complete
C64 multi chunk uses 1280. This is the sum across warps, not the number of static
SASS lines. A C16 one-pair CTA instead uses about `4R+3DV-2` vector/dot FLOPs
(318 at R32/DV64), plus scalar weights/tau; no tensor MMA executes on its branch.

Scalar work is additional: baseline local P needs C² exponentials, C² prefix
subtractions and 2C² weight multiplies (plus min/mask operations); A and S*P
need 2C² multiplies. Pair tau requires C² derivative additions, 2C² multiplies
and C²-1 reduction adds. This gives **9C²-1 ordinary scalar FLOPs plus C²
exponentials**, before compiler masking/CSE/FMA contraction. Pair-row/column tau
uses C² energy multiplies, 2C(C-1) reductions, C row/column adds, C multiplies
and C-1 final adds: 3C²+C-1 rather than 4C²-1. Vector-identity tau costs
approximately 4Cr+2C-2 scalar FLOPs but adds the numerical/liveness costs above.
Each compensated operand split needs two FP32 subtractions per element and
BF16 conversions, three tensor products and one low-product output addition.
Summaries also scale C(r+d) elements; carried readouts scale 2Cr+Cd elements
and form two event energy reductions. The serial two-state scan performs 4TM
FLOPs over T chunks, plus address/loop work; it has no MMA.

Let k/q be valid key/query event counts in a chunk. The vector atomic count is
`R(k+q)+DV*k` for a single chunk and **twice this** for a multi chunk, including
boundary-zero writes. Multi chunks additionally perform one tau atomic; single
chunks store one tau scalar. Final tau scatter adds one contribution per single
block/multi stream. These are element operations: sm120 can emit vectorized
F32x4 REDG instructions. A packed atomic instruction is not one scalar update.

Logical compulsory traffic (not cache-line/DRAM traffic):

| Item | Bytes |
| --- | ---: |
| One vector gather pass | 2(R+DV)(k+q) |
| Multi gather passes, summary + final | 4(R+DV)(k+q) |
| Summary H/G write | 8TM |
| In-place scan H/G read + write | 16TM |
| Final boundary read | 8TM |
| H/G total / allocated snapshot | **32TM / 8TM** |
| Single padded row/prefix/weight/dlog reads | 16C + length/ID metadata |
| Additional multi summary row/prefix/weight reads | 12C + length/reset/ID metadata |
| Vector/tau atomic read-modify-write payload model | 8 bytes per FP32 atomic element |
| Output clearing | 4 BNH (2R+DV), plus partial/tau clearing |
| Partials clearing / single tau stores | 8S / 4S_single |
| Tau partial add, conversion/scatter | O(S), unchanged across non-persistent variants |

Alpha has 4T write bytes and minimally 8T forward/reverse read bytes; the
parallel scan replicates those reads across 16 matrix tiles. Metadata index
loads, cache sectors, redundant lane loads and atomic serialization mean the
logical lower bounds are **not** measured bandwidth. One-pair adds a uniform
length/sign guard; successful CTAs load two slots instead of the padded arrays.
It retains vector and atomic payloads. Persistent recurrence removes 32TM
snapshot traffic but adds spills and duplicated pair work. `scan32` keeps 32TM,
so it cannot claim a global traffic saving.

Actual C64 compact topology and R32/DV64 workload totals:

| B×H×N:labels | single16/32/64 CTAs | multi chunks / streams / max | one-pair CTAs | padded tensor GFLOPs | HMMA equivalents | vector+multi-tau atomic elements | input gather MiB | H/G traffic MiB |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1×1×512:repeat | 1212/31/15 | 42/13/8 | 893 | .400949 | 97,888 | 687,242 | 2.008 | 2.625 |
| 1×1×512:random | 195/0/0 | 0/0/0 | 129 | .022364 | 5,460 | 28,416 | .086 | 0 |
| 2×6×2048:repeat | 58320/1524/756 | 3360/708/32 | 42,972 | 26.391085 | 6,443,136 | 44,986,272 | 130.893 | 210 |
| 2×6×2048:random | 12711/0/0 | 0/0/0 | 2,493 | 1.457799 | 355,908 | 2,961,568 | 10.018 | 0 |

For the four rows, one-pair replaces approximately .102 / .0148 / 4.915 /
.285 GFLOPs of padded tensor work with scalar/vector work, without changing
atomic elements. The B2/H6/N2048 repeat state allocation is 52.5 MiB; its logical
atomic payload model is 343.22 MiB. The computation is therefore not explained
by FLOPs alone. All values count event multiplicity, not unique token loads.

Hardware-counter collection was attempted with `/usr/local/cuda/bin/ncu
--query-metrics ...` and rejected with **ERR_NVGPUCTRPERM**. No driver permission,
clock, power or profiling setting was changed. Consequently measured DRAM/L2
bytes and dynamic shared/spill transactions are unavailable. The static SASS
and compiler audit below reports shared **capacity**, LDS/LDSM/STS instruction
sites and spills explicitly; it does not invent shared traffic from capacity.

## Paired measurements: all algorithm variants

Every timing is GPU microseconds, **median/p95**, 31 warmed CUDA-graph samples,
8 calls/replay, W4/C64, compact plans, R32/DV64. Forward and reverse mean
**variant measurement order**, not a change to model direction. One native plan
and one input tuple are reused by all variants within a case. Seed 120 labels,
seed 318 BF16 vectors scaled by .2, tau=.5, independent packed training rows.
CPU planning, upload and JIT are outside GPU times. Full times include output
clearing and tau reduction; independently timed stages are not additive.

| Case | Variant | Forward median/p95 | Reverse median/p95 | Speedup vs 8d47ebf, F/R |
| --- | --- | ---: | ---: | ---: |
| 1x1x512:repeat | baseline | 32.90/33.15 | 32.91/33.16 | 1.000/1.000 |
| 1x1x512:repeat | tau_vectors | 33.16/33.41 | 33.18/33.39 | 0.992/0.992 |
| 1x1x512:repeat | tau_compensated | 41.40/41.61 | 41.50/41.60 | 0.795/0.793 |
| 1x1x512:repeat | tau_rowcol | 33.15/33.29 | 33.16/33.32 | 0.993/0.993 |
| 1x1x512:repeat | scan32 | 36.48/36.57 | 36.48/36.56 | 0.902/0.902 |
| 1x1x512:repeat | persistent_tf32 | 333.68/334.02 | 333.90/334.20 | 0.099/0.099 |
| 1x1x512:repeat | single_pair | 32.98/33.18 | 33.07/33.16 | 0.998/0.995 |
| 1x1x512:random | baseline | 7.28/7.38 | 7.28/7.40 | 1.000/1.000 |
| 1x1x512:random | tau_vectors | 7.74/7.80 | 7.72/7.83 | 0.941/0.943 |
| 1x1x512:random | tau_compensated | 8.09/8.27 | 8.08/8.26 | 0.901/0.901 |
| 1x1x512:random | tau_rowcol | 7.28/7.41 | 7.28/7.44 | 1.000/1.000 |
| 1x1x512:random | scan32 | 7.29/7.40 | 7.28/7.39 | 0.999/0.999 |
| 1x1x512:random | persistent_tf32 | 7.30/7.36 | 7.27/7.33 | 0.998/1.001 |
| 1x1x512:random | single_pair | 7.30/7.41 | 7.28/7.38 | 0.998/0.999 |
| 2x6x2048:repeat | baseline | 366.67/367.07 | 368.69/369.39 | 1.000/1.000 |
| 2x6x2048:repeat | tau_vectors | 387.19/387.55 | 387.40/388.05 | 0.947/0.952 |
| 2x6x2048:repeat | tau_compensated | 540.83/541.19 | 541.18/542.57 | 0.678/0.681 |
| 2x6x2048:repeat | tau_rowcol | 376.65/377.29 | 377.04/377.39 | 0.974/0.978 |
| 2x6x2048:repeat | scan32 | 539.05/539.62 | 538.79/539.12 | 0.680/0.684 |
| 2x6x2048:repeat | persistent_tf32 | 2250.00/2258.54 | 2249.35/2259.58 | 0.163/0.164 |
| 2x6x2048:repeat | single_pair | 336.42/336.83 | 335.00/335.28 | 1.090/1.101 |
| 2x6x2048:random | baseline | 43.89/43.97 | 44.14/44.20 | 1.000/1.000 |
| 2x6x2048:random | tau_vectors | 49.25/49.35 | 49.26/49.36 | 0.891/0.896 |
| 2x6x2048:random | tau_compensated | 65.88/65.98 | 65.87/65.94 | 0.666/0.670 |
| 2x6x2048:random | tau_rowcol | 45.68/45.81 | 45.68/45.79 | 0.961/0.966 |
| 2x6x2048:random | scan32 | 44.06/44.24 | 44.05/44.16 | 0.996/1.002 |
| 2x6x2048:random | persistent_tf32 | 44.12/44.22 | 44.03/44.18 | 0.995/1.003 |
| 2x6x2048:random | single_pair | 42.50/42.70 | 42.59/42.65 | 1.033/1.037 |

Numerically rejected variants remain in the table for diagnosis: speed cannot
rescue a failed tau gate. `tau_vectors` fails the N512 FP64 gate in repeat and
random inputs and 5/8 synthetic reset/cancellation tests. `persistent_tf32`
fails 1/8 synthetic tests (events=33). `tau_compensated`, `tau_rowcol`, `scan32`
each pass 8/8 synthetic tests. The initial unrestricted one-pair diagnostic
passed the original **142/142** focused tests before production was restricted
to exact BF16 inputs. No rejected algorithm is installed.

| Variant | N512 repeat tau relative L2 vs independent FP64 | B2/H6/N2048 repeat tau relative L2 | Decision |
| --- | ---: | ---: | --- |
| baseline | 1.243e-05 | 7.767e-06 | reference |
| tau_vectors | 2.176e-03 | 1.444e-03 | reject precision and speed |
| tau_compensated | 5.866e-06 | 4.551e-05 | reject extra MMA/spills |
| tau_rowcol | 2.795e-05 | 2.143e-05 | reject throughput; cancellation order less robust |
| scan32 | 1.333e-05 | 6.638e-06 | reject padding/CTA/shared overhead |
| persistent_tf32 | 1.199e-05 | 5.579e-06 | reject precision, spills and serialization |
| single_pair | 1.389e-05 | 7.031e-06 | retain subject to validation below |

### Supported dimensions, B2/H6 at both N512 and N2048

Same method, C64/W4 compact, two measurement orders, all four supported R/DV
combinations. This is a dimension check, not a chunk/warp tuning sweep.

| Case | R/DV | Baseline F median/p95 | One pair F median/p95 | Baseline R median/p95 | One pair R median/p95 | Speedup F/R |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2x6x512:repeat | 16/32 | 72.04/72.18 | 63.32/63.50 | 71.98/72.09 | 63.34/63.58 | 1.138/1.136 |
| 2x6x512:repeat | 16/64 | 85.12/85.43 | 75.68/75.94 | 85.10/85.23 | 75.89/76.08 | 1.125/1.121 |
| 2x6x512:repeat | 32/32 | 72.60/72.85 | 64.37/64.44 | 72.78/72.89 | 64.36/64.44 | 1.128/1.131 |
| 2x6x512:repeat | 32/64 | 89.20/89.45 | 79.72/80.20 | 89.19/89.49 | 79.98/80.28 | 1.119/1.115 |
| 2x6x512:random | 16/32 | 11.65/11.80 | 10.89/10.95 | 11.63/11.74 | 11.02/11.13 | 1.070/1.056 |
| 2x6x512:random | 16/64 | 12.42/12.51 | 11.56/11.64 | 12.42/12.64 | 11.55/11.66 | 1.075/1.076 |
| 2x6x512:random | 32/32 | 11.91/12.20 | 11.14/11.20 | 11.89/12.06 | 11.12/11.25 | 1.070/1.070 |
| 2x6x512:random | 32/64 | 12.68/12.75 | 11.71/11.89 | 12.66/12.78 | 11.86/11.91 | 1.083/1.068 |
| 2x6x2048:repeat | 16/32 | 272.29/272.91 | 235.60/235.90 | 272.71/273.01 | 235.43/235.79 | 1.156/1.158 |
| 2x6x2048:repeat | 16/64 | 340.14/340.36 | 299.06/299.22 | 340.30/340.53 | 298.94/299.46 | 1.137/1.138 |
| 2x6x2048:repeat | 32/32 | 278.10/278.20 | 242.88/243.24 | 278.10/278.37 | 242.77/243.01 | 1.145/1.146 |
| 2x6x2048:repeat | 32/64 | 367.99/368.35 | 335.44/335.75 | 368.14/368.41 | 335.33/335.75 | 1.097/1.098 |
| 2x6x2048:random | 16/32 | 40.00/40.10 | 38.52/38.58 | 40.02/40.13 | 38.50/38.53 | 1.039/1.040 |
| 2x6x2048:random | 16/64 | 43.16/43.38 | 41.60/41.85 | 43.15/43.41 | 41.59/41.76 | 1.038/1.038 |
| 2x6x2048:random | 32/32 | 40.80/40.89 | 39.32/39.55 | 40.80/40.90 | 39.32/39.55 | 1.038/1.038 |
| 2x6x2048:random | 32/64 | 43.98/44.16 | 42.51/42.62 | 43.98/44.15 | 42.53/42.63 | 1.035/1.034 |

**Expanded numerical limitation, not hidden by aggregate relative L2:**
B2/H6/N512 repeat, R16/DV64, head 3 has FP64 tau -0.0159341329911.
Baseline gives -0.0159462038428 (forward order) and -0.0159454531968
(reverse); one-pair gives -0.0159455724061 and -0.0159461479634. Both violate
the unchanged pointwise `rtol=2e-4, atol=2e-6` gate (absolute errors
1.13e-5..1.21e-5, allowed about 5.19e-6). Their whole-vector relative L2 is only
8.4e-6..8.8e-6, demonstrating why that metric alone is insufficient. The
baseline failure reproduces in both orders on the identical plan/inputs. This
is an existing finite-precision event/metadata/reduction limitation; this work
does not claim an arbitrary-input FP64 guarantee or repair it. No tolerance,
oracle or skip was changed. Other repeated-label dimension rows pass this
pointwise check, including the actual R32/DV64 training dimensions. Random
B2/H6 rows have quadratic comparisons, but no independent dense random-label
FP64 oracle in this timing harness; the focused small random fixtures supply
that separate coverage.

### Compiled resources and stage cost

Actual B2/H6/N2048 repeated-label C64 launches; `single16/32/64` denote bucket
width, `multi64` the final readout. Reg/spill are Triton compiler counts;
shared is bytes allocated per CTA. HMMA/REDG/LDL/STL/LDS/LDSM/STS below are
**static SASS sites**, not executed warp operations or bytes. REDG may be F32x4.
Single-pair SASS contains both branches although each CTA executes only one.

| Variant | Stage | Stage median us | Registers/spills | Shared bytes | HMMA / global atomic sites | Local load/store sites | Shared load/store sites |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | single16 | 89.96 | 48/0 | 6144 | 10/4 | 0/0 | 27/15 |
| baseline | single32 | 10.59 | 134/0 | 12288 | 56/8 | 0/0 | 52/33 |
| baseline | single64 | 15.36 | 249/0 | 24576 | 112/16 | 0/0 | 96/40 |
| baseline | _chunk_summaries | 39.81 | 181/0 | 24576 | 96/0 | 0/0 | 64/46 |
| baseline | _summary_scan | 26.62 | 78/0 | 0 | 0/0 | 0/0 | 0/0 |
| baseline | multi64 | 91.54 | 255/2 | 32768 | 224/33 | 1/1 | 232/90 |
| tau_vectors | single16 | 112.29 | 56/0 | 6144 | 10/4 | 0/0 | 34/22 |
| tau_vectors | _summary_scan | 26.62 | 78/0 | 0 | 0/0 | 0/0 | 0/0 |
| tau_vectors | multi64 | 91.19 | 255/0 | 32768 | 224/33 | 0/0 | 234/92 |
| tau_compensated | single16 | 188.52 | 80/0 | 4096 | 32/4 | 0/0 | 51/47 |
| tau_compensated | _summary_scan | 26.62 | 78/0 | 0 | 0/0 | 0/0 | 0/0 |
| tau_compensated | multi64 | 156.41 | 255/74 | 36864 | 384/33 | 73/73 | 340/126 |
| tau_rowcol | single16 | 98.31 | 55/0 | 6144 | 10/4 | 0/0 | 28/17 |
| tau_rowcol | _summary_scan | 26.62 | 78/0 | 0 | 0/0 | 0/0 | 0/0 |
| tau_rowcol | multi64 | 93.39 | 255/0 | 32768 | 224/33 | 0/0 | 251/112 |
| scan32 | single16 | 90.17 | 48/0 | 6144 | 10/4 | 0/0 | 27/15 |
| scan32 | _summary_scan | 196.61 | 255/0 | 32768 | 0/0 | 0/0 | 560/160 |
| scan32 | multi64 | 91.74 | 255/2 | 32768 | 224/33 | 1/1 | 232/90 |
| persistent_tf32 | single16 | 89.95 | 48/0 | 6144 | 10/4 | 0/0 | 27/15 |
| persistent_tf32 | _query_backward | 803.58 | 255/308 | 16384 | 480/2 | 296/286 | 300/44 |
| persistent_tf32 | _key_backward | 1211.05 | 255/420 | 32768 | 528/6 | 447/390 | 362/78 |
| single_pair | single16 | 49.98 | 43/0 | 6144 | 10/7 | 0/0 | 29/17 |
| single_pair | single32 | 11.12 | 136/0 | 12288 | 56/11 | 0/0 | 54/35 |
| single_pair | single64 | 15.48 | 248/0 | 24576 | 112/19 | 0/0 | 98/42 |
| single_pair | _chunk_summaries | 39.01 | 181/0 | 24576 | 96/0 | 0/0 | 64/46 |
| single_pair | _summary_scan | 26.66 | 78/0 | 0 | 0/0 | 0/0 | 0/0 |
| single_pair | multi64 | 91.54 | 255/2 | 32768 | 224/33 | 1/1 | 232/90 |

Every audited kernel in these paired runs has **zero CALL sites and zero shared
FP32 atomic sites**. Summary and serial scan are spill-free. Parallel scan uses
32768 shared bytes versus zero for the serial scan; its scan stage rises from
about 26.6 to 196.8 us. Persistent query/key kernels report hundreds of spills.
The cheaper tau row/column reduction removes multi-readout spills in this case
but still slows the full call. These are direct counterexamples to equating
less source code, fewer spills, or a shallower recurrence with greater speed.

Initial screen artifacts are `screen.json` (15 samples), `second.json`, followed
by the two 31-sample paired runs above. A source-transform assertion initially
matched both local and boundary atomic lines; it stopped before GPU work and
was fixed by anchoring the newline. The initial single-pair multi kernel failed
compilation because row0 was undefined in an eliminated branch; initializing
the unused scalar fixed it. Those rows in `second.json` are compilation failures,
not timings. An initial final-test wrapper had an indentation typo and stopped
before collection; its log is retained separately. No failed run was silently
substituted into a speedup. All compiled/successful algorithm variants appear
in the paired table above; no R64-only result is used as adoption evidence.

### All variants on B2/H6/N512

The larger-head N512 grid is also paired for every algorithm (C64/W4, compact,
R32/DV64, 31 samples). This supplements the B1/H1 small-call table.

| Case | Variant | Forward median/p95 | Reverse median/p95 | Speedup F/R |
| --- | --- | ---: | ---: | ---: |
| 2x6x512:repeat | baseline | 88.90/89.41 | 89.22/89.55 | 1.000/1.000 |
| 2x6x512:repeat | tau_vectors | 95.08/95.39 | 95.15/95.40 | 0.935/0.938 |
| 2x6x512:repeat | tau_compensated | 130.96/131.23 | 131.05/131.24 | 0.679/0.681 |
| 2x6x512:repeat | tau_rowcol | 92.50/92.66 | 92.44/92.58 | 0.961/0.965 |
| 2x6x512:repeat | scan32 | 128.33/128.62 | 128.36/128.63 | 0.693/0.695 |
| 2x6x512:repeat | persistent_tf32 | 416.36/417.56 | 416.06/417.36 | 0.214/0.214 |
| 2x6x512:repeat | single_pair | 79.80/80.30 | 79.95/80.24 | 1.114/1.116 |
| 2x6x512:random | baseline | 12.66/12.74 | 12.65/12.70 | 1.000/1.000 |
| 2x6x512:random | tau_vectors | 13.66/13.74 | 13.68/13.72 | 0.927/0.925 |
| 2x6x512:random | tau_compensated | 16.70/16.77 | 16.73/16.81 | 0.759/0.756 |
| 2x6x512:random | tau_rowcol | 12.87/12.97 | 12.88/12.95 | 0.984/0.982 |
| 2x6x512:random | scan32 | 12.66/12.77 | 12.66/12.73 | 1.000/0.999 |
| 2x6x512:random | persistent_tf32 | 12.67/12.74 | 12.67/12.73 | 1.000/0.999 |
| 2x6x512:random | single_pair | 11.64/11.89 | 11.65/11.88 | 1.088/1.086 |

### Final installed source, ordinary/compact and C32/C64

`current` below loads the installed final source, rather than the generated
`single_pair` diagnostic. Its mixed/FP32 path is explicitly unchanged. Chunks
and warps remain caller choices/defaults; this table validates both existing
chunk choices and does not select a new launch policy. All cases use B2/H6,
R32/DV64 and W4; 31 samples in each order.

| N:labels | C | compact | Baseline F median/p95 | Final F median/p95 | Baseline R median/p95 | Final R median/p95 | Speedup F/R |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512:repeat | 32 | 0 | 93.24/93.37 | 83.81/84.01 | 93.15/93.47 | 83.84/83.98 | 1.113/1.111 |
| 512:repeat | 32 | 1 | 89.02/89.24 | 79.74/80.01 | 89.20/89.47 | 79.71/79.79 | 1.116/1.119 |
| 512:repeat | 64 | 0 | 93.30/93.59 | 84.08/84.47 | 93.06/93.52 | 83.90/84.20 | 1.110/1.109 |
| 512:repeat | 64 | 1 | 89.22/89.52 | 79.99/80.19 | 89.19/89.51 | 79.95/80.28 | 1.115/1.116 |
| 512:random | 32 | 0 | 14.47/14.72 | 13.68/13.76 | 14.47/14.60 | 13.68/13.77 | 1.058/1.058 |
| 512:random | 32 | 1 | 12.67/12.87 | 11.63/11.74 | 12.65/12.69 | 11.82/11.91 | 1.089/1.070 |
| 512:random | 64 | 0 | 14.47/14.67 | 13.68/13.72 | 14.47/14.69 | 13.66/13.72 | 1.057/1.059 |
| 512:random | 64 | 1 | 12.68/12.87 | 11.64/11.83 | 12.64/12.69 | 11.85/11.92 | 1.089/1.067 |
| 2048:repeat | 32 | 0 | 911.14/915.41 | 882.82/887.53 | 911.45/914.85 | 883.52/888.85 | 1.032/1.032 |
| 2048:repeat | 32 | 1 | 863.04/866.70 | 833.34/837.21 | 863.82/865.67 | 833.08/836.77 | 1.036/1.037 |
| 2048:repeat | 64 | 0 | 387.60/387.92 | 360.41/360.91 | 387.41/388.00 | 360.46/360.82 | 1.075/1.075 |
| 2048:repeat | 64 | 1 | 368.43/368.95 | 336.20/336.54 | 368.17/368.82 | 336.38/336.74 | 1.096/1.094 |
| 2048:random | 32 | 0 | 46.38/46.58 | 44.90/44.96 | 46.37/46.52 | 44.91/45.09 | 1.033/1.033 |
| 2048:random | 32 | 1 | 43.88/44.01 | 42.49/42.62 | 43.99/44.18 | 42.38/42.62 | 1.033/1.038 |
| 2048:random | 64 | 0 | 46.20/46.29 | 44.83/44.95 | 46.46/46.50 | 44.91/45.03 | 1.031/1.035 |
| 2048:random | 64 | 1 | 43.89/44.01 | 42.40/42.60 | 44.03/44.15 | 42.53/42.66 | 1.035/1.035 |

All final R32/DV64 repeated-label rows pass the unchanged pointwise FP64 tau
gate in both orders. Maximum final tau relative L2 is 7.42e-6. Maximum vector
relative L2 versus the paired baseline in the forward run is 6.14e-8; the small
differences include global atomic association. This does not supersede the
explicit R16/DV64 expanded-gate failure reported above.

Code/tests/diagnostics commit: **cf5a74378e767c8058273a575eb59f9f31c37c6b**.
This report is committed separately so it can name the exact implementation.

### Unchanged consumer launch defaults

The preceding W4 tables use the historical paired-benchmark setting. A final
check uses `--warps 0`, which passes `num_warps=None` and retains the actual
consumer rule: width16 singles use two warps; wider singles and C64 multi chunks
use four. There is no tuning or default change. C64, compact, R32/DV64, 31
samples per order:

| Case | Baseline F median/p95 | Final F median/p95 | Baseline R median/p95 | Final R median/p95 | Speedup F/R |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2x6x512:repeat | 83.04/83.32 | 75.91/76.21 | 83.08/83.33 | 75.85/76.16 | 1.094/1.095 |
| 2x6x512:random | 11.64/11.80 | 11.13/11.32 | 11.64/11.82 | 11.15/11.30 | 1.046/1.044 |
| 2x6x2048:repeat | 341.02/341.36 | 318.28/318.73 | 342.67/342.98 | 316.69/317.03 | 1.071/1.082 |
| 2x6x2048:random | 38.52/38.67 | 37.18/37.30 | 38.52/38.64 | 37.21/37.34 | 1.036/1.035 |

The actual-default repeated-label tau checks also pass. This extra measurement
confirms that adoption is not dependent on forcing the width16 bucket to W4.

## Validation, resource limits and remaining work

- Final installed source: **154 passed** = original 142 focused cases plus 12
  new event-multiplicity/padding/order/dtype regressions, 173.8 MiB peak PyTorch
  allocation. Existing graph/reset/packed-row, tau-sign, dimension and mixed
  storage gates are unchanged. The new tests explicitly retain the existing
  vector acceptance and `rtol=2e-4, atol=2e-6` tau gate.
- Unfiltered memcheck, synccheck and racecheck: **20 passed each** (the old eight
  long-stream/cancellation smoke cases plus the 12 new cases). Memcheck and
  synccheck: **0 errors**. Racecheck: **0 hazards, 0 errors, 0 warnings**. No
  sanitizer exception or unsupported-instruction warning was used. Peaks were
  8.2 MiB PyTorch allocation. These are sm120 smoke tests, not exhaustive proofs.
- Disabled planner in a fresh process: **3 passed**, 7.6 MiB peak. Production
  planner enablement and dispatcher defaults remain unchanged.
- Final paired source audit: 52 recorded kernel launches per order, **0 CALLs,
  0 shared atomics**, maximum 32768 shared bytes and two compiler spills in
  these R32 C32/C64 fixtures. The original C64 multi readout retains its spills;
  the one-pair single branch adds no spills. Other small launch grids can have
  different compiler counts (the B1/H1 C64 multi readout has ten).
- AST comparison against 8d47ebf confirms that the entire fallback chunk body,
  summary, scan, compensation helper and both host consumer classes are
  unchanged. No C++/CUDA translation unit changed, so rebuilding the native
  extension with build.py was unnecessary. Triton compilation is local.
- All benchmarks/tests limit the PyTorch allocator to 9% of the 31.44-GiB
  device (about 2.83 GiB, below the requested 4 GiB). Observed benchmark peak
  is **955.8 MiB**, including the dense FP64 repeated-label oracle. This is
  allocation, not CUDA context/driver memory. Native planner workers and
  MAX_JOBS/DISM_BUILD_JOBS are bounded at two; OMP_NUM_THREADS is one.
- Python warnings were existing requests-dependency, pytest import-rewrite or
  TorchScript-deprecation warnings. They are not sanitizer waivers. No tests
  downloaded data or launched external jobs. The optional native FP32 binary
  was not rebuilt, and other GPU architectures were not validated.

Adopt only the one-pair specialization. The precision-safe reuse candidate adds
more tensor work than it removes; row/column tau is slower despite fewer scalar
operations; the tested scan exchanges serial depth for excessive short-stream
padding/shared traffic; the tested persistent implementation is both slower and
numerically unacceptable. A scalar one-pair reduction deletes known zero work
without moving cancellation across event derivatives or extending state lifetime.
It improves all large supported-dimension grids tested, while leaving tiny-call
latency essentially unchanged.

Future worthwhile boundaries are concrete: length-bucketed scans when streams
are predominantly much longer than 32 chunks; a compensated persistent algorithm
with bounded live state and direct pairwise local tau; diagonal-tile plus
parallel off-diagonal rectangles if C² work becomes dominant; and cached scalar
coordinates only if memory counters establish a net benefit. None is advertised
as implemented or numerically validated. Fixing the baseline R16/DV64 near-zero
FP64 discrepancy is separate precision work. Event metadata, FP32 sums/atomics,
reduction order, and extreme magnitudes remain finite-precision limitations.

## Reproduction and artifact index

Run from this worktree root. Use fresh output names; all generated JSON, JIT
modules, cubins, SASS, logs and caches remain ignored under build/. The maintained
tool scripts contain the diagnostic transformations, including rejected ones.

```bash
export PY=/home/cicuvc/miniconda3/envs/blkw/bin/python
export PYTHONPATH="$PWD/dism_v3/python:$PWD/dism_v3/tests:$PWD/dism_v3/tools"
export TORCH_EXTENSIONS_DIR="$PWD/build/torch_extensions"
export TRITON_CACHE_DIR="$PWD/build/triton"
export MAX_JOBS=2 DISM_BUILD_JOBS=2 OMP_NUM_THREADS=1

"$PY" dism_v3/tools/bench_planner_event.py --baseline 8d47ebf \
  --output build/planner-algorithm/reproduce-forward.json \
  --cases 1x1x512:repeat 1x1x512:random 2x6x512:repeat 2x6x512:random \
          2x6x2048:repeat 2x6x2048:random \
  --shapes 32x64 --chunks 64 --warps 4 --compact 1 --samples 31 --references \
  --variants baseline tau_vectors tau_compensated tau_rowcol scan32 persistent_tf32 single_pair
# Repeat with reversed --variants and a new output file.

"$PY" dism_v3/tools/bench_planner_event.py --baseline 8d47ebf \
  --output build/planner-algorithm/reproduce-final.json \
  --cases 2x6x512:repeat 2x6x512:random 2x6x2048:repeat 2x6x2048:random \
  --shapes 32x64 --chunks 32 64 --warps 4 --compact 0 1 --samples 31 \
  --variants baseline current --references
# Repeat with --variants current baseline and a new output file.
# Dimension audit: --shapes 16x32 16x64 32x32 32x64 --chunks 64 --compact 1.
# Actual consumer defaults: --warps 0 --chunks 64 --compact 1.

"$PY" dism_v3/tools/check_planner_algorithm.py --variant tau_vectors -- -q \
  dism_v3/tests/test_hard_prefill_backward_fused.py \
  -k 'bf16_tau_cancellation_at_chunk_boundaries or reset_between_chunks'
# Repeat for tau_compensated, tau_rowcol, scan32, persistent_tf32.

# A process-local allocation cap also applies to direct pytest runs.
cat > build/planner-algorithm/reproduce-tests.py <<'PYTEST'
import sys, torch, pytest
torch.set_num_threads(1)
torch.cuda.set_per_process_memory_fraction(.09)
status = pytest.main(sys.argv[1:])
print('peak MiB', torch.cuda.max_memory_allocated()/2**20)
raise SystemExit(status)
PYTEST
"$PY" build/planner-algorithm/reproduce-tests.py -q \
  dism_v3/tests/test_hard_route_consistency.py \
  dism_v3/tests/test_hard_prefill_backward_fused.py \
  dism_v3/tests/test_hard_prefill_backward_launches.py \
  dism_v3/tests/test_hard_prefill_single_pair.py
for check in memcheck synccheck racecheck; do
  /usr/local/cuda/bin/compute-sanitizer --tool "$check" \
    --target-processes all --error-exitcode 86 \
    "$PY" build/planner-algorithm/reproduce-tests.py -q \
    dism_v3/tests/test_hard_training.py \
    dism_v3/tests/test_hard_prefill_backward_fused.py \
    dism_v3/tests/test_hard_prefill_single_pair.py \
    -k 'async_hard_long_stream_safety or bf16_tau_cancellation_at_chunk_boundaries or single_pair_event_multiplicity_and_fallback'
done
"$PY" build/planner-algorithm/reproduce-tests.py -q \
  dism_v3/tests/test_disabled_hard_planner.py
```

Measured artifacts in `build/planner-algorithm/`: `screen.json`, `second.json`,
`paired-forward.json`, `paired-reverse.json`, `n512-forward.json`,
`n512-reverse.json`, `dimensions-forward.json`, `dimensions-reverse.json`,
`final-forward.json`, `final-reverse.json`, `default-forward.json`,
`default-reverse.json`, `topology.json`, the corresponding
logs, `gate-*.log`, `single-pair-focused.log`, `final-focused.log`,
`final-focused-harness-error.log`, `memcheck.log`, `synccheck.log`, `racecheck.log`,
`disabled.log`, `ncu-metrics.txt`, `variants/`, `check-variants/` and `cubins/`.
JSON captures source/extension hashes, GPU status, exact topology and per-kernel
resources. The empirical tables above preserve the results in Git; generated
artifacts are deliberately not committed.
