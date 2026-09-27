# Delta gate copy/ABBA results (2026-09-26)

All 18 training runs completed: 3 methods × 2 seeds × 3 task/init settings.
Each run has 1000 optimizer updates; final evaluation uses 640 fixed sequences
(40,960 supervised tokens per seed). See README.md for exact protocol. Three
gate/target unit tests pass. Original external rl/dism.py is unchanged.

## Ordinary copy

Two-seed means. Delta gap = CE(hard QK, hard delta) − CE(hard QK, method-soft delta).
Positive means hardening worsens CE. Temperature uses T=.1, others T=1.

| Gate bias | Method | Hard CE | Hard accuracy | Delta gap | Keep fraction |
|---|---|---:|---:|---:|---:|
| 2 | temperature | .174584 | 96.713% | +.000001 | 99.614% |
| 2 | independent random | .160535 | 96.879% | +.008674 | 97.849% |
| 2 | shared row flag | .162611 | 96.909% | +.002215 | 92.308% |
| 0 | temperature | .190660 | 96.682% | −.0000001 | 95.275% |
| 0 | independent random | .160010 | 96.896% | −.000002 | 87.500% |
| 0 | shared row flag | .163045 | 96.617% | −.000088 | 91.951% |

The initial bias2 advantage of shared over independent hardening gap is not a
robust ranking: all bias0 ordinary-copy gaps are near zero. Lower hardening gap
does not imply better absolute CE. Temperature bias2 is almost always open.

## Double copy ABBA, bias0

A/B each32 independent random tokens. Only final BA supervised. All below use
fully hard QK and threshold-hard delta for inference.

| Method | Hard CE | Hard accuracy | Each segment after first8 | Delta gap | Keep fraction |
|---|---:|---:|---:|---:|---:|
| temperature | .323740 | 95.520% | 99.881% | +.003746 | 88.160% |
| independent random | .333742 | 95.265% | 99.592% | +.017283 | 75.940% |
| shared row flag | .324524 | 95.369% | 99.644% | +.014755 | 82.952% |

Temperature's gap with *unit-temperature* soft gate is .005540. Thus its smaller
gap is not solely the choice of final evaluation temperature, although the
training procedures and learned parameters differ. Two seeds are insufficient
to establish a general ranking. Per-seed delta gaps:

- temperature: .000036, .007456;
- independent random: .006752, .027815;
- shared: .005547, .023962.

Per-position curves: abba-positions.png. Errors concentrate at the start of
both repeated segments; after a few tokens accuracy approaches100%. There is
no obvious aggregate boundary-specific reset spike. Aggregate head averages
cannot exclude specialized behavior in individual heads.

## Does the trained model need reset?

Same checkpoint/batches, hard QK throughout; replace learned delta by0 at
inference (always continue the diagonal recurrence).

| Method | Learned-gate CE | Always-open CE | Learned accuracy | Always-open accuracy |
|---|---:|---:|---:|---:|
| temperature | .323740 | .318435 | 95.520% | 95.581% |
| independent random | .333742 | .318256 | 95.265% | 95.565% |
| shared | .324524 | .310648 | 95.369% | 95.660% |

All six individual checkpoints have non-increasing CE with always-open gates;
temperature seed0 is effectively unchanged. This experiment provides no evidence
of useful reset learning on ABBA. A plausible explanation is that mismatching
hard Q/K already breaks unwanted chains; this is a hypothesis, not a causal
conclusion from the ablation. The ablation is not a separately trained no-gate
baseline and does not show that gates are useless on other tasks.

## Takeaway

Sharing the actual logM row hard flag is implemented and trains successfully;
no need for a second random draw. Temperature gives the smallest hardening gap
here, while all methods solve most of copy/ABBA. Neither this gap nor end-stage
native-vs-hard agreement proves useful reset behavior: native random/shared
already uses all-hard gates at the final update. A stronger follow-up would use
conflicting repeated suffixes or explicit segment boundaries and compare a
separately trained always-open baseline, rather than only increasing ABBA steps.

Raw results: summary.json, abba-open-ablation.json, per-run evaluation.jsonl and
final.pt in results-v2/, results-copy-bias0/, results-abba-bias0/.
