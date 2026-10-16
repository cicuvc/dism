"""Aggregate P2/P3 over digit targets, preserving head slots and source pairing."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from niah_circuit_probe import atomic_json


def summarize(directories, p1):
    summary = dict(targets="five digit targets only (offsets 1..5); blank target excluded", arms={})
    hashes = set()
    for arm in ("old", "restored", "gqa2"):
        baseline = json.loads((p1 / f"{arm}.json").read_text())
        hashes.add(baseline["suite_sha256"])
        groups = defaultdict(list)
        seen = set()
        errors = []
        score_errors = []
        for directory in directories:
            data = json.loads((directory / f"{arm}.json").read_text())
            hashes.add(data["suite_sha256"])
            assert data["buffers_unchanged"]
            for cell in data["cells"]:
                identity = cell["context"], cell["depth"], cell["trial"]
                assert identity not in seen
                seen.add(identity)
                b = next(c for c in baseline["cells"] if (c["context"], c["depth"]) == identity[:2])
                score_errors.append(abs(cell["score"]["nll"] - b["measurements"]["baseline"]["present"][cell["trial"]]["nll"]))
                for row in cell["rows"]:
                    errors.append(row["reconstruction_rel_l2"])
                    if row["target_offset"] > 0:
                        groups[(cell["context"], cell["depth"], row["layer"], row["slot"])].append(row)
        heads = []
        for (context, depth, layer, slot), rows in sorted(groups.items()):
            correct = [r["needle_pairs"][r["target_offset"]] for r in rows]
            def avg(key):
                return float(np.mean([r[key] for r in rows]))
            heads.append(dict(context=context, depth=depth, layer=layer, slot=slot,
                              base_head=rows[0]["base_head"], channel=rows[0]["channel"],
                              digit_queries=len(rows), tau=rows[0]["tau"],
                              needle_mass=avg("needle_mass"), other_mass=avg("other_mass"), fallback_mass=avg("fallback_mass"),
                              needle_contribution_norm=avg("needle_contribution_norm"), core_norm=avg("core_norm"),
                              any_needle_gate_rate=float(np.mean([any(p["chain_length"] > 0 for p in r["needle_pairs"]) for r in rows])),
                              max_needle_chain_mean=float(np.mean([max(p["chain_length"] for p in r["needle_pairs"]) for r in rows])),
                              source_digit_gate_rate=float(np.mean([p["chain_length"] > 0 for p in correct])),
                              source_digit_chain_mean=float(np.mean([p["chain_length"] for p in correct])),
                              source_digit_mass=float(np.mean([p["mass"] for p in correct])),
                              source_digit_signed_mean=float(np.mean([p["signed_readout"] for p in correct])),
                              source_digit_signed_abs_mean=float(np.mean([abs(p["signed_readout"]) for p in correct])),
                              source_digit_negative_rate=float(np.mean([p["signed_readout"] < 0 for p in correct]))))
        summary["arms"][arm] = dict(heads=heads, trials_total=len(seen),
                                    reconstruction_relative_p50_p95_max=np.quantile(errors,[.5,.95,1.]).tolist(),
                                    max_baseline_nll_drift=max(score_errors))
    assert len(hashes) == 1
    summary["suite_sha256"] = hashes.pop()
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directories", type=Path, nargs="+", required=True)
    p.add_argument("--p1", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    atomic_json(a.out, summarize(a.directories, a.p1))


if __name__ == "__main__":
    main()
