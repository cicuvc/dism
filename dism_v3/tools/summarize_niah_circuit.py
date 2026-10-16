"""Summarize paired P1 deltas with trial-bootstrap confidence intervals."""
import argparse
import json
from pathlib import Path

import numpy as np

from niah_circuit_probe import atomic_json


def estimate(values):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(1729)
    boot = values[rng.integers(len(values), size=(10000, len(values)))].mean(1)
    return dict(mean=float(values.mean()), ci95=np.quantile(boot, [.025, .975]).tolist(), n=len(values))


def summarize(directory):
    data = {arm: json.loads((directory / f"{arm}.json").read_text()) for arm in ("old", "restored", "gqa2")}
    assert len({d["suite_sha256"] for d in data.values()}) == 1
    assert all(d["buffers_unchanged"] for d in data.values())
    result = dict(suite_sha256=data["old"]["suite_sha256"],
                  uncertainty="paired trial bootstrap, 10000 resamples, seed 1729; pointwise, not multiplicity corrected", cells=[])
    for index, old_cell in enumerate(data["old"]["cells"]):
        cell = dict(context=old_cell["context"], depth=old_cell["depth"], arms={}, comparisons={})
        deltas = {}
        baselines = {}
        for arm, d in data.items():
            raw = d["cells"][index]
            assert (raw["context"], raw["depth"]) == (cell["context"], cell["depth"])
            base = raw["measurements"]["baseline"]
            stats = dict(baseline={}, ablations={})
            baselines[arm] = np.array([t["nll"] for t in base["present"]])
            for condition in ("present", "absent"):
                stats["baseline"][condition] = {metric: estimate([t[metric] for t in base[condition]]) for metric in ("nll", "tok_acc", "all_correct")}
            deltas[arm] = {}
            for key, measurement in raw["measurements"].items():
                if key == "baseline":
                    continue
                deltas[arm][key] = np.array([t["nll"] for t in measurement["present"]])-baselines[arm]
                stats["ablations"][key] = {}
                for condition in ("present", "absent"):
                    assert [t["trial"] for t in measurement[condition]] == [t["trial"] for t in base[condition]]
                    stats["ablations"][key][condition] = {
                        metric: estimate([x[metric]-y[metric] for x, y in zip(measurement[condition], base[condition])])
                        for metric in ("nll", "tok_acc")}
            cell["arms"][arm] = stats
        for arm in ("restored", "gqa2"):
            cell["comparisons"][f"{arm}-old"] = dict(
                present_nll=estimate(baselines[arm]-baselines["old"]),
                ablation_delta_difference={key: estimate(deltas[arm][key]-deltas["old"][key]) for key in deltas["old"]})
        result["cells"].append(cell)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    a = p.parse_args()
    result = summarize(a.directory)
    atomic_json(a.directory / "summary.json", result)
    for cell in result["cells"]:
        print('\n', cell["context"], cell["depth"])
        for arm, stats in cell["arms"].items():
            print(arm, 'baseline', round(stats['baseline']['present']['nll']['mean'], 4),
                  'ablations', {k: round(v['present']['nll']['mean'], 4) for k, v in stats['ablations'].items()})
        print('paired comparisons', cell['comparisons'])


if __name__ == "__main__":
    main()
