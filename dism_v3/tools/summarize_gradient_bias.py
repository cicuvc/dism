"""Summarize paired observations without treating trajectory steps as IID."""
import argparse
from collections import defaultdict
import json
import math
import statistics


def quantile(values, p):
    ordered = sorted(values)
    index = (len(ordered)-1)*p
    lo, hi = math.floor(index), math.ceil(index)
    return ordered[lo] + (ordered[hi]-ordered[lo])*(index-lo)


def pooled(rows):
    xx = sum(row['xx'] for row in rows)
    yy = sum(row['yy'] for row in rows)
    xy = sum(row['xy'] for row in rows)
    return dict(cosine=xy/math.sqrt(xx*yy) if xx*yy else None,
                norm_ratio=math.sqrt(xx/yy) if yy else None,
                gain_bias=xy/yy-1 if yy else None)


def summarize(rows):
    valid = [row for row in rows if row['cosine'] is not None]
    result = dict(count=len(rows), zero_reference_count=len(rows)-len(valid), pooled=pooled(rows),
                  rms_reference_norm=math.sqrt(sum(row['yy'] for row in rows)/len(rows)),
                  rms_error_norm=math.sqrt(sum(max(0., row['xx']+row['yy']-2*row['xy'])
                                               for row in rows)/len(rows)))
    if not valid:
        result['max_actual_norm'] = max(row['actual_norm'] for row in rows)
        return result
    cosine = [row['cosine'] for row in valid]
    ratio = [row['norm_ratio'] for row in valid]
    result.update(cosine_min=min(cosine), cosine_p05=quantile(cosine, .05),
                  cosine_median=statistics.median(cosine),
                  norm_ratio_mean=statistics.mean(ratio), norm_ratio_median=statistics.median(ratio),
                  norm_ratio_p05=quantile(ratio, .05), norm_ratio_p95=quantile(ratio, .95),
                  norm_larger_fraction=sum(x > 1 for x in ratio)/len(ratio))
    # Independent unit is a random seed, not a correlated optimizer step.
    per_seed = defaultdict(list)
    for row in rows:
        per_seed[row['seed']].append(row)
    result['by_seed'] = {seed: pooled(group) for seed, group in per_seed.items()}
    threshold = statistics.median(row['oracle_norm'] for row in valid)*.01
    active = [row for row in valid if row['oracle_norm'] >= threshold]
    result['above_one_percent_median_norm'] = dict(
        count=len(active), threshold=threshold,
        cosine_min=min(row['cosine'] for row in active), pooled=pooled(active))
    return result


def error_drift(steps, total):
    """Temporal mean of the full error vector, not mean scalar norm error.

    total contains dot products of SUMMED gradient tensors from the evaluator.
    Never substitute sums of per-step dot products for these quantities.
    """
    count = len(steps)
    error_energy = sum(max(0., row['xx'] + row['yy'] - 2*row['xy']) for row in steps)
    reference_energy = sum(row['yy'] for row in steps)
    sum_error_sq = max(0., total['xx'] + total['yy'] - 2*total['xy'])
    norm_sum_error = math.sqrt(sum_error_sq)
    reference_path_length = sum(row['oracle_norm'] for row in steps)
    return dict(
        steps=count,
        mean_error_vector_norm=norm_sum_error/count,
        rms_step_error_norm=math.sqrt(error_energy/count),
        rms_reference_norm=math.sqrt(reference_energy/count),
        # ~1 only under additional independent zero-mean error assumptions.
        coherence=math.sqrt(sum_error_sq/error_energy) if error_energy else None,
        mean_error_over_rms_error=math.sqrt(sum_error_sq/(count*error_energy)) if error_energy else None,
        mean_error_over_rms_reference=math.sqrt(sum_error_sq/(count*reference_energy)) if reference_energy else None,
        cumulative_relative_error=math.sqrt(sum_error_sq/total['yy']) if total['yy'] else None,
        cumulative_error_over_reference_path=norm_sum_error/reference_path_length if reference_path_length else None,
        error_cosine_with_cumulative_reference=(total['xy']-total['yy'])/math.sqrt(sum_error_sq*total['yy'])
        if sum_error_sq*total['yy'] else None,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input')
    args = parser.parse_args()
    with open(args.input) as handle:
        rows = [json.loads(line) for line in handle]
    groups, detailed, windows = defaultdict(list), defaultdict(list), defaultdict(list)
    temporal, totals = defaultdict(list), {}
    for row in rows:
        key = (row['source'], row['mode'], row['gradient'])
        groups[key].append(row)
        if row['source'] == 'iid_core':
            detailed[key + (row['n'], row['tau'])].append(row)
        elif row['source'] == 'trajectory':
            windows[key + (row['step']//32,)].append(row)
            temporal[(row['seed'], row['mode'], row['gradient'])].append(row)
        elif row['source'] == 'trajectory_sum':
            totals[(row['seed'], row['mode'], row['gradient'])] = row
    report = dict(groups={'/'.join(map(str, k)): summarize(v) for k, v in groups.items()},
                  iid_by_shape_tau={'/'.join(map(str, k)): summarize(v) for k, v in detailed.items()},
                  trajectory_windows={'/'.join(map(str, k)): summarize(v) for k, v in windows.items()},
                  error_drift={'/'.join(map(str, k)): error_drift(v, totals[k])
                               for k, v in temporal.items() if k in totals})
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
