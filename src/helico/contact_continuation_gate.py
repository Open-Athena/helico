"""Predeclared development gates for focused continuation; no FoldBench access."""
from __future__ import annotations

from collections import defaultdict
import math
import random
import statistics


def mean(values):
    return statistics.mean(values) if values else None


def interval(values, seed=2401, repeats=4000):
    if not values:
        return None
    rng = random.Random(seed)
    draws = sorted(mean(rng.choices(values, k=len(values))) for _ in range(repeats))
    return [draws[int(.025 * repeats)], draws[int(.975 * repeats)]]


def expected_keys(config):
    return {(weight, pdb, offset, mode)
        for weight in config['weights'] for pdb, group in config['target_groups'].items()
        for offset in config['seeds']
        for mode in (['unknown', 'repeat', 'rescue1', 'rescue4', 'rescue16', 'negative4']
                     if group in {'intra', 'pp', 'pl'} else ['unknown', 'repeat'])}


def fetch_rows(fs, config):
    import json
    from concurrent.futures import ThreadPoolExecutor
    rows = {}
    def load(key):
        weight, pdb, offset, mode = key
        uri = config['output_uri'] + f'/focused/{weight}/{pdb}/s{offset}-{mode}.json'
        row = json.loads(fs.cat_file(uri))
        if row['status'] != 'complete' or len(row['metrics']) != 1:
            raise ValueError('Missing or incomplete development result: ' + uri)
        return key, row
    with ThreadPoolExecutor(max_workers=16) as pool:
        rows.update(pool.map(load, sorted(expected_keys(config))))
    return rows


def assess(parent, candidate, config, thresholds, step):
    """Average seeds/modes within targets, then bootstrap whole targets.

    Undefined constraint metrics are excluded only when the count is zero.
    Missing/nonfinite metrics for an actual measured quantity fail closed.
    """
    if set(candidate) != expected_keys(config):
        raise ValueError('Development panel incomplete')
    reports = {}
    for weight in config['weights']:
        retention, groups, steering, improvements, steer_groups = (defaultdict(list) for _ in range(5))
        for pdb, group in config['target_groups'].items():
            per_metric = defaultdict(list)
            s, improve = [], []
            for offset in config['seeds']:
                for mode in ('unknown', 'repeat'):
                    new = candidate[weight, pdb, offset, mode]['metrics'][0]
                    old = parent['ema', pdb, offset, mode]['metrics'][0]
                    if not math.isfinite(old['observed_lddt']) or not math.isfinite(new['observed_lddt']):
                        raise ValueError('Nonfinite structure quality')
                    for metric in ('observed_lddt', 'interface_lddt'):
                        if math.isfinite(old[metric]):
                            if not math.isfinite(new[metric]):
                                raise ValueError(f'Nonfinite {metric}: {pdb}')
                            per_metric[metric].append(new[metric] - old[metric])
                    per_metric['clash_fraction'].append(float(new['interchain_close_atom_pairs'] > 0)
                                                       - float(old['interchain_close_atom_pairs'] > 0))
                if group in {'intra', 'pp', 'pl'}:
                    for mode in ('rescue1', 'rescue4', 'rescue16'):
                        new = candidate[weight, pdb, offset, mode]['metrics'][0]
                        old = parent['ema', pdb, offset, mode]['metrics'][0]
                        if new['requested_contact_count'] != old['requested_contact_count']:
                            raise ValueError('Frozen constraints changed')
                        if not old['requested_contact_count']:
                            continue
                        value = new['change_requested_contact_satisfaction']
                        previous = old['change_requested_contact_satisfaction']
                        if not math.isfinite(value) or not math.isfinite(previous):
                            raise ValueError('Nonfinite sparse steering')
                        s.append(value); improve.append(value - previous)
            for metric, values in per_metric.items():
                retention[metric].append(mean(values))
                groups[group + '/' + metric].append(mean(values))
            if s:
                steering[pdb].append(mean(s)); improvements[pdb].append(mean(improve))
                steer_groups[group].append(mean(s))
        r = {key: dict(mean=mean(values), ci95=interval(values), n=len(values)) for key, values in retention.items()}
        gr = {key: dict(mean=mean(values), n=len(values)) for key, values in groups.items()}
        s = [mean(values) for values in steering.values()]
        imp = [mean(values) for values in improvements.values()]
        structural_pass = (r['observed_lddt']['mean'] >= thresholds['min_lddt_delta'] and
                           r['interface_lddt']['mean'] >= thresholds['min_interface_delta'] and
                           r['clash_fraction']['mean'] <= thresholds['max_clash_delta'])
        ci = interval(s)
        steer_pass = bool(s and mean(s) >= thresholds['min_steering'] and
                          mean(imp) >= thresholds['min_steering_improvement'] and ci[0] >= 0 and
                          all(len(steer_groups[g]) >= 4 and mean(steer_groups[g]) >= -.05
                              for g in ('intra', 'pp', 'pl')))
        reports[weight] = dict(retention=r, retention_groups=gr, retention_pass=structural_pass,
            sparse=dict(mean=mean(s), ci95=ci, improvement=mean(imp), improvement_ci95=interval(imp), n=len(s)),
            sparse_groups={g:dict(mean=mean(v),n=len(v)) for g,v in steer_groups.items()}, steering_pass=steer_pass)
    retain = all(v['retention_pass'] for v in reports.values())
    proceed = retain and (step == 500 or any(v['steering_pass'] for v in reports.values()))
    return dict(step=step, decision='continue' if proceed else 'stop', weights=reports,
                rationale='500-update retention gate' if step == 500 else 'retention and sparse-steering gate')
