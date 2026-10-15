"""Summarize durable evaluations; selection and oracle performance stay separate."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np
import pandas as pd


def collect(fs, uri, output):
    output = Path(output)
    root = fs._strip_protocol(uri).rstrip('/')
    paths = [p for p in fs.find(uri) if p.endswith('.json')]

    def fetch(path):
        dest = output / path.removeprefix(root + '/')
        dest.parent.mkdir(parents=True, exist_ok=True)
        data = fs.cat_file(path)
        dest.write_bytes(data)
        return json.loads(data)

    with ThreadPoolExecutor(max_workers=12) as pool:
        return list(pool.map(fetch, paths))


def bootstrap_mean(values, seed=2301, iterations=4000):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return dict(n=0, mean=float('nan'), low=float('nan'), high=float('nan'))
    draws = np.random.default_rng(seed).choice(values, (iterations, len(values))).mean(1)
    return dict(n=len(values), mean=float(values.mean()),
                low=float(np.quantile(draws, .025)), high=float(np.quantile(draws, .975)))


def summarize_audit(records, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    rows, bins = [], []
    for r in records:
        if r.get('status') != 'complete' or 'calibration' not in r:
            continue
        identity = {k: r[k] for k in ('pdb_id', 'weights', 'msa', 'tokens', 'seconds')}
        rows.extend({**identity, **m} for m in r['metrics'])
        bins.extend({**identity, **b} for b in r['calibration'])
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df.to_csv(output / 'contact_per_target.csv', index=False)
    summary = []
    for (weights, msa, group), frame in df.groupby(['weights', 'msa', 'group']):
        valid = frame[frame.pairs > 0]
        row = dict(weights=weights, msa=msa, group=group, targets=len(frame),
                   targets_with_pairs=len(valid), pairs=int(valid.pairs.sum()),
                   positives=int(valid.positives.sum()))
        if len(valid):
            row['pair_prevalence'] = row['positives'] / row['pairs']
            for metric in ('bce', 'brier'):
                row['pair_weighted_' + metric] = float(np.average(valid[metric], weights=valid.pairs))
            for metric in ('bce', 'brier', 'ap', 'precision_at_1', 'precision_at_4',
                           'precision_at_16', 'precision_at_64'):
                row.update({metric + '_' + k: v for k, v in bootstrap_mean(valid[metric]).items()})
        summary.append(row)
    pd.DataFrame(summary).to_csv(output / 'contact_summary.csv', index=False)
    calibration = pd.DataFrame(bins).groupby(['weights', 'msa', 'group', 'bin'], as_index=False)[
        ['count', 'probability_sum', 'positive_sum']].sum()
    calibration['mean_probability'] = calibration.probability_sum / calibration['count']
    calibration['observed_fraction'] = calibration.positive_sum / calibration['count']
    calibration.to_csv(output / 'contact_calibration.csv', index=False)
    paired = []
    for (msa, group), frame in df.groupby(['msa', 'group']):
        for metric in ('bce', 'brier', 'ap', 'precision_at_16'):
            wide = frame.pivot(index='pdb_id', columns='weights', values=metric)
            if {'raw', 'ema'} <= set(wide):
                paired.append(dict(msa=msa, group=group, metric=metric,
                    contrast='ema-minus-raw', **bootstrap_mean(wide.ema - wide.raw)))
    pd.DataFrame(paired).to_csv(output / 'contact_paired_differences.csv', index=False)
    return pd.DataFrame(summary)


def summarize_generation(records, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    rows, proposals = [], []
    for r in records:
        if r.get('status') != 'complete' or 'mode' not in r:
            continue
        identity = {k: r[k] for k in ('pdb_id', 'weights', 'msa', 'tokens', 'seconds', 'seed', 'mode')}
        for m in r['metrics']:
            rows.append({**identity, **{k: v for k, v in m.items() if k != 'missing_contact_proposals'}})
            proposals.extend({**identity, 'sample': m['sample'], **p}
                             for p in m.get('missing_contact_proposals', []))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df.to_csv(output / 'structure_per_sample.csv', index=False)
    pd.DataFrame(proposals).to_csv(output / 'discrepancy_proposals.csv', index=False)
    # Only confidence selects candidates for the primary report. Seeds and
    # samples remain available for separately labelled oracle analyses.
    keys = ['pdb_id', 'weights', 'msa', 'mode']
    selected = df.loc[df.groupby(keys).ranking.idxmax()].copy()
    selected.to_csv(output / 'structure_model_selected.csv', index=False)
    summaries = []
    metrics = ['observed_lddt', 'interface_lddt', 'pp_contact_recall', 'pl_contact_recall',
        'intra_contact_recall', 'pp_unrequested_recall', 'pl_unrequested_recall',
        'requested_contact_satisfaction', 'requested_no_contact_satisfaction',
        'interchain_close_atom_pairs', 'model_has_clash', 'ranking']
    for (weight, msa, mode), frame in selected.groupby(['weights', 'msa', 'mode']):
        for metric in metrics:
            summaries.append(dict(weights=weight, msa=msa, mode=mode, metric=metric,
                                  **bootstrap_mean(frame[metric])))
    pd.DataFrame(summaries).to_csv(output / 'structure_summary.csv', index=False)
    paired = []
    for (msa, mode), frame in selected.groupby(['msa', 'mode']):
        for metric in ('observed_lddt', 'interface_lddt'):
            wide = frame.pivot(index='pdb_id', columns='weights', values=metric)
            for a, b in [('ema', 'raw'), ('ema', 'protenix'), ('raw', 'protenix')]:
                if {a, b} <= set(wide):
                    paired.append(dict(msa=msa, mode=mode, metric=metric,
                        contrast=a + '-minus-' + b, **bootstrap_mean(wide[a] - wide[b])))
    pd.DataFrame(paired).to_csv(output / 'structure_paired_differences.csv', index=False)
    return pd.DataFrame(summaries)


def candidates_at_budget(metrics, budget):
    """Charge full batch cost; never rank a candidate by reference quality."""
    eligible = [m for m in metrics if m['available_seconds'] <= budget]
    if not eligible:
        return None
    best = max(eligible, key=lambda r: r['ranking'])
    return dict(candidates=len(eligible), selected=best,
                oracle_observed_lddt=max(r['observed_lddt'] for r in eligible))


def summarize_search(records, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in records:
        if r.get('status') != 'complete' or 'costs' not in r:
            continue
        # Both full adaptive horizons are reported. Within each horizon all
        # methods have exactly the same GPU-time allowance.
        for horizon in ('tree', 'unmask'):
            budget = r['costs'][horizon]
            for method in ('ordinary', 'tree', 'unmask'):
                candidates = candidates_at_budget([m for m in r['metrics'] if m['method'] == method], budget)
                identity = dict(pdb_id=r['pdb_id'], weights=r['weights'], seed=r['seed'],
                                horizon=horizon, budget_seconds=budget, method=method)
                if candidates is None:
                    rows.append({**identity, 'candidates': 0})
                    continue
                selected = candidates.pop('selected')
                rows.append({**identity, **candidates,
                    **{'selected_' + k: v for k, v in selected.items() if k not in {'method', 'parent'}}})
    df = pd.DataFrame(rows)
    df.to_csv(output / 'search_matched_time.csv', index=False)
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    records = [json.loads(p.read_text()) for p in args.records.rglob('*.json')]
    for func in (summarize_audit, summarize_generation, summarize_search):
        result = func(records, args.output)
        if len(result): print(result.to_string(index=False))


if __name__ == '__main__':
    main()
