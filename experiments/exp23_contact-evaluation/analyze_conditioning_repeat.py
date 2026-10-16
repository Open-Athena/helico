"""Compare sparse conditioning to an independent same-seed unconditioned repeat."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from helico.contact_evaluate import load_payload
from helico.contact_eval_analysis import bootstrap_mean
from helico.contact_eval_metrics import contact_distances
from helico.contact_state_analysis import read_artifact


def identity(record):
    return tuple(record[k] for k in ['pdb_id', 'seed', 'msa', 'weights'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--conditioning', type=Path, required=True)
    parser.add_argument('--repeat', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); torch.set_num_threads(2)
    root = Path(__file__).resolve().parent
    original = [json.loads(p.read_text()) for p in (args.conditioning/'generation').rglob('*.json')]
    repeated = [json.loads(p.read_text()) for p in (args.repeat/'generation').rglob('*.json')]
    if len(original) != 768 or len(repeated) != 96:
        raise ValueError('Both complete conditioning and repeat panels are required')
    if any(r['status'] != 'complete' for r in original + repeated):
        raise ValueError('Incomplete inference records')
    repeat = {identity(r): r for r in repeated}
    baseline = {identity(r): r for r in original if r['mode'] == 'unknown'}
    if len(repeat) != 96 or set(repeat) != set(baseline):
        raise ValueError('Repeat seeds, MSA modes or target identities differ')
    groups = pd.read_csv(root/'data/development_panel.csv').set_index('pdb_id').panel_group.to_dict()
    rows = []
    for record in original:
        if record['mode'] not in ['rescue_positive4', 'corrective_negative4']:
            continue
        key = identity(record)
        before = read_artifact(baseline[key], args.conditioning)
        conditioned = read_artifact(record, args.conditioning)
        null = read_artifact(repeat[key], args.repeat)
        state = conditioned['contact_state']; pairs = (state > 0).triu(1).nonzero()
        if not len(pairs):
            continue
        f = load_payload(args.features/(record['pdb_id']+'.pt.gz'))['features']
        positive = record['mode'] == 'rescue_positive4'
        distances = [np.array(contact_distances(v['all_coords'][0, 0], f['atom_to_token'],
                         pairs, f['atom_element_idx'] != 0)) for v in [before, conditioned, null]]
        correct = [d < 5 if positive else d >= 5 for d in distances]
        probabilities = [v['contact_probs'][0, pairs[:, 0], pairs[:, 1]].numpy()
                         for v in [before, conditioned, null]]
        if not positive:
            probabilities = [1-p for p in probabilities]
        row = dict(pdb_id=key[0], seed=key[1], msa=key[2], weights=key[3],
                   group=groups[key[0]], mode=record['mode'], pairs=len(pairs))
        for name, values in zip(['baseline', 'conditioned', 'repeat'], zip(correct, probabilities)):
            row[name+'_satisfaction'] = float(values[0].mean())
            row[name+'_probability'] = float(values[1].mean())
        row.update(conditioned_minus_repeat_satisfaction=float((correct[1].astype(float)-correct[2]).mean()),
                   conditioned_minus_repeat_probability=float((probabilities[1]-probabilities[2]).mean()),
                   conditioned_minus_repeat_toward_request_angstrom=float(
                       ((distances[2]-distances[1]) if positive else (distances[1]-distances[2])).mean()),
                   conditioned_minus_repeat_lddt=(record['metrics'][0]['observed_lddt']-
                                                  repeat[key]['metrics'][0]['observed_lddt']))
        rows.append(row)
    frame = pd.DataFrame(rows); args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output/'repeat_control_per_case.csv', index=False)
    metrics = [k for k in frame if k.startswith(('baseline_', 'conditioned_', 'repeat_'))]
    targets = frame.groupby(['pdb_id', 'group', 'msa', 'mode'], as_index=False)[metrics].mean()
    targets.to_csv(args.output/'repeat_control_per_target.csv', index=False)
    summary = [dict(group=g, msa=m, mode=s, metric=metric, **bootstrap_mean(d[metric]))
               for (g, m, s), d in targets.groupby(['group', 'msa', 'mode']) for metric in metrics]
    result = pd.DataFrame(summary)
    result.to_csv(args.output/'repeat_control_summary.csv', index=False)
    print(result[result.msa & result['mode'].eq('rescue_positive4')].to_string(index=False))


if __name__ == '__main__':
    main()
