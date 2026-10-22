"""Measure the contact head's response on already-generated oracle constraints."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import pandas as pd
import torch

from helico.contact_eval_analysis import bootstrap_mean


def artifact(path, record):
    if hashlib.sha256(path.read_bytes()).hexdigest() != record['artifact_sha256']:
        raise ValueError('Conditioning artifact checksum mismatch')
    with gzip.open(path, 'rb') as handle:
        return torch.load(handle, map_location='cpu', weights_only=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--panel', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); torch.set_num_threads(2)
    groups = pd.read_csv(args.panel).set_index('pdb_id').panel_group.to_dict()
    rows = []
    for folder in sorted((args.records / 'generation/ema').iterdir()):
        if not folder.is_dir():
            continue
        for base_path in sorted(folder.glob('*-unknown.json')):
            base_record = json.loads(base_path.read_text())
            base = artifact(args.records / base_record['artifact'], base_record)
            prefix = base_path.name.removesuffix('unknown.json')
            for path in sorted(folder.glob(prefix + '*.json')):
                record = json.loads(path.read_text())
                if record['mode'] == 'unknown':
                    continue
                value = artifact(args.records / record['artifact'], record)
                state = value['contact_state']; n = state.shape[0]
                for label, name in [(2, 'contact'), (1, 'no_contact')]:
                    mask = (state == label).triu(1)
                    if not mask.any():
                        continue
                    before = base['contact_probs'][0, :n, :n][mask]
                    after = value['contact_probs'][0, :n, :n][mask]
                    if label == 1:
                        before, after = 1 - before, 1 - after
                    rows.append(dict(pdb_id=record['pdb_id'], group=groups[record['pdb_id']],
                        msa=record['msa'], seed=record['seed'], mode=record['mode'],
                        requested_label=name, pairs=int(mask.sum()),
                        before=float(before.mean()), after=float(after.mean()),
                        change=float((after-before).mean())))
    frame = pd.DataFrame(rows)
    if frame.pdb_id.nunique() != 24:
        raise ValueError('Expected the complete 24-target conditioning panel')
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / 'conditioning_head_per_case.csv', index=False)
    keys = ['group', 'msa', 'mode', 'requested_label']
    targets = frame.groupby(keys + ['pdb_id'], as_index=False)[['before', 'after', 'change']].mean()
    summary = []
    for identity, subset in targets.groupby(keys):
        for metric in ['before', 'after', 'change']:
            summary.append(dict(zip(keys, identity)) | dict(metric=metric) |
                           bootstrap_mean(subset[metric]))
    result = pd.DataFrame(summary)
    result.to_csv(args.output / 'conditioning_head_summary.csv', index=False)
    print(result[(result.msa) & result['mode'].isin(
        ['rescue_positive4', 'corrective_negative4'])].to_string(index=False))


if __name__ == '__main__':
    main()
