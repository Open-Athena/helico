"""Summarize one checksum-verified corrected validation stage."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import torch

from helico.contact_eval_analysis import (
    bootstrap_mean, summarize_conditioning, summarize_generation, summarize_search,
)
from helico.contact_state_analysis import (
    candidate_diversity, requested_distance_changes, score_alternative_references,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--stage', required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = Path(__file__).resolve().parent
    output = root / 'data' / args.stage
    output.mkdir(parents=True, exist_ok=True)
    result = json.loads((args.records / 'result.json').read_text())
    if result['status'] != 'complete':
        raise ValueError('Completed stage required')
    records = [json.loads(p.read_text()) for p in args.records.rglob('*.json')]
    records = [r for r in records if 'artifact' in r]
    config = root.parents[1] / 'configs/eval' / ('contact-' + args.stage + '.json')
    expected_config = hashlib.sha256(config.read_bytes()).hexdigest()
    if any(r.get('config_sha256') != expected_config for r in records):
        raise ValueError('Records do not match the immutable stage configuration')
    expected = {'development-v2': 144, 'conditioning-v2': 768, 'search-v2': 48,
                'contact-hypotheses-v2': 54, 'validation-structures-v3': 768,
                'alternative-states-v2': 8}
    if len(records) != expected[args.stage] or any(r['status'] != 'complete' for r in records):
        raise ValueError(f'Incomplete target records: {len(records)}')
    (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    panel = pd.read_csv(root / 'data/development_panel.csv')
    if args.stage == 'contact-hypotheses-v2':
        subprocess.run([sys.executable, str(root / 'analyze_contact_hypotheses.py'),
                        '--records', str(args.records), '--features', str(args.features),
                        '--output', str(output)], check=True)
    elif args.stage.startswith(('development-', 'validation-structures-')):
        summarize_generation(records, output)
    elif args.stage.startswith('conditioning-'):
        summarize_generation(records, output)
        summarize_conditioning(records, panel.to_dict('records'), output)
        distances = requested_distance_changes(records, args.records, args.features, output)
        distances['group'] = distances.pdb_id.map(panel.set_index('pdb_id').panel_group)
        targets = distances.groupby(['pdb_id', 'group', 'msa', 'mode'], as_index=False)[
            'toward_request_angstrom'].mean()
        rows = [dict(group=g, msa=m, mode=s, **bootstrap_mean(d.toward_request_angstrom))
                for (g, m, s), d in targets.groupby(['group', 'msa', 'mode'])]
        pd.DataFrame(rows).to_csv(output / 'conditioning_distance_summary.csv', index=False)
        subprocess.run([sys.executable, str(root / 'analyze_conditioning_head.py'),
                        '--records', str(args.records), '--panel', str(root / 'data/development_panel.csv'),
                        '--output', str(output)], check=True)
    else:
        summarize_search(records, output)
        candidate_diversity(records, args.records, args.features, output)
        if args.stage.startswith('alternative-states-'):
            alternative = json.loads((root / 'data/alternative_state_panel.json').read_text())
            score_alternative_references(records, args.records, args.features, alternative, output)
    print('ANALYSIS COMPLETE', args.stage, flush=True)


if __name__ == '__main__':
    main()
