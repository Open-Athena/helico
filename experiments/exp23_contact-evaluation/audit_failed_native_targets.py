"""Diagnose official scorer failures using native-versus-native inputs.

Run with PYTHONPATH=src in the pinned OpenStructure environment. This audit
never replaces model scores or removes a benchmark target from denominators.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

from helico.contact_foldbench_score import score_prediction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--foldbench', type=Path, required=True)
    parser.add_argument('--references', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--summary', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.foldbench))
    files = sorted(args.results.glob('*/scores/per_prediction.csv'))
    scores = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    failed = scores[scores.status.ne('complete')]
    ids = sorted(failed.pdb_id.unique())
    targets = pd.concat([pd.read_csv(p).assign(category=p.stem)
                         for p in sorted((args.foldbench/'targets').glob('*.csv'))],
                        ignore_index=True)
    targets['target_index'] = range(len(targets))
    tasks = []
    for pid in ids:
        reference = args.references/(pid+'.cif')
        row = dict(model='native', pdb_id=pid, seed=0, sample=0, ranking_score=1.,
                   prediction_path=str(reference),
                   prediction_sha256=hashlib.sha256(reference.read_bytes()).hexdigest())
        metadata = [{k: v for k, v in r.items() if pd.notna(v)}
                    for r in targets[targets.pdb_id.eq(pid)].to_dict('records')]
        tasks.append((row, metadata, str(args.references), str(args.output),
                      str(Path(sys.executable).parent/'ost'), 600, str(args.foldbench)))
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = [row for group in pool.map(score_prediction, tasks) for row in group]
    result = pd.DataFrame(rows)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.summary, index=False)
    receipt = dict(scored_stage_files=len(files), failed_prediction_target_rows=len(failed),
                   tested_assemblies=len(ids), tested_native_targets=len(result),
                   native_scoring_failures=int(result.status.ne('complete').sum()),
                   policy='Diagnostic only; original model scores and denominators unchanged.')
    args.summary.with_suffix('.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
