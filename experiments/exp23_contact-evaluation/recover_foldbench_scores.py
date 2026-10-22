"""Revisit failed official scores after inference, retaining original evidence."""
import argparse
import json
import os
from pathlib import Path
import subprocess

import pandas as pd


def recover(stage, foldbench, references, ost_python, workers=8):
    stage = Path(stage); marker = stage / 'done.json'
    done = json.loads(marker.read_text())
    if done.get('scoring_recovery_schema') == 1:
        return done
    work = Path(__file__).resolve().parents[2]
    env = {**os.environ, 'PYTHONPATH': str(work / 'src'),
           'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    with (stage / 'scoring-recovery.log').open('w') as handle:
        subprocess.run([str(ost_python), '-m', 'helico.contact_foldbench_score',
            '--foldbench-root', str(foldbench), '--predictions', done['predictions'],
            '--references', str(references), '--output', str(stage / 'scores'),
            '--workers', str(workers)], cwd=work, env=env, stdout=handle,
            stderr=subprocess.STDOUT, check=True)
    scores = pd.read_csv(done['scores'])
    if len(scores) != done['target_scores']:
        raise ValueError('Score recovery changed the target denominator')
    done['failed_target_scores_before_recovery'] = done['failed_target_scores']
    done['failed_target_scores'] = int(scores.status.ne('complete').sum())
    done['primary_scores_recovered'] = (done['failed_target_scores_before_recovery'] -
                                        done['failed_target_scores'])
    done['scoring_recovery_schema'] = 1
    done['scoring_source_commit'] = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=work, text=True).strip()
    pending = marker.with_suffix('.json.pending')
    pending.write_text(json.dumps(done, indent=2) + '\n'); pending.replace(marker)
    return done


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--foldbench', type=Path, required=True)
    parser.add_argument('--references', type=Path, required=True)
    parser.add_argument('--ost-python', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(recover(args.stage, args.foldbench, args.references,
                             args.ost_python, args.workers)))
